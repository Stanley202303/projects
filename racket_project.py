"""
Live 3D cuboid visualizer for the matching Pico2W_IMU_Cube.ino sketch.

Orientation is estimated with quaternion gyro integration plus a conservative
complementary correction from accelerometer tilt and tilt-compensated magnetic
heading.

Install the two desktop dependencies:
    python -m pip install pygame PyOpenGL

Run:
    python visualizer.py

The Pico and computer must be on the same local network. The Arduino sketch
sends UDP packets to this computer on port 5005.

Controls:
    R       Make the current orientation the display zero
    C       Start or finish in-memory magnetometer calibration
    L       Clear the current magnetometer calibration
    Escape Close the visualizer

No extra project files are required. Magnetometer calibration remains in memory
and is intentionally not written to a third file.
"""

import json
import math
import shutil
import socket
import subprocess
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    import pygame
    from pygame.locals import (
        DOUBLEBUF,
        KEYDOWN,
        OPENGL,
        QUIT,
        RESIZABLE,
        VIDEORESIZE,
    )
    from OpenGL.GL import (
        GL_COLOR_BUFFER_BIT,
        GL_DEPTH_BUFFER_BIT,
        GL_DEPTH_TEST,
        GL_BLEND,
        GL_LINES,
        GL_LINE_LOOP,
        GL_LINE_STRIP,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_QUADS,
        GL_RGB,
        GL_RGBA,
        GL_ONE_MINUS_SRC_ALPHA,
        GL_SRC_ALPHA,
        GL_UNSIGNED_BYTE,
        glBegin,
        glBlendFunc,
        glClear,
        glClearColor,
        glColor3f,
        glDisable,
        glDrawPixels,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMultMatrixf,
        glMatrixMode,
        glOrtho,
        glPopMatrix,
        glPushMatrix,
        glRasterPos2f,
        glReadPixels,
        glRotatef,
        glVertex3f,
        glViewport,
    )
    from OpenGL.GLU import gluLookAt
except ImportError as error:
    raise SystemExit(
        "A visualizer dependency is missing.\n"
        "Install both packages with:\n"
        "    python -m pip install pygame PyOpenGL\n\n"
        f"Original import error: {error}"
    ) from error


# -------------------------- User configuration -------------------------------

LISTEN_IP = "0.0.0.0"
LISTEN_PORT = 5005
WINDOW_SIZE = (960, 720)
ORTHO_HALF_HEIGHT = 3.4

# The simple view draws a top-down stick: X is screen-right and Y is screen-up.
# Flip DISPLAY_YAW_SIGN to -1.0 if clockwise/counter-clockwise motion is mirrored.
SIMPLE_TOP_DOWN_VIEW = False
DISPLAY_YAW_SIGN = 1.0
DISPLAY_PITCH_SIGN = -1.0
DISPLAY_YAW_OFFSET_DEGREES = 0.0
# Static drawing correction only. This does not change the IMU data or filter.
# Flip this between 0.0 and 180.0 if the racket face appears upside down.
DISPLAY_MODEL_UPSIDE_DOWN_CORRECTION_DEGREES = 180.0

# Acceleration history shown in the lower-left overlay. Keep every packet
# received from the Pico; 320 points is approximately 0.8 seconds at 400 Hz.
ACCEL_GRAPH_SAMPLES = 320
ACCEL_GRAPH_HALF_RANGE_G = 32.0
ACCEL_GRAPH_TRIGGER_G = 4
ANGLE_GRAPH_SAMPLES = 96  # approximately 0.8 seconds at 120 display frames/s
VIDEO_FPS = 240
# Unlock only on deliberate movement; low-level MPU6050 noise must not
# repeatedly release the stationary-pose lock.
STATIONARY_GYRO_LIMIT_DPS = 20.0
STATIONARY_ACCEL_TOLERANCE_G = 0.12
STATIONARY_LOCK_DELAY_SECONDS = 0.0
ZERO_MOTION_GYRO_LIMIT_DPS = 25.0
ZERO_MOTION_ACCEL_TOLERANCE_G = 0.30
GYRO_DEADBAND_DPS = 2.0
GYRO_BIAS_LEARNING_RATE = 0.03

# Axis mapping items are (source axis index, sign). Set an entry in either
# inversion tuple to True when that physical axis is mounted backwards.
# The IMU map is shared by accelerometer, gyro, and the 3D model.
IMU_AXIS_ORDER = (0, 1, 2)
IMU_AXIS_INVERT = (False, False, True)
IMU_AXIS_MAP = tuple(
    (source, -1.0 if inverted else 1.0)
    for source, inverted in zip(IMU_AXIS_ORDER, IMU_AXIS_INVERT)
)

# Keep the sensor and model coordinates explicit and unchanged by default.
# If the board is physically mounted differently, change IMU_AXIS_MAP once;
# acceleration, gyro, and the 3D model then use the same convention.
RACKET_ROLL_AXIS = 0
RACKET_PITCH_AXIS = 1

# Use the magnetometer to limit long-term yaw drift, but do not use gravity to
# pull the racket's deliberate tilt/roll back toward level.
USE_ABSOLUTE_ORIENTATION_CORRECTION = False
USE_ACCELEROMETER_CORRECTION = True
USE_MAGNETOMETER_CORRECTION = True

# Change this when the MMC5603 breakout has a different physical orientation.
MAG_AXIS_ORDER = (0, 1, 2)
MAG_AXIS_INVERT = (False, True, False)
MAG_AXIS_MAP = tuple(
    (source, -1.0 if inverted else 1.0)
    for source, inverted in zip(MAG_AXIS_ORDER, MAG_AXIS_INVERT)
)

# Kalman tuning values are variances in radians. Larger process noise trusts
# the gyro prediction less; larger measurement noise trusts that sensor less.
KALMAN_PROCESS_NOISE = math.radians(2.0) ** 2
KALMAN_ACCEL_MEASUREMENT_NOISE = math.radians(4.0) ** 2
KALMAN_MAG_MEASUREMENT_NOISE = math.radians(10.0) ** 2

# Reject obviously bad magnetic samples.
MAG_MIN_FIELD = 5.0
MAG_MAX_FIELD = 120.0

# -----------------------------------------------------------------------------

DEG = 180.0 / math.pi


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def vector_norm(vector):
    return math.sqrt(sum(component * component for component in vector))


def apply_axis_map(vector, mapping):
    return tuple(vector[index] * sign for index, sign in mapping)


def apply_gyro_deadband(gyro):
    """Remove low-rate gyro noise that would otherwise accumulate as drift."""
    return tuple(
        0.0 if abs(value) < GYRO_DEADBAND_DPS else value
        for value in gyro
    )


def subtract_vectors(left, right):
    return tuple(left[index] - right[index] for index in range(3))


def blend_vectors(current, measured, amount):
    return tuple(
        current[index] + amount * (measured[index] - current[index])
        for index in range(3)
    )


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def blend_angle(current, measured, amount):
    return wrap_angle(current + amount * wrap_angle(measured - current))


def quaternion_normalize(q):
    magnitude = math.sqrt(sum(value * value for value in q))
    if magnitude < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(value / magnitude for value in q)


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quaternion_conjugate(q):
    return (q[0], -q[1], -q[2], -q[3])


def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return quaternion_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def opengl_matrix_from_quaternion(q):
    """Return a column-major OpenGL rotation matrix from a normalized quaternion."""
    w, x, y, z = quaternion_normalize(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        1.0 - 2.0 * (yy + zz),
        2.0 * (xy + wz),
        2.0 * (xz - wy),
        0.0,
        2.0 * (xy - wz),
        1.0 - 2.0 * (xx + zz),
        2.0 * (yz + wx),
        0.0,
        2.0 * (xz + wy),
        2.0 * (yz - wx),
        1.0 - 2.0 * (xx + yy),
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )


def euler_from_quaternion(q):
    w, x, y, z = quaternion_normalize(q)

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )

    pitch_term = 2.0 * (w * y - z * x)
    pitch = math.asin(clamp(pitch_term, -1.0, 1.0))

    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )

    return roll, pitch, yaw


def racket_angles(q):
    """Return angles labelled for the racket convention (roll about shaft)."""
    raw_angles = euler_from_quaternion(q)
    return (
        raw_angles[RACKET_ROLL_AXIS],
        raw_angles[RACKET_PITCH_AXIS],
        raw_angles[2],
    )


def unwrap_degrees(previous, current):
    """Choose the equivalent current angle nearest to the previous one."""
    delta = (current - previous + 180.0) % 360.0 - 180.0
    return previous + delta


def accel_tilt(acceleration):
    ax, ay, az = acceleration
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
    return roll, pitch


def tilt_compensated_heading(magnetic, roll, pitch):
    mx, my, mz = magnetic

    horizontal_x = mx * math.cos(pitch) + mz * math.sin(pitch)
    horizontal_y = (
        mx * math.sin(roll) * math.sin(pitch)
        + my * math.cos(roll)
        - mz * math.sin(roll) * math.cos(pitch)
    )

    return wrap_angle(math.atan2(-horizontal_y, horizontal_x))


class MagnetometerCalibration:
    """Simple hard-iron offset and per-axis scale calibration."""

    def __init__(self):
        self.offset = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.active = False
        self.minimum = [float("inf")] * 3
        self.maximum = [float("-inf")] * 3
        self.samples = 0

    def start(self):
        self.active = True
        self.minimum = [float("inf")] * 3
        self.maximum = [float("-inf")] * 3
        self.samples = 0
        print(
            "Magnetometer calibration started. Slowly rotate the complete "
            "assembly through every orientation, then press C again."
        )

    def add(self, vector):
        if not self.active:
            return

        for axis in range(3):
            self.minimum[axis] = min(self.minimum[axis], vector[axis])
            self.maximum[axis] = max(self.maximum[axis], vector[axis])
        self.samples += 1

    def finish(self):
        if not self.active:
            return False

        self.active = False
        spans = [
            self.maximum[axis] - self.minimum[axis]
            for axis in range(3)
        ]

        if self.samples < 100 or min(spans) < 5.0:
            print(
                "Calibration rejected: there was not enough rotation. "
                "Try again and cover more orientations."
            )
            return False

        radii = [span * 0.5 for span in spans]
        average_radius = sum(radii) / 3.0

        self.offset = [
            (self.maximum[axis] + self.minimum[axis]) * 0.5
            for axis in range(3)
        ]
        self.scale = [
            average_radius / radius if radius > 1e-9 else 1.0
            for radius in radii
        ]

        print("Magnetometer calibration updated in memory.")
        print("Offset:", [round(value, 3) for value in self.offset])
        print("Scale: ", [round(value, 3) for value in self.scale])
        return True

    def clear(self):
        self.offset = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.active = False
        print("Magnetometer calibration cleared.")

    def apply(self, vector):
        return tuple(
            (vector[axis] - self.offset[axis]) * self.scale[axis]
            for axis in range(3)
        )


class AngleKalmanFilter:
    """One-angle Kalman corrector for wrapped Euler angles."""

    def __init__(self, process_noise, measurement_noise):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.estimate_error = measurement_noise

    def predict(self, dt):
        self.estimate_error += self.process_noise * dt

    def correct(self, estimate, measurement):
        kalman_gain = self.estimate_error / (
            self.estimate_error + self.measurement_noise
        )
        estimate = blend_angle(estimate, measurement, kalman_gain)
        self.estimate_error *= 1.0 - kalman_gain
        return estimate


class KalmanOrientation:
    def __init__(self):
        self.q = (1.0, 0.0, 0.0, 0.0)
        self.initialized = False
        self.roll_filter = AngleKalmanFilter(
            KALMAN_PROCESS_NOISE,
            KALMAN_ACCEL_MEASUREMENT_NOISE,
        )
        self.pitch_filter = AngleKalmanFilter(
            KALMAN_PROCESS_NOISE,
            KALMAN_ACCEL_MEASUREMENT_NOISE,
        )
        self.yaw_filter = AngleKalmanFilter(
            KALMAN_PROCESS_NOISE,
            KALMAN_MAG_MEASUREMENT_NOISE,
        )

    def initialize(self, acceleration, magnetic):
        roll, pitch = accel_tilt(acceleration)
        yaw = 0.0

        if MAG_MIN_FIELD <= vector_norm(magnetic) <= MAG_MAX_FIELD:
            yaw = tilt_compensated_heading(magnetic, roll, pitch)

        self.q = quaternion_from_euler(roll, pitch, yaw)
        self.initialized = True

    def update(
        self,
        gyro_dps,
        acceleration,
        magnetic,
        dt,
        use_magnetometer=True,
        use_absolute_corrections=True,
        use_accelerometer=True,
    ):
        dt = clamp(dt, 0.001, 0.1)

        if not self.initialized:
            self.initialize(acceleration, magnetic)
            return

        # Integrate body-axis angular rate into a body-to-world quaternion.
        gx, gy, gz = (math.radians(value) for value in gyro_dps)
        q_dot = quaternion_multiply(self.q, (0.0, gx, gy, gz))
        self.q = quaternion_normalize(
            tuple(
                self.q[index] + 0.5 * q_dot[index] * dt
                for index in range(4)
            )
        )

        roll, pitch, yaw = euler_from_quaternion(self.q)
        self.roll_filter.predict(dt)
        self.pitch_filter.predict(dt)
        self.yaw_filter.predict(dt)

        # Correct roll and pitch only while acceleration remains close to 1 g.
        acceleration_magnitude = vector_norm(acceleration)
        gyro_magnitude = vector_norm(gyro_dps)
        is_stationary = gyro_magnitude < 8.0
        if (
            use_absolute_corrections
            and use_accelerometer
            and is_stationary
            and 0.75 <= acceleration_magnitude <= 1.25
        ):
            measured_roll, measured_pitch = accel_tilt(acceleration)
            roll = self.roll_filter.correct(roll, measured_roll)
            pitch = self.pitch_filter.correct(pitch, measured_pitch)

        # Correct long-term yaw drift from the magnetometer.
        magnetic_magnitude = vector_norm(magnetic)
        if (
            use_absolute_corrections
            and use_magnetometer
            and is_stationary
            and MAG_MIN_FIELD <= magnetic_magnitude <= MAG_MAX_FIELD
        ):
            measured_yaw = tilt_compensated_heading(
                magnetic,
                roll,
                pitch,
            )
            yaw = self.yaw_filter.correct(yaw, measured_yaw)

        self.q = quaternion_from_euler(roll, pitch, yaw)


CUBOID_HALF_LENGTH = 2.8
CUBOID_HALF_WIDTH = 0.22
CUBOID_HALF_THICKNESS = 0.08

CUBOID_VERTICES = (
    (-CUBOID_HALF_WIDTH, -CUBOID_HALF_LENGTH, -CUBOID_HALF_THICKNESS),
    (CUBOID_HALF_WIDTH, -CUBOID_HALF_LENGTH, -CUBOID_HALF_THICKNESS),
    (CUBOID_HALF_WIDTH, CUBOID_HALF_LENGTH, -CUBOID_HALF_THICKNESS),
    (-CUBOID_HALF_WIDTH, CUBOID_HALF_LENGTH, -CUBOID_HALF_THICKNESS),
    (-CUBOID_HALF_WIDTH, -CUBOID_HALF_LENGTH, CUBOID_HALF_THICKNESS),
    (CUBOID_HALF_WIDTH, -CUBOID_HALF_LENGTH, CUBOID_HALF_THICKNESS),
    (CUBOID_HALF_WIDTH, CUBOID_HALF_LENGTH, CUBOID_HALF_THICKNESS),
    (-CUBOID_HALF_WIDTH, CUBOID_HALF_LENGTH, CUBOID_HALF_THICKNESS),
)

CUBOID_FACES = (
    ((0, 1, 2, 3), (0.22, 0.36, 0.76)),
    ((4, 7, 6, 5), (0.25, 0.72, 0.46)),
    ((0, 4, 5, 1), (0.84, 0.34, 0.31)),
    ((1, 5, 6, 2), (0.88, 0.67, 0.23)),
    ((2, 6, 7, 3), (0.48, 0.31, 0.72)),
    ((3, 7, 4, 0), (0.24, 0.68, 0.72)),
)

CUBOID_EDGES = (
    (0, 1, 2, 3),
    (4, 5, 6, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def draw_cuboid():
    glBegin(GL_QUADS)
    for indices, color in CUBOID_FACES:
        glColor3f(*color)
        for index in indices:
            glVertex3f(*CUBOID_VERTICES[index])
    glEnd()

    glColor3f(0.05, 0.05, 0.06)
    glLineWidth(2.0)

    for edge in CUBOID_EDGES:
        glBegin(GL_LINE_LOOP if len(edge) == 4 else GL_LINES)
        for index in edge:
            glVertex3f(*CUBOID_VERTICES[index])
        glEnd()

    # White marker showing the cuboid's long axis and positive Y direction.
    glColor3f(1.0, 1.0, 1.0)
    glLineWidth(5.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, -CUBOID_HALF_LENGTH * 0.9, CUBOID_HALF_THICKNESS * 1.2)
    glVertex3f(0.0, CUBOID_HALF_LENGTH * 0.9, CUBOID_HALF_THICKNESS * 1.2)
    glEnd()

    glColor3f(1.0, 0.95, 0.25)
    glLineWidth(8.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, CUBOID_HALF_LENGTH * 0.65, CUBOID_HALF_THICKNESS * 1.35)
    glVertex3f(0.0, CUBOID_HALF_LENGTH, CUBOID_HALF_THICKNESS * 1.35)
    glEnd()


def draw_racket_model():
    """Draw a badminton racket with unambiguous six-axis markers."""
    draw_cuboid()

    # Handle and neck, extending from the bottom of the head toward negative Y.
    glColor3f(0.12, 0.12, 0.14)
    glLineWidth(10.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, -CUBOID_HALF_LENGTH, 0.0)
    glVertex3f(0.0, -CUBOID_HALF_LENGTH - 1.5, 0.0)
    glEnd()

    glLineWidth(5.0)
    glBegin(GL_LINES)
    glVertex3f(-0.28, 0.0, 0.0)
    glVertex3f(-0.75, 0.55, 0.0)
    glVertex3f(0.28, 0.0, 0.0)
    glVertex3f(0.75, 0.55, 0.0)
    glEnd()

    # Oval badminton-racket head in the X/Y plane, with an inner frame and
    # a simple string bed. It is intentionally not a symmetric cuboid.
    center_y = 1.75
    outer_radius_x = 1.45
    outer_radius_y = 1.65
    inner_radius_x = 1.25
    inner_radius_y = 1.45
    segments = 48

    glColor3f(0.82, 0.86, 0.92)
    glLineWidth(6.0)
    glBegin(GL_LINE_LOOP)
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        glVertex3f(
            outer_radius_x * math.cos(angle),
            center_y + outer_radius_y * math.sin(angle),
            0.0,
        )
    glEnd()

    glColor3f(0.38, 0.65, 0.82)
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        glVertex3f(
            inner_radius_x * math.cos(angle),
            center_y + inner_radius_y * math.sin(angle),
            0.02,
        )
    glEnd()

    glColor3f(0.62, 0.72, 0.82)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for index in range(-4, 5):
        x = index * inner_radius_x / 4.0
        half_height = inner_radius_y * math.sqrt(
            max(0.0, 1.0 - (x / inner_radius_x) ** 2)
        )
        glVertex3f(x, center_y - half_height, 0.03)
        glVertex3f(x, center_y + half_height, 0.03)
    for index in range(-4, 5):
        y = index * inner_radius_y / 4.0
        half_width = inner_radius_x * math.sqrt(
            max(0.0, 1.0 - (y / inner_radius_y) ** 2)
        )
        glVertex3f(-half_width, center_y + y, 0.03)
        glVertex3f(half_width, center_y + y, 0.03)
    glEnd()

    # Six colored rays identify both signs of every body axis. Bright colors
    # are positive directions; darker colors are the corresponding negatives.
    axis_markers = (
        ((1.0, 0.18, 0.18), (3.2, 0.0, 0.0)),
        ((0.45, 0.04, 0.04), (-3.2, 0.0, 0.0)),
        ((0.18, 1.0, 0.28), (0.0, 3.2, 0.0)),
        ((0.72, 0.48, 0.04), (0.0, -3.2, 0.0)),
        ((0.25, 0.55, 1.0), (0.0, 0.0, 3.2)),
        ((0.08, 0.22, 0.58), (0.0, 0.0, -3.2)),
    )
    glLineWidth(4.0)
    for color, endpoint in axis_markers:
        glColor3f(*color)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(*endpoint)
        glEnd()


def start_video_recording(width, height):
    """Start an ffmpeg pipe for rendered RGB frames."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and Path("/opt/homebrew/bin/ffmpeg").is_file():
        ffmpeg = "/opt/homebrew/bin/ffmpeg"
    if ffmpeg is None:
        print("Could not start video recording: ffmpeg is not on PATH.")
        return None, None
    recording_directory = Path(__file__).resolve().parent / "recordings"
    recording_directory.mkdir(parents=True, exist_ok=True)
    filename = datetime.now().strftime("swing_%Y%m%d_%H%M%S_%f.mp4")
    path = recording_directory / filename
    command = (
        ffmpeg, "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-framerate", str(VIDEO_FPS), "-i", "-",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-an", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", str(path),
    )
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        print(f"Could not start video recording: {error}")
        return None, None
    print(f"Video recording started: {path}")
    return process, path


def write_video_frame(process, width, height):
    """Read the OpenGL back buffer and append one vertically-corrected frame."""
    if process is None or process.stdin is None:
        return False
    if process.poll() is not None:
        print(f"Video encoder exited early with status {process.returncode}.")
        return False
    try:
        pixels = bytes(glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE))
    except (TypeError, ValueError, RuntimeError) as error:
        print(f"OpenGL frame capture failed: {error}")
        return False
    row_size = width * 3
    expected_size = row_size * height
    if len(pixels) != expected_size:
        print(
            "OpenGL frame capture returned "
            f"{len(pixels)} bytes; expected {expected_size}."
        )
        return False
    frame = b"".join(
        pixels[row:row + row_size]
        for row in range((height - 1) * row_size, -1, -row_size)
    )
    try:
        process.stdin.write(frame)
        process.stdin.flush()
    except (BrokenPipeError, OSError) as error:
        print(f"Video frame write failed: {error}")
        return False
    return True


def finish_video_recording(process):
    """Close ffmpeg cleanly and return whether encoding succeeded."""
    if process is None:
        return False
    try:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait(timeout=10)
        diagnostics = b"" if process.stderr is None else process.stderr.read()
        if return_code != 0:
            message = diagnostics.decode(errors="replace").strip()
            print(f"Video encoder failed ({return_code}): {message}")
        return return_code == 0
    except (BrokenPipeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Video finalization failed: {error}")
        process.kill()
        return False

def draw_world_axes():
    glLineWidth(2.0)
    glBegin(GL_LINES)

    glColor3f(0.85, 0.25, 0.25)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(2.2, 0.0, 0.0)

    glColor3f(0.25, 0.8, 0.35)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 2.2, 0.0)

    glColor3f(0.30, 0.45, 0.90)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 2.2)

    glEnd()


def draw_acceleration_graph(history, width, height):
    """Draw recent X/Y/Z acceleration traces as a 2D OpenGL overlay."""
    graph_width = min(700.0, max(320.0, width * 0.70))
    graph_height = 180.0
    left = 20.0
    bottom = 20.0
    right = left + graph_width
    top = bottom + graph_height

    glDisable(GL_DEPTH_TEST)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, float(width), 0.0, float(height), -1.0, 1.0)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Graph background and border.
    glColor3f(0.04, 0.05, 0.07)
    glBegin(GL_QUADS)
    glVertex3f(left, bottom, 0.0)
    glVertex3f(right, bottom, 0.0)
    glVertex3f(right, top, 0.0)
    glVertex3f(left, top, 0.0)
    glEnd()

    glColor3f(0.35, 0.37, 0.42)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(left, bottom, 0.0)
    glVertex3f(right, bottom, 0.0)
    glVertex3f(right, top, 0.0)
    glVertex3f(left, top, 0.0)
    glEnd()

    # Zero-g reference line.
    zero_y = bottom + graph_height * 0.5
    glColor3f(0.28, 0.29, 0.33)
    glBegin(GL_LINES)
    glVertex3f(left, zero_y, 0.0)
    glVertex3f(right, zero_y, 0.0)
    glEnd()

    graph_half_range = ACCEL_GRAPH_HALF_RANGE_G
    if history:
        colors = ((0.95, 0.25, 0.25), (0.25, 0.85, 0.35), (0.30, 0.55, 1.0))
        samples = tuple(history)
        denominator = max(1, len(samples) - 1)
        peak = max(abs(value) for sample in samples for value in sample)
        graph_half_range = max(ACCEL_GRAPH_HALF_RANGE_G, peak * 1.10)

        for axis, color in enumerate(colors):
            glColor3f(*color)
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            for index, sample in enumerate(samples):
                value = sample[axis]
                x = left + graph_width * index / denominator
                y = zero_y + (
                    value / graph_half_range
                ) * (graph_height * 0.5)
                glVertex3f(x, y, 0.0)
            glEnd()

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def draw_overlay_text(font, text, x, y, color=(230, 230, 235)):
    """Draw a small pygame label at OpenGL overlay coordinates."""
    surface = font.render(text, True, color)
    pixels = pygame.image.tostring(surface, "RGBA", True)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glRasterPos2f(float(x), float(y))
    glDrawPixels(
        surface.get_width(),
        surface.get_height(),
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        pixels,
    )
    glDisable(GL_BLEND)


def draw_angle_graph(history, width, height, font):
    """Draw recent roll/pitch/yaw traces in degrees as a labeled overlay."""
    graph_width = min(700.0, max(320.0, width * 0.70))
    graph_height = 180.0
    left = 20.0
    bottom = 220.0
    right = left + graph_width
    top = bottom + graph_height

    glDisable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0.0, float(width), 0.0, float(height), -1.0, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glColor3f(0.04, 0.05, 0.07)
    glBegin(GL_QUADS)
    glVertex3f(left, bottom, 0.0)
    glVertex3f(right, bottom, 0.0)
    glVertex3f(right, top, 0.0)
    glVertex3f(left, top, 0.0)
    glEnd()

    glColor3f(0.35, 0.37, 0.42)
    glLineWidth(1.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(left, bottom, 0.0)
    glVertex3f(right, bottom, 0.0)
    glVertex3f(right, top, 0.0)
    glVertex3f(left, top, 0.0)
    glEnd()

    zero_y = bottom + graph_height * 0.5
    glColor3f(0.28, 0.29, 0.33)
    glBegin(GL_LINES)
    glVertex3f(left, zero_y, 0.0)
    glVertex3f(right, zero_y, 0.0)
    glEnd()

    samples = tuple(history)
    angle_min = min((value for sample in samples for value in sample), default=-180.0)
    angle_max = max((value for sample in samples for value in sample), default=180.0)
    angle_center = (angle_min + angle_max) * 0.5
    angle_half_range = max(180.0, (angle_max - angle_min) * 0.6)

    # The scale follows the unwrapped data, preventing a 180/-180 seam from
    # becoming a false vertical line.
    for tick in (
        angle_center - angle_half_range,
        angle_center,
        angle_center + angle_half_range,
    ):
        y = bottom + (tick - angle_center + angle_half_range) / (
            2.0 * angle_half_range
        ) * graph_height
        glColor3f(0.25, 0.26, 0.30)
        glBegin(GL_LINES)
        glVertex3f(left, y, 0.0)
        glVertex3f(left + 8.0, y, 0.0)
        glEnd()

    if history:
        colors = ((0.95, 0.25, 0.25), (0.25, 0.85, 0.35), (0.30, 0.55, 1.0))
        denominator = max(1, len(samples) - 1)
        for axis, color in enumerate(colors):
            glColor3f(*color)
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            for index, sample in enumerate(samples):
                x = left + graph_width * index / denominator
                y = bottom + (
                    (sample[axis] - angle_center + angle_half_range)
                    / (2.0 * angle_half_range)
                ) * graph_height
                glVertex3f(x, y, 0.0)
            glEnd()

    draw_overlay_text(font, "Angle (degrees)", left + 8.0, top - 18.0)
    draw_overlay_text(font, "X / roll", right - 175.0, top - 18.0, (240, 80, 80))
    draw_overlay_text(font, "Y / pitch", right - 110.0, top - 18.0, (80, 220, 100))
    draw_overlay_text(font, "Z / yaw", right - 42.0, top - 18.0, (90, 150, 255))
    draw_overlay_text(font, f"{angle_center + angle_half_range:.0f}", left - 2.0, top - 34.0)
    draw_overlay_text(font, f"{angle_center:.0f}", left + 4.0, zero_y - 5.0)
    draw_overlay_text(font, f"{angle_center - angle_half_range:.0f}", left - 2.0, bottom + 4.0)
    draw_overlay_text(font, "angle", left + 8.0, bottom + 22.0)
    draw_overlay_text(font, "time (0.8 s)", right - 78.0, bottom + 4.0)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)


def configure_projection(width, height):
    height = max(height, 1)
    aspect = width / height
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(
        -ORTHO_HALF_HEIGHT * aspect,
        ORTHO_HALF_HEIGHT * aspect,
        -ORTHO_HALF_HEIGHT,
        ORTHO_HALF_HEIGHT,
        -100.0,
        100.0,
    )
    glMatrixMode(GL_MODELVIEW)


def configure_rendering(width, height):
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.055, 0.065, 0.085, 1.0)
    configure_projection(width, height)


def valid_vector(value):
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) for item in value)
    )


def validate_packet(packet):
    return (
        isinstance(packet, dict)
        and isinstance(packet.get("seq"), int)
        and valid_vector(packet.get("a"))
        and valid_vector(packet.get("g"))
        and valid_vector(packet.get("m"))
        and (
            packet.get("t_ms") is None
            or isinstance(packet.get("t_ms"), (int, float))
        )
    )


def main():
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        udp_socket.bind((LISTEN_IP, LISTEN_PORT))
    except OSError as error:
        raise SystemExit(
            f"Could not listen on UDP port {LISTEN_PORT}: {error}"
        ) from error

    udp_socket.setblocking(False)

    pygame.init()
    pygame.display.set_caption(
        f"Pico IMU Cuboid - waiting for UDP port {LISTEN_PORT}"
    )
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL | RESIZABLE)

    configure_rendering(*WINDOW_SIZE)

    calibration = MagnetometerCalibration()
    orientation = KalmanOrientation()
    zero_inverse = (1.0, 0.0, 0.0, 0.0)
    acceleration_history = deque(maxlen=ACCEL_GRAPH_SAMPLES)
    angle_history = deque(maxlen=ANGLE_GRAPH_SAMPLES)
    acceleration_triggered = False
    graph_frozen = False
    orientation_frozen = False
    stationary_pose_locked = False
    stationary_since = None
    gyro_bias = (0.0, 0.0, 0.0)
    last_graph_angles = None
    video_process = None
    video_path = None
    video_frame_count = 0
    last_video_path = None
    window_width, window_height = WINDOW_SIZE
    graph_font = pygame.font.Font(None, 16)

    clock = pygame.time.Clock()
    running = True
    packet_count = 0
    malformed_count = 0
    last_packet_time = None
    last_sample_time_ms = None
    last_title_update = 0.0
    latest_temperature = 0.0

    print(f"Listening on UDP {LISTEN_IP}:{LISTEN_PORT}")
    print(
        "R = zero pose, C = calibrate magnetometer, L = clear calibration, "
        "Space = restart acceleration capture"
    )
    print("Escape = quit")

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                if video_process is not None:
                    finish_video_recording(video_process)
                    video_process = None
                    video_path = None
                    print("Video capture stopped because the window was resized.")
                pygame.display.set_mode(
                    (event.w, event.h),
                    DOUBLEBUF | OPENGL | RESIZABLE,
                )
                window_width, window_height = event.w, event.h
                configure_rendering(event.w, event.h)
            elif event.type == KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    zero_inverse = quaternion_conjugate(orientation.q)
                    print("Display pose zeroed.")
                elif event.key == pygame.K_c:
                    if calibration.active:
                        calibration.finish()
                    else:
                        calibration.start()
                elif event.key == pygame.K_l:
                    calibration.clear()
                elif event.key == pygame.K_SPACE:
                    acceleration_history.clear()
                    angle_history.clear()
                    acceleration_triggered = False
                    graph_frozen = False
                    orientation_frozen = False
                    stationary_pose_locked = False
                    stationary_since = None
                    gyro_bias = (0.0, 0.0, 0.0)
                    last_graph_angles = None
                    if video_process is not None:
                        finish_video_recording(video_process)
                        video_process = None
                    video_path = None
                    video_frame_count = 0
                    last_video_path = None
                    print("Acceleration graph capture restarted.")

        # Drain queued UDP datagrams. The orientation uses the newest sample,
        # while the graph capture records every valid acceleration sample.
        newest_packet = None
        while True:
            try:
                payload, _sender = udp_socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError:
                break

            try:
                packet = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                malformed_count += 1
                continue

            if validate_packet(packet):
                packet_acceleration = apply_axis_map(
                    tuple(float(value) for value in packet["a"]),
                    IMU_AXIS_MAP,
                )
                if not graph_frozen:
                    acceleration_history.append(packet_acceleration)
                    acceleration_magnitude = vector_norm(packet_acceleration)
                    if acceleration_magnitude > ACCEL_GRAPH_TRIGGER_G:
                        if not acceleration_triggered:
                            stationary_pose_locked = False
                            stationary_since = None
                            video_process, video_path = start_video_recording(
                                window_width,
                                window_height,
                            )
                            video_frame_count = 0
                            print("Swing recording started.")
                        acceleration_triggered = True

                    if acceleration_triggered and acceleration_magnitude <= ACCEL_GRAPH_TRIGGER_G:
                        graph_frozen = True
                        print(
                            "Acceleration graph frozen after swing "
                            f"({acceleration_magnitude:.2f} g). "
                            "Press Space to restart."
                        )
                newest_packet = packet
            else:
                malformed_count += 1

        now = time.perf_counter()

        if newest_packet is not None:
            acceleration = apply_axis_map(
                tuple(float(value) for value in newest_packet["a"]),
                IMU_AXIS_MAP,
            )
            gyro = apply_axis_map(
                tuple(float(value) for value in newest_packet["g"]),
                IMU_AXIS_MAP,
            )
            acceleration_magnitude = vector_norm(acceleration)
            raw_gyro = gyro
            gyro = subtract_vectors(raw_gyro, gyro_bias)
            raw_gyro_magnitude = vector_norm(raw_gyro)
            corrected_gyro_magnitude = vector_norm(gyro)
            if (
                not acceleration_triggered
                and 1.0 - STATIONARY_ACCEL_TOLERANCE_G
                <= acceleration_magnitude
                <= 1.0 + STATIONARY_ACCEL_TOLERANCE_G
                and corrected_gyro_magnitude <= STATIONARY_GYRO_LIMIT_DPS
            ):
                gyro_bias = blend_vectors(
                    gyro_bias,
                    raw_gyro,
                    GYRO_BIAS_LEARNING_RATE,
                )
                gyro = subtract_vectors(raw_gyro, gyro_bias)
                corrected_gyro_magnitude = vector_norm(gyro)
            gyro = apply_gyro_deadband(gyro)
            if (
                not acceleration_triggered
                and corrected_gyro_magnitude <= STATIONARY_GYRO_LIMIT_DPS
            ):
                gyro = (0.0, 0.0, 0.0)
            magnetic_uncalibrated = apply_axis_map(
                tuple(float(value) for value in newest_packet["m"]),
                MAG_AXIS_MAP,
            )

            gyro_magnitude = vector_norm(gyro)
            zero_motion_sample = gyro_magnitude < GYRO_DEADBAND_DPS
            if acceleration_triggered:
                stationary_pose_locked = False
                stationary_since = None
            elif stationary_pose_locked:
                if gyro_magnitude > STATIONARY_GYRO_LIMIT_DPS:
                    stationary_pose_locked = False
                    stationary_since = None
            else:
                if gyro_magnitude > STATIONARY_GYRO_LIMIT_DPS:
                    stationary_since = None
                else:
                    if stationary_since is None:
                        stationary_since = now
                    if now - stationary_since >= STATIONARY_LOCK_DELAY_SECONDS:
                        stationary_pose_locked = True
                        print("Stationary pose locked; movement will unlock it.")

            calibration.add(magnetic_uncalibrated)
            magnetic = calibration.apply(magnetic_uncalibrated)

            sample_time_ms = newest_packet.get("t_ms")
            if (
                isinstance(sample_time_ms, (int, float))
                and last_sample_time_ms is not None
                and sample_time_ms > last_sample_time_ms
            ):
                dt = (sample_time_ms - last_sample_time_ms) / 1000.0
            elif last_packet_time is None:
                dt = 1.0 / 50.0
            else:
                dt = now - last_packet_time

            if isinstance(sample_time_ms, (int, float)):
                last_sample_time_ms = sample_time_ms
            last_packet_time = now

            if not orientation_frozen and not stationary_pose_locked:
                orientation.update(
                    gyro,
                    acceleration,
                    magnetic,
                    dt,
                    use_magnetometer=(
                        USE_MAGNETOMETER_CORRECTION
                        and not calibration.active
                        and not acceleration_triggered
                    ),
                    use_absolute_corrections=(
                        USE_ABSOLUTE_ORIENTATION_CORRECTION
                        and not zero_motion_sample
                    ),
                    use_accelerometer=USE_ACCELEROMETER_CORRECTION,
                )
            if (
                not graph_frozen
                and not orientation_frozen
                and not stationary_pose_locked
            ):
                raw_angles = tuple(
                    value * DEG for value in racket_angles(orientation.q)
                )
                if last_graph_angles is None:
                    graph_angles = raw_angles
                else:
                    graph_angles = tuple(
                        unwrap_degrees(previous, current)
                        for previous, current in zip(last_graph_angles, raw_angles)
                    )
                last_graph_angles = graph_angles
                angle_history.append(graph_angles)
            if graph_frozen:
                orientation_frozen = True
            latest_temperature = float(newest_packet.get("temp", 0.0))
            packet_count += 1

        display_q = quaternion_multiply(zero_inverse, orientation.q)
        roll, pitch, yaw = racket_angles(display_q)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(
            7.0,
            -9.0,
            7.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )

        draw_world_axes()

        glPushMatrix()
        if SIMPLE_TOP_DOWN_VIEW:
            glRotatef(
                DISPLAY_YAW_OFFSET_DEGREES + DISPLAY_YAW_SIGN * yaw * DEG,
                0.0,
                0.0,
                1.0,
            )
        else:
            # Drive the 3D racket directly from the quaternion. Using Euler
            # rotations here makes a pure lift leak into apparent clockwise
            # rotation near vertical attitudes.
            glRotatef(DISPLAY_YAW_OFFSET_DEGREES, 0.0, 0.0, 1.0)
            glMultMatrixf(opengl_matrix_from_quaternion(display_q))
            glRotatef(
                DISPLAY_MODEL_UPSIDE_DOWN_CORRECTION_DEGREES,
                0.0,
                1.0,
                0.0,
            )
        draw_racket_model()
        glPopMatrix()

        draw_acceleration_graph(
            acceleration_history,
            window_width,
            window_height,
        )
        draw_angle_graph(
            angle_history,
            window_width,
            window_height,
            graph_font,
        )

        if video_process is not None:
            if not write_video_frame(video_process, window_width, window_height):
                finish_video_recording(video_process)
                video_process = None
                video_path = None
                video_frame_count = 0
                print("Video recording stopped because ffmpeg closed the pipe.")
            elif graph_frozen and video_frame_count >= 3:
                video_frame_count += 1
                video_ok = finish_video_recording(video_process)
                video_process = None
                if (
                    video_ok
                    and video_frame_count > 0
                    and video_path is not None
                    and video_path.is_file()
                    and video_path.stat().st_size > 0
                ):
                    last_video_path = video_path
                    print(
                        f"Swing video saved to {last_video_path} "
                        f"({video_path.stat().st_size} bytes)"
                    )
                else:
                    if video_ok:
                        print("Video recording had no frames; no MP4 was saved.")
                    video_path = None
                    video_frame_count = 0
            else:
                video_frame_count += 1

        pygame.display.flip()

        clock.tick(VIDEO_FPS)

        if now - last_title_update > 0.1:
            connected = (
                last_packet_time is not None
                and now - last_packet_time < 1.0
            )
            status = "CONNECTED" if connected else "WAITING"
            if calibration.active:
                status += " | CALIBRATING MAG"
            if acceleration_triggered and not graph_frozen:
                status += " | SWING DETECTED"
            elif graph_frozen:
                status += " | GRAPH FROZEN"
            if last_video_path is not None:
                status += " | VIDEO SAVED"
            elif video_process is not None:
                status += " | RECORDING VIDEO"

            pygame.display.set_caption(
                "Pico IMU Racket | 3D Kalman | {} | packets {} | "
                "roll {:+6.1f} pitch {:+6.1f} yaw {:+6.1f} deg | "
                "temp {:.1f} C | graph X:red Y:green Z:blue (auto ±32 g min, 0.8 s) | "
                "R zero, C calibrate, L clear, Space restart".format(
                    status,
                    packet_count,
                    roll * DEG,
                    pitch * DEG,
                    yaw * DEG,
                    latest_temperature,
                )
            )
            last_title_update = now

    if video_process is not None:
        if (
            finish_video_recording(video_process)
            and video_frame_count > 0
            and video_path is not None
            and video_path.is_file()
            and video_path.stat().st_size > 0
        ):
            last_video_path = video_path
            print(
                f"Swing video saved to {last_video_path} "
                f"({video_path.stat().st_size} bytes)"
            )
    udp_socket.close()
    pygame.quit()
    print(
        f"Stopped. Received {packet_count} packets; "
        f"discarded {malformed_count} malformed packets."
    )


if __name__ == "__main__":
    main()

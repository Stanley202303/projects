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
import socket
import time

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
        GL_LINES,
        GL_LINE_LOOP,
        GL_MODELVIEW,
        GL_PROJECTION,
        GL_QUADS,
        glBegin,
        glClear,
        glClearColor,
        glColor3f,
        glEnable,
        glEnd,
        glLineWidth,
        glLoadIdentity,
        glMatrixMode,
        glOrtho,
        glPopMatrix,
        glPushMatrix,
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
SIMPLE_TOP_DOWN_VIEW = True
DISPLAY_YAW_SIGN = 1.0
DISPLAY_YAW_OFFSET_DEGREES = 0.0

# Each axis mapping item is (source axis index, sign).
# The same map is used for MPU6050 accelerometer and gyro data.
IMU_AXIS_MAP = ((0, 1.0), (1, 1.0), (2, 1.0))

# Change this when the MMC5603 breakout has a different physical orientation.
MAG_AXIS_MAP = ((0, 1.0), (1, 1.0), (2, 1.0))

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
        if 0.75 <= acceleration_magnitude <= 1.25:
            measured_roll, measured_pitch = accel_tilt(acceleration)
            roll = self.roll_filter.correct(roll, measured_roll)
            pitch = self.pitch_filter.correct(pitch, measured_pitch)

        # Correct long-term yaw drift from the magnetometer.
        magnetic_magnitude = vector_norm(magnetic)
        if (
            use_magnetometer
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

    clock = pygame.time.Clock()
    running = True
    packet_count = 0
    malformed_count = 0
    last_packet_time = None
    last_sample_time_ms = None
    last_title_update = 0.0
    latest_temperature = 0.0

    print(f"Listening on UDP {LISTEN_IP}:{LISTEN_PORT}")
    print("R = zero pose, C = calibrate magnetometer, L = clear calibration")
    print("Escape = quit")

    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == VIDEORESIZE:
                pygame.display.set_mode(
                    (event.w, event.h),
                    DOUBLEBUF | OPENGL | RESIZABLE,
                )
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

        # Drain queued UDP datagrams and process only the newest valid sample.
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
            magnetic_uncalibrated = apply_axis_map(
                tuple(float(value) for value in newest_packet["m"]),
                MAG_AXIS_MAP,
            )

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

            orientation.update(
                gyro,
                acceleration,
                magnetic,
                dt,
                use_magnetometer=not calibration.active,
            )

            latest_temperature = float(newest_packet.get("temp", 0.0))
            packet_count += 1

        display_q = quaternion_multiply(zero_inverse, orientation.q)
        roll, pitch, yaw = euler_from_quaternion(display_q)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(
            0.0,
            0.0,
            8.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        )

        draw_world_axes()

        glPushMatrix()
        glRotatef(
            DISPLAY_YAW_OFFSET_DEGREES + DISPLAY_YAW_SIGN * yaw * DEG,
            0.0,
            0.0,
            1.0,
        )
        if not SIMPLE_TOP_DOWN_VIEW:
            glRotatef(pitch * DEG, 0.0, 1.0, 0.0)
            glRotatef(roll * DEG, 1.0, 0.0, 0.0)
        draw_cuboid()
        glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

        if now - last_title_update > 0.1:
            connected = (
                last_packet_time is not None
                and now - last_packet_time < 1.0
            )
            status = "CONNECTED" if connected else "WAITING"
            if calibration.active:
                status += " | CALIBRATING MAG"

            pygame.display.set_caption(
                "Pico IMU Cuboid | top-down Kalman | {} | packets {} | "
                "roll {:+6.1f} pitch {:+6.1f} yaw {:+6.1f} deg | "
                "temp {:.1f} C | R zero, C calibrate, L clear".format(
                    status,
                    packet_count,
                    roll * DEG,
                    pitch * DEG,
                    yaw * DEG,
                    latest_temperature,
                )
            )
            last_title_update = now

    udp_socket.close()
    pygame.quit()
    print(
        f"Stopped. Received {packet_count} packets; "
        f"discarded {malformed_count} malformed packets."
    )


if __name__ == "__main__":
    main()

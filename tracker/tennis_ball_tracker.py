from __future__ import annotations

import json
import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


# ============================================================
# NETWORK SETTINGS
# ============================================================

CAMERA_SOURCE: Union[str, int] = "http://192.168.0.96:81/stream"
GIGA_HOST = "192.168.0.47"
GIGA_PORT = 5000

CONNECTION_TIMEOUT_SECONDS = 3.0
RECONNECT_DELAY_SECONDS = 0.25


# ============================================================
# WINDOWS / FILES
# ============================================================

WINDOW_NAME = "Shape-aware tennis-ball tracker"
DIAGNOSTIC_WINDOW_NAME = "Detector diagnostics"

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
CALIBRATION_FILE = SCRIPT_DIRECTORY / "tennis_ball_calibration.json"
COLOUR_MODEL_FILE = SCRIPT_DIRECTORY / "tennis_ball_colour_model.npz"


# ============================================================
# SHAPE-AWARE DETECTOR SETTINGS
# ============================================================

# Very broad yellow/green limits. The learned histogram is the main colour
# detector; these limits only reject clearly impossible pixels.
BROAD_HSV_LOWER = np.array((12, 35, 35), dtype=np.uint8)
BROAD_HSV_UPPER = np.array((90, 255, 255), dtype=np.uint8)

HISTOGRAM_H_BINS = 36
HISTOGRAM_S_BINS = 32
COLOUR_PROBABILITY_THRESHOLD = 34

MINIMUM_CONTOUR_AREA = 14.0
MINIMUM_BALL_DIAMETER_PIXELS = 4.0
MAXIMUM_BALL_DIAMETER_FRAME_FRACTION = 0.28
MAXIMUM_BLUR_ASPECT_RATIO = 4.8
MINIMUM_SOLIDITY = 0.48
MINIMUM_CANDIDATE_QUALITY = 0.34

OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 5

# Frame-difference motion is a scoring bonus, not a hard requirement.
# This is important because the ball may be stationary during calibration
# and the camera itself moves with the servos.
MOTION_THRESHOLD = 20
MAXIMUM_RELIABLE_MOTION_FRACTION = 0.30

# Association gate used after a track has been acquired.
MINIMUM_ASSOCIATION_GATE_PIXELS = 55.0
ASSOCIATION_GATE_DIAMETERS = 8.0
ASSOCIATION_GATE_FRAME_FRACTION = 0.18
MAXIMUM_DIAMETER_RATIO_CHANGE = 3.0
MAXIMUM_LOST_FRAMES = 15

# Hold the ball inside this centre box and press B once to learn its colour.
SAMPLE_BOX_FRACTION = 0.22

# Explicit circular-shape recognition. HoughCircles searches for circular edge
# patterns, while Hu-moment matching provides a contour-based fallback.
HOUGH_DP = 1.20
HOUGH_MIN_DISTANCE_PIXELS = 14.0
HOUGH_CANNY_THRESHOLD = 110.0
HOUGH_ACCUMULATOR_THRESHOLD = 13.0
HOUGH_MIN_RADIUS_PIXELS = 3
HOUGH_MAX_RADIUS_FRAME_FRACTION = 0.16

EDGE_CANNY_LOW = 55
EDGE_CANNY_HIGH = 140
EDGE_RING_HALF_WIDTH = 2
MINIMUM_EDGE_SUPPORT = 0.045
MINIMUM_ANGULAR_EDGE_COVERAGE = 0.70
MINIMUM_INSIDE_COLOUR_FRACTION = 0.18
MINIMUM_INSIDE_OUTSIDE_COLOUR_CONTRAST = 0.10
MINIMUM_HOUGH_QUALITY = 0.34

# Hu-moment fallback tolerates a mildly blurred ellipse, but rejects elongated
# streaks and irregular blobs.
MAXIMUM_HU_SHAPE_DISTANCE = 0.28
MINIMUM_FALLBACK_CIRCULARITY = 0.50
MINIMUM_ENCLOSING_CIRCLE_FILL = 0.68
MINIMUM_FALLBACK_QUALITY = 0.40


# ============================================================
# KALMAN FILTER / PREDICTION
# ============================================================

PREDICTION_SECONDS = 0.10
TRACK_TRAIL_LENGTH = 35

# Base process and measurement variances. Measurement covariance is adjusted
# per detection: strong candidates are trusted more than weak candidates.
KALMAN_POSITION_PROCESS_VARIANCE = 1.2
KALMAN_VELOCITY_PROCESS_VARIANCE = 70.0
KALMAN_DIAMETER_PROCESS_VARIANCE = 0.8
KALMAN_DIAMETER_RATE_PROCESS_VARIANCE = 15.0


# ============================================================
# DISTANCE SETTINGS
# ============================================================

BALL_DIAMETER_METRES = 0.067
CALIBRATION_DISTANCE_METRES = 1.0
DISTANCE_SMOOTHING_ALPHA = 0.22
FIXED_FOCAL_LENGTH_PIXELS: Optional[float] = None


# ============================================================
# SERVO SETTINGS
# ============================================================

PAN_MIN = 0.0
PAN_MAX = 180.0
TILT_MIN = 0.0
TILT_MAX = 80.0

PAN_START = 90.0
TILT_START = 40.0

PAN_DIRECTION = -1.0
TILT_DIRECTION = -1.0

DEAD_ZONE_X = 0.075
DEAD_ZONE_Y = 0.090

MINIMUM_PAN_STEP = 0.6
MAXIMUM_PAN_STEP = 8.0
MINIMUM_TILT_STEP = 0.5
MAXIMUM_TILT_STEP = 5.0

COMMAND_INTERVAL_SECONDS = 0.04
MANUAL_PAN_STEP = 5.0
MANUAL_TILT_STEP = 3.0


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class BallDetection:
    x: float
    y: float
    diameter: float
    major_axis: float
    angle: float
    area: float
    circularity: float
    fill_ratio: float
    solidity: float
    aspect_ratio: float
    colour_score: float
    motion_score: float
    quality: float

    @property
    def radius(self) -> float:
        return self.diameter / 2.0


# ============================================================
# LOW-LATENCY LATEST-FRAME CAMERA
# ============================================================

class LatestFrameCamera:
    """Continuously drains the stream and exposes only the newest frame."""

    def __init__(self, source: Union[str, int]) -> None:
        self.source = source
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_error_message = 0.0

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="camera-reader",
            daemon=True,
        )
        self._thread.start()

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        print(f"Opening camera: {self.source}")
        capture = cv2.VideoCapture(self.source)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not capture.isOpened():
            capture.release()
            return None

        return capture

    def _reader_loop(self) -> None:
        while self._running:
            if self._capture is None:
                self._capture = self._open_capture()

                if self._capture is None:
                    now = time.monotonic()
                    if now - self._last_error_message >= 2.0:
                        print("Camera unavailable; retrying")
                        self._last_error_message = now
                    time.sleep(0.5)
                    continue

            frame_ok, frame = self._capture.read()

            if not frame_ok or frame is None:
                print("Camera stream lost; reconnecting")
                self._capture.release()
                self._capture = None
                time.sleep(0.25)
                continue

            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def read(
        self,
        after_sequence: int,
        timeout_seconds: float = 2.0,
    ) -> tuple[bool, Optional[np.ndarray], int]:
        deadline = time.monotonic() + timeout_seconds

        with self._condition:
            while self._running and self._sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False, None, after_sequence
                self._condition.wait(timeout=remaining)

            if self._frame is None:
                return False, None, after_sequence

            return True, self._frame.copy(), self._sequence

    def close(self) -> None:
        self._running = False

        with self._condition:
            self._condition.notify_all()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


# ============================================================
# WI-FI SERVO CONTROLLER
# ============================================================

class WiFiServoController:
    def __init__(
        self,
        host: str,
        port: int,
        starting_pan: float,
        starting_tilt: float,
    ) -> None:
        self.host = host
        self.port = port
        self.connection: Optional[socket.socket] = None
        self.pan = self.clamp(starting_pan, PAN_MIN, PAN_MAX)
        self.tilt = self.clamp(starting_tilt, TILT_MIN, TILT_MAX)

        self.connect()
        self.move_to(self.pan, self.tilt, force=True)

    @staticmethod
    def clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def connect(self) -> None:
        self.close()
        print(f"Connecting to GIGA at {self.host}:{self.port}")

        try:
            connection = socket.create_connection(
                (self.host, self.port),
                timeout=CONNECTION_TIMEOUT_SECONDS,
            )
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            connection.settimeout(CONNECTION_TIMEOUT_SECONDS)
            self.connection = connection
        except OSError as error:
            self.connection = None
            raise RuntimeError(
                f"Could not connect to GIGA at {self.host}:{self.port}: {error}"
            ) from error

        print("Connected to GIGA")

    def send_command(self, command: str) -> None:
        payload = (command.rstrip("\r\n") + "\n").encode("ascii")
        last_error: Optional[OSError] = None

        for attempt in range(2):
            try:
                if self.connection is None:
                    self.connect()

                if self.connection is None:
                    raise RuntimeError("GIGA connection is unavailable")

                self.connection.sendall(payload)
                return
            except OSError as error:
                last_error = error
                print("GIGA connection failed; reconnecting")
                self.close()

                if attempt == 0:
                    time.sleep(RECONNECT_DELAY_SECONDS)

        raise RuntimeError(f"Could not send command to GIGA: {last_error}")

    def move_to(self, pan: float, tilt: float, force: bool = False) -> None:
        new_pan = self.clamp(pan, PAN_MIN, PAN_MAX)
        new_tilt = self.clamp(tilt, TILT_MIN, TILT_MAX)

        changed = (
            abs(new_pan - self.pan) >= 0.25
            or abs(new_tilt - self.tilt) >= 0.25
        )

        self.pan = new_pan
        self.tilt = new_tilt

        if not changed and not force:
            return

        self.send_command(f"P{round(self.pan)},{round(self.tilt)}")

    def move_relative(self, pan_change: float, tilt_change: float) -> None:
        self.move_to(
            pan=self.pan + pan_change,
            tilt=self.tilt + tilt_change,
        )

    def centre(self) -> None:
        self.move_to(PAN_START, TILT_START, force=True)

    def close(self) -> None:
        if self.connection is None:
            return

        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

        try:
            self.connection.close()
        except OSError:
            pass

        self.connection = None


# ============================================================
# SAVED COLOUR MODEL
# ============================================================

class TennisBallColourModel:
    """Learns and saves a 2-D HSV histogram from a ball sample."""

    def __init__(self) -> None:
        self.histogram: Optional[np.ndarray] = None
        self.load()

    @property
    def is_trained(self) -> bool:
        return self.histogram is not None

    def load(self) -> None:
        try:
            with np.load(COLOUR_MODEL_FILE) as data:
                histogram = np.asarray(data["histogram"], dtype=np.float32)

            expected_shape = (HISTOGRAM_H_BINS, HISTOGRAM_S_BINS)
            if histogram.shape != expected_shape:
                raise ValueError(
                    f"expected histogram shape {expected_shape}, "
                    f"found {histogram.shape}"
                )

            if not np.all(np.isfinite(histogram)) or histogram.max() <= 0.0:
                raise ValueError("histogram is empty or invalid")

            self.histogram = histogram
            print(f"Loaded tennis-ball colour model: {COLOUR_MODEL_FILE}")
        except FileNotFoundError:
            print(
                "No saved ball-colour model. Hold the ball inside the "
                "centre box and press B once."
            )
        except (OSError, ValueError, KeyError) as error:
            print(f"Could not load colour model: {error}")
            self.histogram = None

    def learn(self, frame: np.ndarray, sample_box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = sample_box
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            raise ValueError("sample region is empty")

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        circle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            circle_mask,
            (width // 2, height // 2),
            max(3, int(min(width, height) * 0.40)),
            255,
            -1,
        )

        broad_mask = cv2.inRange(hsv, BROAD_HSV_LOWER, BROAD_HSV_UPPER)
        sample_mask = cv2.bitwise_and(circle_mask, broad_mask)

        if cv2.countNonZero(sample_mask) < 80:
            raise ValueError(
                "not enough yellow/green pixels in the sample box; "
                "move the ball into the box and improve lighting"
            )

        histogram = cv2.calcHist(
            [hsv],
            [0, 1],
            sample_mask,
            [HISTOGRAM_H_BINS, HISTOGRAM_S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        histogram = histogram.astype(np.float32)

        if histogram.max() <= 0.0:
            raise ValueError("could not calculate a usable colour histogram")

        self.histogram = histogram
        np.savez_compressed(COLOUR_MODEL_FILE, histogram=histogram)
        print(f"Saved tennis-ball colour model: {COLOUR_MODEL_FILE}")

    def clear(self) -> None:
        self.histogram = None
        try:
            COLOUR_MODEL_FILE.unlink()
        except FileNotFoundError:
            pass
        print("Ball-colour model cleared; press B to learn it again")

    def probability(self, hsv: np.ndarray) -> np.ndarray:
        if self.histogram is None:
            return cv2.inRange(hsv, BROAD_HSV_LOWER, BROAD_HSV_UPPER)

        probability = cv2.calcBackProject(
            [hsv],
            [0, 1],
            self.histogram,
            [0, 180, 0, 256],
            1.0,
        )
        return cv2.GaussianBlur(probability, (7, 7), 0)


# ============================================================
# KALMAN BALL TRACKER
# ============================================================

class KalmanBallTracker:
    """Tracks x, y, velocity, apparent diameter and diameter rate."""

    def __init__(self, maximum_lost_frames: int) -> None:
        self.maximum_lost_frames = maximum_lost_frames
        self.filter = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
        self.filter.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.last_time = 0.0
        self.lost_frames = 0
        self.filter.statePost = np.zeros((6, 1), dtype=np.float32)
        self.filter.statePre = np.zeros((6, 1), dtype=np.float32)
        self.filter.errorCovPost = np.eye(6, dtype=np.float32) * 100.0

    @property
    def x(self) -> float:
        return float(self.filter.statePost[0, 0])

    @property
    def y(self) -> float:
        return float(self.filter.statePost[1, 0])

    @property
    def vx(self) -> float:
        return float(self.filter.statePost[2, 0])

    @property
    def vy(self) -> float:
        return float(self.filter.statePost[3, 0])

    @property
    def diameter(self) -> float:
        return max(0.0, float(self.filter.statePost[4, 0]))

    def _set_transition(self, dt: float) -> None:
        self.filter.transitionMatrix = np.array(
            [
                [1, 0, dt, 0, 0, 0],
                [0, 1, 0, dt, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        self.filter.processNoiseCov = np.diag(
            [
                KALMAN_POSITION_PROCESS_VARIANCE * max(dt, 0.01),
                KALMAN_POSITION_PROCESS_VARIANCE * max(dt, 0.01),
                KALMAN_VELOCITY_PROCESS_VARIANCE * max(dt, 0.01),
                KALMAN_VELOCITY_PROCESS_VARIANCE * max(dt, 0.01),
                KALMAN_DIAMETER_PROCESS_VARIANCE * max(dt, 0.01),
                KALMAN_DIAMETER_RATE_PROCESS_VARIANCE * max(dt, 0.01),
            ]
        ).astype(np.float32)

    def predict(self, timestamp: float) -> Optional[tuple[float, float, float]]:
        if not self.active:
            return None

        dt = max(1.0 / 240.0, min(0.25, timestamp - self.last_time))
        self._set_transition(dt)
        prediction = self.filter.predict()
        self.last_time = timestamp

        return (
            float(prediction[0, 0]),
            float(prediction[1, 0]),
            max(0.0, float(prediction[4, 0])),
        )

    def update(self, detection: BallDetection, timestamp: float) -> None:
        if not self.active:
            state = np.array(
                [
                    detection.x,
                    detection.y,
                    0.0,
                    0.0,
                    detection.diameter,
                    0.0,
                ],
                dtype=np.float32,
            ).reshape(6, 1)
            self.filter.statePost = state.copy()
            self.filter.statePre = state.copy()
            self.filter.errorCovPost = np.diag(
                [25.0, 25.0, 400.0, 400.0, 16.0, 100.0]
            ).astype(np.float32)
            self.active = True
            self.last_time = timestamp
            self.lost_frames = 0
            print("Ball lock acquired")
            return

        quality = max(0.0, min(1.0, detection.quality))
        position_variance = 4.0 + (1.0 - quality) * 40.0
        diameter_variance = 2.0 + (1.0 - quality) * 20.0
        self.filter.measurementNoiseCov = np.diag(
            [position_variance, position_variance, diameter_variance]
        ).astype(np.float32)

        measurement = np.array(
            [detection.x, detection.y, detection.diameter],
            dtype=np.float32,
        ).reshape(3, 1)
        self.filter.correct(measurement)
        self.last_time = timestamp
        self.lost_frames = 0

    def mark_missed(self) -> None:
        if not self.active:
            return

        self.lost_frames += 1
        if self.lost_frames > self.maximum_lost_frames:
            print("Ball lock lost")
            self.reset()

    def future_position(self, horizon_seconds: float) -> Optional[tuple[float, float]]:
        if not self.active:
            return None

        return (
            self.x + self.vx * horizon_seconds,
            self.y + self.vy * horizon_seconds,
        )


# ============================================================
# DISTANCE ESTIMATOR
# ============================================================

class DistanceEstimator:
    def __init__(self) -> None:
        self.focal_length_pixels: Optional[float] = FIXED_FOCAL_LENGTH_PIXELS
        self.filtered_distance: Optional[float] = None

        if self.focal_length_pixels is None:
            self._load()

    def _load(self) -> None:
        try:
            data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
            value = float(data["focal_length_pixels"])

            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("invalid focal length")

            self.focal_length_pixels = value
            print(
                f"Loaded distance calibration: {value:.2f} px\n"
                f"Calibration file: {CALIBRATION_FILE}"
            )
        except FileNotFoundError:
            print(
                "No saved distance calibration. Hold the ball at "
                f"{CALIBRATION_DISTANCE_METRES:.2f} m and press K once."
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"Could not load saved distance calibration: {error}")

    def calibrate(self, measured_diameter_pixels: float) -> float:
        if measured_diameter_pixels <= 0.0:
            raise ValueError("ball diameter must be positive")

        self.focal_length_pixels = (
            measured_diameter_pixels
            * CALIBRATION_DISTANCE_METRES
            / BALL_DIAMETER_METRES
        )
        self.filtered_distance = None

        data = {
            "focal_length_pixels": self.focal_length_pixels,
            "ball_diameter_metres": BALL_DIAMETER_METRES,
            "calibration_distance_metres": CALIBRATION_DISTANCE_METRES,
        }
        temporary_file = CALIBRATION_FILE.with_suffix(".tmp")
        temporary_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary_file.replace(CALIBRATION_FILE)

        print(
            f"Distance calibration saved: {self.focal_length_pixels:.2f} px\n"
            f"Calibration file: {CALIBRATION_FILE}"
        )
        return self.focal_length_pixels

    def estimate(self, diameter_pixels: float) -> Optional[float]:
        if self.focal_length_pixels is None or diameter_pixels <= 0.0:
            return None

        raw_distance = (
            self.focal_length_pixels
            * BALL_DIAMETER_METRES
            / diameter_pixels
        )

        if self.filtered_distance is None:
            self.filtered_distance = raw_distance
        else:
            self.filtered_distance = (
                DISTANCE_SMOOTHING_ALPHA * raw_distance
                + (1.0 - DISTANCE_SMOOTHING_ALPHA) * self.filtered_distance
            )

        return self.filtered_distance


# ============================================================
# HOUGH-CIRCLE + HU-MOMENT SHAPE DETECTOR
# ============================================================

class ShapeAwareTennisBallDetector:
    """Detects a tennis ball using circular edges, shape, colour and motion."""

    def __init__(self, colour_model: TennisBallColourModel) -> None:
        self.colour_model = colour_model
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE),
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE),
        )
        self.previous_gray: Optional[np.ndarray] = None
        self.circle_template = self._make_circle_template()

    @staticmethod
    def _make_circle_template() -> np.ndarray:
        image = np.zeros((128, 128), dtype=np.uint8)
        cv2.circle(image, (64, 64), 42, 255, -1)
        contours, _ = cv2.findContours(
            image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours[0]

    @staticmethod
    def _circle_masks(
        shape: tuple[int, int],
        centre_x: float,
        centre_y: float,
        radius: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        height, width = shape
        margin = int(math.ceil(radius * 1.42 + EDGE_RING_HALF_WIDTH + 3))
        x1 = max(0, int(round(centre_x)) - margin)
        y1 = max(0, int(round(centre_y)) - margin)
        x2 = min(width, int(round(centre_x)) + margin + 1)
        y2 = min(height, int(round(centre_y)) + margin + 1)

        roi_height = y2 - y1
        roi_width = x2 - x1
        local_x = int(round(centre_x - x1))
        local_y = int(round(centre_y - y1))
        integer_radius = max(2, int(round(radius)))

        inside = np.zeros((roi_height, roi_width), dtype=np.uint8)
        edge_outer = np.zeros_like(inside)
        edge_inner = np.zeros_like(inside)
        background_outer = np.zeros_like(inside)
        background_inner = np.zeros_like(inside)

        cv2.circle(
            inside,
            (local_x, local_y),
            max(1, int(round(integer_radius * 0.86))),
            255,
            -1,
        )
        cv2.circle(
            edge_outer,
            (local_x, local_y),
            integer_radius + EDGE_RING_HALF_WIDTH,
            255,
            -1,
        )
        cv2.circle(
            edge_inner,
            (local_x, local_y),
            max(1, integer_radius - EDGE_RING_HALF_WIDTH),
            255,
            -1,
        )
        edge_ring = cv2.subtract(edge_outer, edge_inner)

        cv2.circle(
            background_outer,
            (local_x, local_y),
            max(integer_radius + 3, int(round(integer_radius * 1.38))),
            255,
            -1,
        )
        cv2.circle(
            background_inner,
            (local_x, local_y),
            max(integer_radius + 1, int(round(integer_radius * 1.08))),
            255,
            -1,
        )
        background_ring = cv2.subtract(background_outer, background_inner)
        return inside, edge_ring, background_ring, (x1, y1, x2, y2)

    @staticmethod
    def _angular_edge_coverage(
        edge_map: np.ndarray,
        centre_x: float,
        centre_y: float,
        radius: float,
        samples: int = 72,
    ) -> float:
        height, width = edge_map.shape[:2]
        hits = 0
        usable = 0

        for index in range(samples):
            angle = 2.0 * math.pi * index / samples
            found = False
            point_usable = False

            for radial_offset in range(-EDGE_RING_HALF_WIDTH, EDGE_RING_HALF_WIDTH + 1):
                sample_radius = max(1.0, radius + radial_offset)
                x = int(round(centre_x + sample_radius * math.cos(angle)))
                y = int(round(centre_y + sample_radius * math.sin(angle)))

                if 0 <= x < width and 0 <= y < height:
                    point_usable = True
                    x1 = max(0, x - 1)
                    y1 = max(0, y - 1)
                    x2 = min(width, x + 2)
                    y2 = min(height, y + 2)
                    if cv2.countNonZero(edge_map[y1:y2, x1:x2]) > 0:
                        found = True
                        break

            if point_usable:
                usable += 1
                if found:
                    hits += 1

        return hits / max(usable, 1)

    @staticmethod
    def _deduplicate(
        detections: list[BallDetection],
    ) -> list[BallDetection]:
        ordered = sorted(detections, key=lambda item: item.quality, reverse=True)
        kept: list[BallDetection] = []

        for detection in ordered:
            duplicate = False
            for existing in kept:
                centre_distance = math.hypot(
                    detection.x - existing.x,
                    detection.y - existing.y,
                )
                radius_scale = max(
                    3.0,
                    0.40 * max(detection.diameter, existing.diameter),
                )
                diameter_ratio = detection.diameter / max(existing.diameter, 1.0)
                if (
                    centre_distance <= radius_scale
                    and 0.65 <= diameter_ratio <= 1.55
                ):
                    duplicate = True
                    break

            if not duplicate:
                kept.append(detection)

        return kept

    def _motion_mask(self, gray: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.previous_gray is None:
            motion_mask = np.zeros_like(gray)
        else:
            difference = cv2.absdiff(gray, self.previous_gray)
            _, motion_mask = cv2.threshold(
                difference,
                MOTION_THRESHOLD,
                255,
                cv2.THRESH_BINARY,
            )
            motion_mask = cv2.dilate(
                motion_mask,
                self.open_kernel,
                iterations=2,
            )

        self.previous_gray = gray
        motion_fraction = cv2.countNonZero(motion_mask) / float(motion_mask.size)
        return motion_mask, motion_fraction <= MAXIMUM_RELIABLE_MOTION_FRACTION

    def _score_circle(
        self,
        centre_x: float,
        centre_y: float,
        radius: float,
        colour_probability: np.ndarray,
        colour_mask: np.ndarray,
        edge_map: np.ndarray,
        motion_mask: np.ndarray,
        motion_reliable: bool,
    ) -> Optional[BallDetection]:
        (
            inside_mask,
            edge_ring_mask,
            background_ring_mask,
            (x1, y1, x2, y2),
        ) = self._circle_masks(
            colour_mask.shape,
            centre_x,
            centre_y,
            radius,
        )

        probability_roi = colour_probability[y1:y2, x1:x2]
        colour_roi = colour_mask[y1:y2, x1:x2]
        edge_roi = edge_map[y1:y2, x1:x2]
        motion_roi = motion_mask[y1:y2, x1:x2]

        inside_pixels = max(1, cv2.countNonZero(inside_mask))
        edge_ring_pixels = max(1, cv2.countNonZero(edge_ring_mask))
        background_pixels = max(1, cv2.countNonZero(background_ring_mask))

        colour_score = float(
            cv2.mean(probability_roi, mask=inside_mask)[0] / 255.0
        )
        inside_colour_fraction = float(
            cv2.countNonZero(cv2.bitwise_and(colour_roi, inside_mask))
            / inside_pixels
        )
        outside_colour_fraction = float(
            cv2.countNonZero(cv2.bitwise_and(colour_roi, background_ring_mask))
            / background_pixels
        )
        colour_contrast = max(
            0.0,
            inside_colour_fraction - outside_colour_fraction,
        )
        edge_support = float(
            cv2.countNonZero(cv2.bitwise_and(edge_roi, edge_ring_mask))
            / edge_ring_pixels
        )
        angular_coverage = self._angular_edge_coverage(
            edge_map,
            centre_x,
            centre_y,
            radius,
        )

        if motion_reliable:
            motion_score = float(
                cv2.mean(motion_roi, mask=inside_mask)[0] / 255.0
            )
        else:
            motion_score = 0.50

        if inside_colour_fraction < MINIMUM_INSIDE_COLOUR_FRACTION:
            return None
        if colour_contrast < MINIMUM_INSIDE_OUTSIDE_COLOUR_CONTRAST:
            return None
        if edge_support < MINIMUM_EDGE_SUPPORT:
            return None
        if angular_coverage < MINIMUM_ANGULAR_EDGE_COVERAGE:
            return None

        edge_score = min(1.0, edge_support / 0.20)
        angular_score = min(1.0, angular_coverage / 0.72)
        colour_fill_score = min(1.0, inside_colour_fraction / 0.72)
        contrast_score = min(1.0, colour_contrast / 0.55)

        quality = (
            0.25 * colour_score
            + 0.17 * colour_fill_score
            + 0.16 * contrast_score
            + 0.15 * edge_score
            + 0.23 * angular_score
            + 0.04 * motion_score
        )

        if quality < MINIMUM_HOUGH_QUALITY:
            return None

        diameter = 2.0 * radius
        return BallDetection(
            x=float(centre_x),
            y=float(centre_y),
            diameter=float(diameter),
            major_axis=float(diameter),
            angle=0.0,
            area=float(math.pi * radius * radius),
            circularity=angular_coverage,
            fill_ratio=inside_colour_fraction,
            solidity=1.0,
            aspect_ratio=1.0,
            colour_score=colour_score,
            motion_score=motion_score,
            quality=quality,
        )

    def _contour_fallback(
        self,
        colour_mask: np.ndarray,
        colour_probability: np.ndarray,
        motion_mask: np.ndarray,
        motion_reliable: bool,
        maximum_diameter: float,
    ) -> list[BallDetection]:
        contours, _ = cv2.findContours(
            colour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[BallDetection] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < MINIMUM_CONTOUR_AREA:
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue

            circularity = max(
                0.0,
                min(1.0, 4.0 * math.pi * area / (perimeter * perimeter)),
            )
            if circularity < MINIMUM_FALLBACK_CIRCULARITY:
                continue

            shape_distance = float(
                cv2.matchShapes(
                    contour,
                    self.circle_template,
                    cv2.CONTOURS_MATCH_I1,
                    0.0,
                )
            )
            if shape_distance > MAXIMUM_HU_SHAPE_DISTANCE:
                continue

            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / max(hull_area, 1.0)
            if solidity < MINIMUM_SOLIDITY:
                continue

            if len(contour) >= 5:
                (centre_x, centre_y), axes, angle = cv2.fitEllipse(contour)
                axis_a, axis_b = float(axes[0]), float(axes[1])
            else:
                (centre_x, centre_y), radius = cv2.minEnclosingCircle(contour)
                axis_a = axis_b = 2.0 * float(radius)
                angle = 0.0

            minor_axis = min(axis_a, axis_b)
            major_axis = max(axis_a, axis_b)
            if minor_axis <= 0.0:
                continue

            aspect_ratio = major_axis / minor_axis
            if aspect_ratio > 1.65:
                continue

            (_, _), enclosing_radius = cv2.minEnclosingCircle(contour)
            enclosing_fill = area / max(
                math.pi * float(enclosing_radius) * float(enclosing_radius),
                1.0,
            )
            if enclosing_fill < MINIMUM_ENCLOSING_CIRCLE_FILL:
                continue

            diameter = minor_axis
            if not (
                MINIMUM_BALL_DIAMETER_PIXELS
                <= diameter
                <= maximum_diameter
            ):
                continue

            ellipse_area = math.pi * major_axis * minor_axis / 4.0
            fill_ratio = area / max(ellipse_area, 1.0)
            if not (0.30 <= fill_ratio <= 1.35):
                continue

            x, y, width, height = cv2.boundingRect(contour)
            local = contour.copy()
            local[:, 0, 0] -= x
            local[:, 0, 1] -= y
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask, [local], -1, 255, -1)

            probability_roi = colour_probability[y:y + height, x:x + width]
            motion_roi = motion_mask[y:y + height, x:x + width]
            colour_score = float(cv2.mean(probability_roi, mask=mask)[0] / 255.0)
            motion_score = (
                float(cv2.mean(motion_roi, mask=mask)[0] / 255.0)
                if motion_reliable
                else 0.50
            )

            shape_score = math.exp(-5.0 * shape_distance)
            aspect_score = math.exp(-1.8 * (aspect_ratio - 1.0))
            fill_score = max(0.0, 1.0 - abs(fill_ratio - 0.78) / 0.78)

            enclosing_score = min(
                1.0,
                max(0.0, (enclosing_fill - 0.60) / 0.35),
            )
            quality = (
                0.30 * colour_score
                + 0.18 * shape_score
                + 0.16 * circularity
                + 0.12 * aspect_score
                + 0.09 * fill_score
                + 0.11 * enclosing_score
                + 0.04 * motion_score
            )
            if quality < MINIMUM_FALLBACK_QUALITY:
                continue

            candidates.append(
                BallDetection(
                    x=float(centre_x),
                    y=float(centre_y),
                    diameter=float(diameter),
                    major_axis=float(major_axis),
                    angle=float(angle),
                    area=area,
                    circularity=circularity,
                    fill_ratio=fill_ratio,
                    solidity=solidity,
                    aspect_ratio=aspect_ratio,
                    colour_score=colour_score,
                    motion_score=motion_score,
                    quality=quality,
                )
            )

        return candidates

    def detect(
        self,
        frame: np.ndarray,
    ) -> tuple[
        list[BallDetection],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
    ]:
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        colour_probability = self.colour_model.probability(hsv)
        broad_mask = cv2.inRange(hsv, BROAD_HSV_LOWER, BROAD_HSV_UPPER)

        if self.colour_model.is_trained:
            _, probability_mask = cv2.threshold(
                colour_probability,
                COLOUR_PROBABILITY_THRESHOLD,
                255,
                cv2.THRESH_BINARY,
            )
            colour_mask = cv2.bitwise_and(probability_mask, broad_mask)
        else:
            colour_mask = broad_mask

        colour_mask = cv2.morphologyEx(
            colour_mask,
            cv2.MORPH_OPEN,
            self.open_kernel,
            iterations=1,
        )
        colour_mask = cv2.morphologyEx(
            colour_mask,
            cv2.MORPH_CLOSE,
            self.close_kernel,
            iterations=2,
        )

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        motion_mask, motion_reliable = self._motion_mask(gray)
        edge_map = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

        # Dilating the colour mask gives HoughCircles access to the ball edge,
        # which often lies just outside a conservative colour threshold.
        search_mask = cv2.dilate(colour_mask, self.close_kernel, iterations=1)
        circle_input = cv2.bitwise_and(gray, gray, mask=search_mask)
        circle_input = cv2.medianBlur(circle_input, 5)

        frame_height, frame_width = frame.shape[:2]
        maximum_radius = max(
            HOUGH_MIN_RADIUS_PIXELS + 1,
            int(min(frame_width, frame_height) * HOUGH_MAX_RADIUS_FRAME_FRACTION),
        )

        raw_circles = cv2.HoughCircles(
            circle_input,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DISTANCE_PIXELS,
            param1=HOUGH_CANNY_THRESHOLD,
            param2=HOUGH_ACCUMULATOR_THRESHOLD,
            minRadius=HOUGH_MIN_RADIUS_PIXELS,
            maxRadius=maximum_radius,
        )

        candidates: list[BallDetection] = []
        if raw_circles is not None:
            for centre_x, centre_y, radius in raw_circles[0]:
                candidate = self._score_circle(
                    float(centre_x),
                    float(centre_y),
                    float(radius),
                    colour_probability,
                    colour_mask,
                    edge_map,
                    motion_mask,
                    motion_reliable,
                )
                if candidate is not None:
                    candidates.append(candidate)

        # Hu-moment contour matching is a fallback for imperfect circles and
        # brief blur. Hough detections normally rank above fallback candidates.
        maximum_diameter = (
            min(frame_width, frame_height)
            * MAXIMUM_BALL_DIAMETER_FRAME_FRACTION
        )
        candidates.extend(
            self._contour_fallback(
                colour_mask,
                colour_probability,
                motion_mask,
                motion_reliable,
                maximum_diameter,
            )
        )

        return (
            self._deduplicate(candidates),
            colour_mask,
            colour_probability,
            motion_mask,
            edge_map,
            motion_reliable,
        )

    @staticmethod
    def select(
        detections: list[BallDetection],
        tracker: KalmanBallTracker,
        predicted: Optional[tuple[float, float, float]],
        frame_width: int,
        frame_height: int,
    ) -> Optional[BallDetection]:
        if not detections:
            return None

        if predicted is None or not tracker.active:
            return max(
                detections,
                key=lambda detection: (
                    detection.quality
                    + 0.018 * math.log1p(detection.area)
                ),
            )

        predicted_x, predicted_y, predicted_diameter = predicted
        frame_diagonal = math.hypot(frame_width, frame_height)
        reference_diameter = max(tracker.diameter, predicted_diameter, 1.0)
        gate = max(
            MINIMUM_ASSOCIATION_GATE_PIXELS,
            reference_diameter * ASSOCIATION_GATE_DIAMETERS,
            frame_diagonal * ASSOCIATION_GATE_FRAME_FRACTION,
        )

        acceptable: list[tuple[float, BallDetection]] = []
        for detection in detections:
            distance = math.hypot(
                detection.x - predicted_x,
                detection.y - predicted_y,
            )
            if distance > gate:
                continue

            diameter_ratio = detection.diameter / reference_diameter
            if not (
                1.0 / MAXIMUM_DIAMETER_RATIO_CHANGE
                <= diameter_ratio
                <= MAXIMUM_DIAMETER_RATIO_CHANGE
            ):
                continue

            score = (
                0.60 * (distance / gate)
                + 0.18 * abs(math.log(max(diameter_ratio, 1.0e-6)))
                + 0.22 * (1.0 - detection.quality)
            )
            acceptable.append((score, detection))

        if not acceptable:
            return None

        return min(acceptable, key=lambda item: item[0])[1]


# ============================================================
# SERVO MOVEMENT
# ============================================================

def calculate_axis_step(
    error: float,
    dead_zone: float,
    minimum_step: float,
    maximum_step: float,
    direction: float,
) -> float:
    if abs(error) <= dead_zone:
        return 0.0

    usable_range = max(0.001, 1.0 - dead_zone)
    normalized_error = min(1.0, (abs(error) - dead_zone) / usable_range)
    response = normalized_error ** 1.35
    step_size = minimum_step + response * (maximum_step - minimum_step)
    return direction * math.copysign(step_size, error)


def calculate_servo_steps(
    target_x: float,
    target_y: float,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    centre_x = frame_width / 2.0
    centre_y = frame_height / 2.0
    error_x = (target_x - centre_x) / centre_x
    error_y = (target_y - centre_y) / centre_y

    pan_step = calculate_axis_step(
        error_x,
        DEAD_ZONE_X,
        MINIMUM_PAN_STEP,
        MAXIMUM_PAN_STEP,
        PAN_DIRECTION,
    )
    tilt_step = calculate_axis_step(
        error_y,
        DEAD_ZONE_Y,
        MINIMUM_TILT_STEP,
        MAXIMUM_TILT_STEP,
        TILT_DIRECTION,
    )
    return pan_step, tilt_step


# ============================================================
# DISPLAY HELPERS
# ============================================================

def sample_box_for_frame(frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    box_size = max(50, int(min(width, height) * SAMPLE_BOX_FRACTION))
    centre_x = width // 2
    centre_y = height // 2
    return (
        centre_x - box_size // 2,
        centre_y - box_size // 2,
        centre_x + box_size // 2,
        centre_y + box_size // 2,
    )


def build_diagnostics(
    colour_mask: np.ndarray,
    probability: np.ndarray,
    motion_mask: np.ndarray,
    edge_map: np.ndarray,
    motion_reliable: bool,
) -> np.ndarray:
    mask_bgr = cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR)
    probability_bgr = cv2.applyColorMap(probability, cv2.COLORMAP_TURBO)
    motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    edge_bgr = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)

    height, width = colour_mask.shape[:2]
    labels = [
        (mask_bgr, "colour mask", (0, 255, 0)),
        (probability_bgr, "colour probability", (255, 255, 255)),
        (
            motion_bgr,
            "motion reliable" if motion_reliable else "motion ignored",
            (0, 255, 0) if motion_reliable else (0, 0, 255),
        ),
        (edge_bgr, "circle edge map", (0, 255, 255)),
    ]
    for image, label, colour in labels:
        cv2.putText(
            image,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
        )

    panel_width = max(1, width // 4)
    return cv2.hconcat(
        [cv2.resize(image, (panel_width, height)) for image, _, _ in labels]
    )


def draw_display(
    frame: np.ndarray,
    detections: list[BallDetection],
    selected: Optional[BallDetection],
    tracker: KalmanBallTracker,
    controller: WiFiServoController,
    distance_metres: Optional[float],
    tracking_enabled: bool,
    fps: float,
    trail: deque[tuple[int, int]],
    colour_model_trained: bool,
) -> None:
    height, width = frame.shape[:2]
    centre_x = width // 2
    centre_y = height // 2

    half_dead_width = int(width / 2.0 * DEAD_ZONE_X)
    half_dead_height = int(height / 2.0 * DEAD_ZONE_Y)
    cv2.rectangle(
        frame,
        (centre_x - half_dead_width, centre_y - half_dead_height),
        (centre_x + half_dead_width, centre_y + half_dead_height),
        (255, 255, 0),
        1,
    )

    sample_box = sample_box_for_frame(frame)
    cv2.rectangle(
        frame,
        (sample_box[0], sample_box[1]),
        (sample_box[2], sample_box[3]),
        (255, 0, 255),
        1,
    )

    for detection in detections:
        is_selected = detection is selected
        colour = (0, 255, 0) if is_selected else (130, 130, 130)
        thickness = 3 if is_selected else 1

        cv2.ellipse(
            frame,
            (
                (round(detection.x), round(detection.y)),
                (
                    max(2, round(detection.major_axis)),
                    max(2, round(detection.diameter)),
                ),
                detection.angle,
            ),
            colour,
            thickness,
        )
        cv2.putText(
            frame,
            f"{detection.quality:.2f}",
            (round(detection.x + detection.major_axis / 2), round(detection.y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            colour,
            1,
        )

    if tracker.active:
        filtered_point = (round(tracker.x), round(tracker.y))
        cv2.drawMarker(
            frame,
            filtered_point,
            (0, 255, 255),
            cv2.MARKER_CROSS,
            16,
            2,
        )

        predicted = tracker.future_position(PREDICTION_SECONDS)
        if predicted is not None:
            predicted_point = (round(predicted[0]), round(predicted[1]))
            cv2.circle(frame, predicted_point, 6, (0, 0, 255), 2)
            cv2.line(frame, filtered_point, predicted_point, (0, 0, 255), 1)

        trail.append(filtered_point)
    else:
        trail.clear()

    points = list(trail)
    for index in range(1, len(points)):
        cv2.line(frame, points[index - 1], points[index], (0, 200, 255), 1)

    status = "TRACKING" if tracking_enabled else "PAUSED"
    lock_status = "LOCK" if tracker.active else "SEARCH"
    model_status = "LEARNED" if colour_model_trained else "BROAD HSV"

    cv2.putText(
        frame,
        f"{status} | {lock_status} | {fps:.1f} FPS | colour {model_status}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"Pan {controller.pan:.0f}  Tilt {controller.tilt:.0f}",
        (10, 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
    )

    if distance_metres is None:
        distance_text = (
            f"Distance uncalibrated: hold ball at "
            f"{CALIBRATION_DISTANCE_METRES:.2f} m and press K"
        )
    else:
        distance_text = (
            f"Distance {distance_metres:.2f} m | "
            f"shape diameter {tracker.diameter:.1f} px"
        )

    cv2.putText(
        frame,
        distance_text,
        (10, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "Put ball in magenta box + B learn | X clear | K distance | Space auto",
        (10, height - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        frame,
        "R reset | A/D/W/S manual | C centre | Q quit",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    controller: Optional[WiFiServoController] = None
    camera: Optional[LatestFrameCamera] = None

    try:
        controller = WiFiServoController(
            host=GIGA_HOST,
            port=GIGA_PORT,
            starting_pan=PAN_START,
            starting_tilt=TILT_START,
        )

        camera = LatestFrameCamera(CAMERA_SOURCE)
        camera.start()

        colour_model = TennisBallColourModel()
        detector = ShapeAwareTennisBallDetector(colour_model)
        tracker = KalmanBallTracker(MAXIMUM_LOST_FRAMES)
        distance_estimator = DistanceEstimator()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 960, 720)
        cv2.namedWindow(DIAGNOSTIC_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(DIAGNOSTIC_WINDOW_NAME, 960, 300)

        tracking_enabled = True
        last_command_time = 0.0
        last_sequence = 0
        last_frame_time = time.monotonic()
        fps = 0.0
        trail: deque[tuple[int, int]] = deque(maxlen=TRACK_TRAIL_LENGTH)

        while True:
            frame_ok, frame, last_sequence = camera.read(
                after_sequence=last_sequence,
                timeout_seconds=2.0,
            )

            if not frame_ok or frame is None:
                print("Waiting for a camera frame")
                continue

            now = time.monotonic()
            frame_dt = max(1.0 / 240.0, now - last_frame_time)
            instantaneous_fps = 1.0 / frame_dt
            fps = (
                instantaneous_fps
                if fps == 0.0
                else 0.90 * fps + 0.10 * instantaneous_fps
            )
            last_frame_time = now

            predicted = tracker.predict(now)
            (
                detections,
                colour_mask,
                colour_probability,
                motion_mask,
                edge_map,
                motion_reliable,
            ) = detector.detect(frame)

            height, width = frame.shape[:2]
            selected = detector.select(
                detections,
                tracker,
                predicted,
                width,
                height,
            )

            if selected is not None:
                tracker.update(selected, now)
            else:
                tracker.mark_missed()

            distance_metres = None
            if tracker.active:
                distance_metres = distance_estimator.estimate(tracker.diameter)

            if (
                tracking_enabled
                and tracker.active
                and now - last_command_time >= COMMAND_INTERVAL_SECONDS
            ):
                target = tracker.future_position(PREDICTION_SECONDS)
                if target is not None:
                    pan_step, tilt_step = calculate_servo_steps(
                        target[0],
                        target[1],
                        width,
                        height,
                    )
                    if pan_step != 0.0 or tilt_step != 0.0:
                        controller.move_relative(pan_step, tilt_step)
                last_command_time = now

            draw_display(
                frame,
                detections,
                selected,
                tracker,
                controller,
                distance_metres,
                tracking_enabled,
                fps,
                trail,
                colour_model.is_trained,
            )
            diagnostics = build_diagnostics(
                colour_mask,
                colour_probability,
                motion_mask,
                edge_map,
                motion_reliable,
            )

            cv2.imshow(WINDOW_NAME, frame)
            cv2.imshow(DIAGNOSTIC_WINDOW_NAME, diagnostics)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                tracking_enabled = not tracking_enabled
                print("Tracking enabled" if tracking_enabled else "Tracking paused")

            elif key in (ord("b"), ord("B")):
                try:
                    colour_model.learn(frame, sample_box_for_frame(frame))
                    tracker.reset()
                    detector.previous_gray = None
                except (OSError, ValueError) as error:
                    print(f"Colour learning failed: {error}")

            elif key in (ord("x"), ord("X")):
                colour_model.clear()
                tracker.reset()

            elif key in (ord("r"), ord("R")):
                tracker.reset()
                trail.clear()
                print("Ball lock reset")

            elif key in (ord("k"), ord("K")):
                if tracker.active and tracker.diameter > 0.0:
                    try:
                        distance_estimator.calibrate(tracker.diameter)
                    except (OSError, ValueError) as error:
                        print(f"Distance calibration failed: {error}")
                else:
                    print("Distance calibration requires a locked, stationary ball")

            elif key in (ord("c"), ord("C")):
                tracking_enabled = False
                controller.centre()
                print("Servos centred; tracking paused")

            elif key in (ord("a"), ord("A")):
                tracking_enabled = False
                controller.move_relative(-MANUAL_PAN_STEP, 0.0)

            elif key in (ord("d"), ord("D")):
                tracking_enabled = False
                controller.move_relative(MANUAL_PAN_STEP, 0.0)

            elif key in (ord("w"), ord("W")):
                tracking_enabled = False
                controller.move_relative(0.0, MANUAL_TILT_STEP)

            elif key in (ord("s"), ord("S")):
                tracking_enabled = False
                controller.move_relative(0.0, -MANUAL_TILT_STEP)

    except KeyboardInterrupt:
        print("\nStopped")
    except RuntimeError as error:
        print(f"\nERROR: {error}")
    finally:
        if camera is not None:
            camera.close()
        if controller is not None:
            controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
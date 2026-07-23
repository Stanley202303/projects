from __future__ import annotations

import json
import math
import os
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
# USER SETTINGS
# ============================================================

CAMERA_SOURCE: Union[str, int] = "http://192.168.0.96:81/stream"
GIGA_HOST = "192.168.0.47"
GIGA_PORT = 5000

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "tennis_ball_best.pt"
COCO_MODEL_PATH = SCRIPT_DIR / "yolo11n.pt"
COCO_STRONG_MODEL_PATH = SCRIPT_DIR / "yolo11s.pt"
COLOUR_MODEL_PATH = SCRIPT_DIR / "tennis_ball_colour_model.npz"
CALIBRATION_FILE = SCRIPT_DIR / "tennis_ball_calibration.json"

# Custom model class name. The code also accepts class ID 0.
BALL_CLASS_NAME = "tennis_ball"

# Start at 640. Try 960 if the ball is very small and the computer/GPU is fast.
INFERENCE_SIZE = 960
DETECTION_CONFIDENCE = 0.10
DETECTION_IOU = 0.45
COCO_DETECTION_CONFIDENCE = 0.18
COCO_INFERENCE_SIZE = 640
DEVICE: Optional[Union[str, int]] = None  # None=automatic, 0=first CUDA GPU, "cpu"=CPU

# Run AI often while acquiring, then back off once locked so control stays responsive.
AI_ACQUIRE_EVERY_N_FRAMES = 1
AI_LOCKED_EVERY_N_FRAMES = 15
AI_RESULT_MAX_AGE_FRAMES = 3
COLOUR_ACQUIRE_EVERY_N_FRAMES = 1
COLOUR_LOCKED_EVERY_N_FRAMES = 12

# Drawing diagnostics is surprisingly expensive on small machines.
SHOW_DIAGNOSTIC_WINDOW = False
DRAW_CANDIDATES = False

# Require several consistent detections before moving the mount.
ACQUIRE_HITS = 2
ACQUIRE_MAX_JUMP_DIAMETERS = 3.0
ACQUIRE_MAX_DIAMETER_RATIO = 1.8
FAST_LOCK_FUSION_CONFIDENCE = 0.78
FAST_LOCK_AI_CONFIDENCE = 0.55
FAST_LOCK_COLOUR_CONFIDENCE = 0.84
FAST_LOCK_COLOUR_SHARE = 0.55
COLOUR_ONLY_ACQUIRE_MIN_CONFIDENCE = 0.82
COLOUR_ONLY_ACQUIRE_MIN_SHARE = 0.55
COLOUR_ONLY_ACQUIRE_MIN_MOTION = 0.08
COCO_ONLY_ACQUIRE_MIN_CONFIDENCE = 0.50
COCO_ONLY_ACQUIRE_MIN_DIAMETER = 24.0
COCO_ONLY_ACQUIRE_MIN_MOTION = 0.08
MAX_AI_MISSES = 8
MAX_TOTAL_MISSES = 36
PREDICT_ONLY_MAX_MISSES = 30

WINDOW_NAME = "AI fusion tennis-ball tracker"
DIAGNOSTIC_WINDOW = "AI mask / optical flow"

# Distance
BALL_DIAMETER_METRES = 0.067
CALIBRATION_DISTANCE_METRES = 1.0
DISTANCE_ALPHA = 0.22
DISTANCE_MEDIAN_WINDOW = 7
DISTANCE_MAX_JUMP_FRACTION = 0.45

# Prediction
PREDICTION_SECONDS = 0.06
OFFSCREEN_PREDICTION_SECONDS = 0.18

# Servo geometry
PAN_MIN, PAN_MAX = 0.0, 180.0
TILT_MIN, TILT_MAX = 0.0, 80.0
PAN_START, TILT_START = 90.0, 40.0
PAN_DIRECTION = -1.0
TILT_DIRECTION = -1.0

DEAD_ZONE_X = 0.032
DEAD_ZONE_Y = 0.042
PAN_GAIN = 6.6
TILT_GAIN = 4.4
PAN_D_GAIN = 0.12
TILT_D_GAIN = 0.08
PAN_I_GAIN = 0.0
TILT_I_GAIN = 0.0
PID_INTEGRAL_LIMIT = 0.22
PID_DERIVATIVE_LIMIT = 1.0
ERROR_FILTER_ALPHA = 0.52
MINIMUM_PAN_STEP, MAXIMUM_PAN_STEP = 0.20, 3.6
MINIMUM_TILT_STEP, MAXIMUM_TILT_STEP = 0.16, 2.6
MAXIMUM_PAN_STEP_CHANGE = 0.85
MAXIMUM_TILT_STEP_CHANGE = 0.65
COMMAND_INTERVAL_SECONDS = 0.010

MANUAL_PAN_STEP = 5.0
MANUAL_TILT_STEP = 3.0

# Higher process noise lets the tracker follow thrown-ball acceleration.
KALMAN_POSITION_PROCESS_VARIANCE = 3.0
KALMAN_VELOCITY_PROCESS_VARIANCE = 260.0
KALMAN_DIAMETER_PROCESS_VARIANCE = 1.4
KALMAN_DIAMETER_RATE_PROCESS_VARIANCE = 45.0
KALMAN_VELOCITY_MEASUREMENT_BLEND = 0.65

# Detection-association gates
MIN_BALL_DIAMETER = 4.0
MAX_BALL_FRAME_FRACTION = 0.35
MAX_ASSOCIATION_DISTANCE_FRACTION = 0.22
MAX_DIAMETER_RATIO = 3.0

# Shape validation. Segmentation masks are preferred; boxes use weaker checks.
MIN_MASK_CIRCULARITY = 0.38
MIN_MASK_SOLIDITY = 0.62
MAX_MASK_ASPECT_RATIO = 2.8

# Classic colour/shape fallback. The histogram is trained in HS colour space.
COLOUR_BACKPROJECT_THRESHOLD = 32
COLOUR_MIN_SATURATION = 55
COLOUR_MIN_VALUE = 55
COLOUR_HSV_LOWER = np.array([24, COLOUR_MIN_SATURATION, COLOUR_MIN_VALUE], dtype=np.uint8)
COLOUR_HSV_UPPER = np.array([50, 255, 255], dtype=np.uint8)
COLOUR_SAMPLE_HSV_LOWER = np.array([15, 35, 35], dtype=np.uint8)
COLOUR_SAMPLE_HSV_UPPER = np.array([70, 255, 255], dtype=np.uint8)
COLOUR_MIN_CIRCULARITY = 0.48
COLOUR_MIN_SOLIDITY = 0.70
COLOUR_MAX_ASPECT_RATIO = 2.0
COLOUR_MIN_AREA = 18.0
COLOUR_KERNEL_SIZE = 5
COLOUR_MIN_MASK_SHARE_UNPREDICTED = 0.40
COLOUR_MIN_MASK_SHARE_TRACKED = 0.08
HOUGH_DP = 1.2
HOUGH_MIN_DISTANCE = 16.0
HOUGH_CANNY_THRESHOLD = 110.0
HOUGH_ACCUMULATOR_THRESHOLD = 13.0
HOUGH_MIN_RADIUS = 3
HOUGH_MAX_RADIUS_FRAME_FRACTION = 0.18
HOUGH_MIN_COLOUR_FRACTION = 0.16
HOUGH_MIN_EDGE_FRACTION = 0.025

# Motion is used as corroborating evidence, not as the only detector.
MOTION_THRESHOLD = 18
MOTION_BLUR_SIZE = 5
MOTION_MIN_AREA_FRACTION = 0.08

# Put the ball in the centre box and press B to relearn its colour.
SAMPLE_BOX_FRACTION = 0.22
COLOUR_HISTOGRAM_H_BINS = 36
COLOUR_HISTOGRAM_S_BINS = 32

# Optical-flow fallback
FLOW_MAX_CORNERS = 50
FLOW_QUALITY = 0.01
FLOW_MIN_DISTANCE = 3
FLOW_MIN_GOOD_POINTS = 5
FLOW_FB_ERROR = 2.0
FLOW_MAX_JUMP_FRACTION = 0.12

CONNECTION_TIMEOUT_SECONDS = 3.0
RECONNECT_DELAY_SECONDS = 0.25


# ============================================================
# DATA
# ============================================================

@dataclass
class Observation:
    x: float
    y: float
    diameter: float
    confidence: float
    source: str
    mask: Optional[np.ndarray] = None
    circularity: float = 0.0
    solidity: float = 0.0
    aspect_ratio: float = 1.0
    colour_share: float = 0.0
    motion_score: float = 0.0


@dataclass
class DistanceEstimate:
    raw_metres: float
    filtered_metres: float
    diameter_pixels: float
    source: str


@dataclass
class AIDetectionResult:
    sequence: int
    observation: Optional[Observation]
    mask: np.ndarray
    candidates: list[Observation]


# ============================================================
# LOW-LATENCY CAMERA
# ============================================================

class LatestFrameCamera:
    def __init__(self, source: Union[str, int]) -> None:
        self.source = source
        self._condition = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._last_capture_time: Optional[float] = None
        self._capture_fps = 0.0

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _open(self) -> Optional[cv2.VideoCapture]:
        print(f"Opening camera: {self.source}")
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def _loop(self) -> None:
        while self._running:
            if self._capture is None:
                self._capture = self._open()
                if self._capture is None:
                    print("Camera unavailable; retrying")
                    time.sleep(0.5)
                    continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                print("Camera stream lost; reconnecting")
                self._capture.release()
                self._capture = None
                self._last_capture_time = None
                self._capture_fps = 0.0
                time.sleep(0.25)
                continue

            now = time.monotonic()
            if self._last_capture_time is not None:
                frame_dt = max(1.0 / 240.0, now - self._last_capture_time)
                instant_fps = 1.0 / frame_dt
                self._capture_fps = (
                    instant_fps
                    if self._capture_fps == 0.0
                    else 0.90 * self._capture_fps + 0.10 * instant_fps
                )
            self._last_capture_time = now

            with self._condition:
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()

        if self._capture is not None:
            self._capture.release()

    def read(self, after_sequence: int, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._running and self._sequence <= after_sequence:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, after_sequence
                self._condition.wait(remaining)

            if self._frame is None:
                return False, None, after_sequence
            return True, self._frame.copy(), self._sequence

    @property
    def fps(self) -> float:
        return self._capture_fps

    def stop(self) -> None:
        self._running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ============================================================
# GIGA SERVO CLIENT
# ============================================================

class WiFiServoController:
    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.connection: Optional[socket.socket] = None
        self.pan, self.tilt = PAN_START, TILT_START
        self._sent_pan: Optional[int] = None
        self._sent_tilt: Optional[int] = None
        self.connect()
        self.move_to(self.pan, self.tilt, force=True)

    @staticmethod
    def clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def connect(self) -> None:
        self.close()
        print(f"Connecting to GIGA at {self.host}:{self.port}")
        connection = socket.create_connection(
            (self.host, self.port),
            timeout=CONNECTION_TIMEOUT_SECONDS,
        )
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.connection = connection
        print("Connected to GIGA")

    def send(self, command: str) -> None:
        payload = (command.rstrip() + "\n").encode("ascii")
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                if self.connection is None:
                    self.connect()
                assert self.connection is not None
                self.connection.sendall(payload)
                return
            except (OSError, AssertionError) as error:
                last_error = error
                self.close()
                if attempt == 0:
                    time.sleep(RECONNECT_DELAY_SECONDS)
        raise RuntimeError(f"Could not send command to GIGA: {last_error}")

    def move_to(self, pan: float, tilt: float, force: bool = False) -> None:
        pan = self.clamp(pan, PAN_MIN, PAN_MAX)
        tilt = self.clamp(tilt, TILT_MIN, TILT_MAX)
        self.pan, self.tilt = pan, tilt
        rounded_pan = round(pan)
        rounded_tilt = round(tilt)
        changed = rounded_pan != self._sent_pan or rounded_tilt != self._sent_tilt
        if force or changed:
            self.send(f"P{rounded_pan},{rounded_tilt}")
            self._sent_pan = rounded_pan
            self._sent_tilt = rounded_tilt

    def move_relative(self, pan_change: float, tilt_change: float) -> None:
        self.move_to(self.pan + pan_change, self.tilt + tilt_change)

    def centre(self) -> None:
        self.move_to(PAN_START, TILT_START, force=True)

    def close(self) -> None:
        if self.connection is None:
            return
        try:
            self.connection.close()
        except OSError:
            pass
        self.connection = None


# ============================================================
# KALMAN STATE: x, y, vx, vy, diameter, diameter_rate
# ============================================================

class BallKalman:
    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(6, 3, 0, cv2.CV_32F)
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0]], dtype=np.float32
        )
        self.filter.measurementNoiseCov = np.diag([8.0, 8.0, 5.0]).astype(np.float32)
        self.filter.errorCovPost = np.eye(6, dtype=np.float32) * 20.0
        self.initialised = False
        self.last_time = time.monotonic()
        self.last_measurement_time: Optional[float] = None
        self.last_measurement: Optional[Observation] = None

    def reset(self) -> None:
        self.initialised = False
        self.last_time = time.monotonic()
        self.last_measurement_time = None
        self.last_measurement = None

    def _set_transition(self, dt: float) -> None:
        self.filter.transitionMatrix = np.array(
            [[1, 0, dt, 0, 0, 0],
             [0, 1, 0, dt, 0, 0],
             [0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, dt],
             [0, 0, 0, 0, 0, 1]], dtype=np.float32
        )
        self.filter.processNoiseCov = np.diag(
            [
                KALMAN_POSITION_PROCESS_VARIANCE,
                KALMAN_POSITION_PROCESS_VARIANCE,
                KALMAN_VELOCITY_PROCESS_VARIANCE,
                KALMAN_VELOCITY_PROCESS_VARIANCE,
                KALMAN_DIAMETER_PROCESS_VARIANCE,
                KALMAN_DIAMETER_RATE_PROCESS_VARIANCE,
            ]
        ).astype(np.float32) * max(dt, 1 / 120)

    def initialise(self, obs: Observation) -> None:
        state = np.array([[obs.x], [obs.y], [0], [0], [obs.diameter], [0]], dtype=np.float32)
        self.filter.statePost = state.copy()
        self.filter.statePre = state.copy()
        self.initialised = True
        self.last_time = time.monotonic()
        self.last_measurement_time = self.last_time
        self.last_measurement = obs

    def predict(self, now: float) -> Optional[np.ndarray]:
        if not self.initialised:
            return None
        dt = min(0.20, max(1 / 240, now - self.last_time))
        self.last_time = now
        self._set_transition(dt)
        prediction = self.filter.predict()
        self.filter.statePost = prediction.copy()
        self.filter.errorCovPost = self.filter.errorCovPre.copy()
        return prediction.reshape(-1)

    def correct(self, obs: Observation) -> np.ndarray:
        if not self.initialised:
            self.initialise(obs)
            return self.filter.statePost.reshape(-1)

        now = time.monotonic()
        measured_velocity: Optional[tuple[float, float, float]] = None
        if self.last_measurement is not None and self.last_measurement_time is not None:
            dt = now - self.last_measurement_time
            if 1 / 240 <= dt <= 0.35:
                measured_velocity = (
                    (obs.x - self.last_measurement.x) / dt,
                    (obs.y - self.last_measurement.y) / dt,
                    (obs.diameter - self.last_measurement.diameter) / dt,
                )

        # Trust high-confidence AI observations more than optical-flow fallback.
        variance = max(1.5, 12.0 * (1.0 - obs.confidence))
        if obs.source == "flow":
            variance *= 2.5
        self.filter.measurementNoiseCov = np.diag(
            [variance, variance, max(2.0, variance * 0.7)]
        ).astype(np.float32)

        measurement = np.array([[obs.x], [obs.y], [obs.diameter]], dtype=np.float32)
        state = self.filter.correct(measurement)

        if measured_velocity is not None:
            blend = KALMAN_VELOCITY_MEASUREMENT_BLEND
            if obs.source == "flow":
                blend *= 0.75
            state[2, 0] = (1.0 - blend) * state[2, 0] + blend * measured_velocity[0]
            state[3, 0] = (1.0 - blend) * state[3, 0] + blend * measured_velocity[1]
            state[5, 0] = (1.0 - blend) * state[5, 0] + blend * measured_velocity[2]
            self.filter.statePost = state

        self.last_measurement_time = now
        self.last_measurement = obs
        return self.filter.statePost.reshape(-1)

    def future(self, seconds: float) -> Optional[tuple[float, float, float]]:
        if not self.initialised:
            return None
        s = self.filter.statePost.reshape(-1)
        return (
            float(s[0] + s[2] * seconds),
            float(s[1] + s[3] * seconds),
            max(MIN_BALL_DIAMETER, float(s[4] + s[5] * seconds)),
        )


# ============================================================
# AI + MASK SHAPE VALIDATION
# ============================================================

class AIBallDetector:
    def __init__(
        self,
        model_path: Path,
        accepted_names: set[str],
        source_name: str,
        confidence: float,
        inference_size: int,
        accept_class_zero: bool,
    ) -> None:
        if not model_path.exists():
            raise RuntimeError(
                f"AI model not found: {model_path}"
            )

        os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tracker-matplotlib-cache")
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "ultralytics is not installed; using the classic colour detector only."
            ) from error

        print(f"Loading AI model: {model_path}")
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.accepted_names = accepted_names
        self.source_name = source_name
        self.confidence = confidence
        self.inference_size = inference_size
        self.accept_class_zero = accept_class_zero

    def _class_is_ball(self, class_id: int) -> bool:
        name = str(self.class_names.get(class_id, "")).lower().replace(" ", "_")
        return (self.accept_class_zero and class_id == 0) or name in self.accepted_names

    @staticmethod
    def _mask_metrics(mask: np.ndarray) -> tuple[float, float, float, float]:
        binary = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0, 0.0, 99.0, 0.0

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1e-6)

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        equivalent_diameter = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
        return circularity, solidity, aspect, equivalent_diameter

    def detect(
        self,
        frame: np.ndarray,
        predicted: Optional[tuple[float, float, float]],
    ) -> tuple[Optional[Observation], Optional[np.ndarray], list[Observation]]:
        result = self.model.predict(
            source=frame,
            imgsz=self.inference_size,
            conf=self.confidence,
            iou=DETECTION_IOU,
            device=DEVICE,
            verbose=False,
            retina_masks=True,
            max_det=20,
        )[0]

        h, w = frame.shape[:2]
        max_diameter = min(h, w) * MAX_BALL_FRAME_FRACTION
        observations: list[Observation] = []
        diagnostic_mask = np.zeros((h, w), dtype=np.uint8)

        boxes = result.boxes
        if boxes is None:
            return None, diagnostic_mask, observations

        masks_data = None
        if result.masks is not None and result.masks.data is not None:
            masks_data = result.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            if not self._class_is_ball(class_id):
                continue

            confidence = float(boxes.conf[i].item())
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
            box_w, box_h = x2 - x1, y2 - y1
            diameter = min(box_w, box_h)
            x, y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if diameter < MIN_BALL_DIAMETER or diameter > max_diameter:
                continue

            mask = None
            circularity, solidity, aspect = 0.0, 0.0, max(box_w, box_h) / max(1.0, min(box_w, box_h))

            if masks_data is not None and i < len(masks_data):
                mask = cv2.resize(
                    masks_data[i],
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = (mask > 0.5).astype(np.uint8) * 255
                circularity, solidity, aspect, mask_diameter = self._mask_metrics(mask)

                # Mask-derived centre and diameter are preferable.
                moments = cv2.moments(mask, binaryImage=True)
                if moments["m00"] > 0:
                    x = moments["m10"] / moments["m00"]
                    y = moments["m01"] / moments["m00"]
                if mask_diameter >= MIN_BALL_DIAMETER:
                    diameter = mask_diameter

                if (
                    circularity < MIN_MASK_CIRCULARITY
                    or solidity < MIN_MASK_SOLIDITY
                    or aspect > MAX_MASK_ASPECT_RATIO
                ):
                    continue
                diagnostic_mask = cv2.max(diagnostic_mask, mask)

            obs = Observation(
                x=x,
                y=y,
                diameter=diameter,
                confidence=confidence,
                source=self.source_name,
                mask=mask,
                circularity=circularity,
                solidity=solidity,
                aspect_ratio=aspect,
            )
            observations.append(obs)

        if not observations:
            return None, diagnostic_mask, observations

        def score(obs: Observation) -> float:
            value = obs.confidence
            if obs.mask is not None:
                value += 0.15 * min(1.0, obs.circularity)
                value += 0.10 * min(1.0, obs.solidity)
            if predicted is not None:
                px, py, pd = predicted
                diagonal = math.hypot(w, h)
                distance = math.hypot(obs.x - px, obs.y - py)
                gate = max(pd * 7.0, diagonal * 0.05)
                if distance > max(gate, diagonal * MAX_ASSOCIATION_DISTANCE_FRACTION):
                    return -999.0
                ratio = max(obs.diameter, pd) / max(MIN_BALL_DIAMETER, min(obs.diameter, pd))
                if ratio > MAX_DIAMETER_RATIO:
                    return -999.0
                value += 0.50 * max(0.0, 1.0 - distance / max(gate, 1.0))
                value += 0.20 * max(0.0, 1.0 - abs(math.log(max(ratio, 1e-6))) / math.log(MAX_DIAMETER_RATIO))
            return value

        selected = max(observations, key=score)
        return (selected if score(selected) > -100 else None), diagnostic_mask, observations


class AsyncAIDetector:
    def __init__(self, detectors: list[AIBallDetector]) -> None:
        self.detectors = detectors
        self._condition = threading.Condition()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._request_sequence = -1
        self._processed_sequence = -1
        self._frame: Optional[np.ndarray] = None
        self._predicted: Optional[tuple[float, float, float]] = None
        self._result: Optional[AIDetectionResult] = None

    def start(self) -> None:
        if not self.detectors:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(
        self,
        sequence: int,
        frame: np.ndarray,
        predicted: Optional[tuple[float, float, float]],
    ) -> None:
        if not self._running:
            return
        with self._condition:
            if sequence <= self._request_sequence:
                return
            self._request_sequence = sequence
            self._frame = frame.copy()
            self._predicted = predicted
            self._condition.notify_all()

    def latest(self) -> Optional[AIDetectionResult]:
        with self._condition:
            return self._result

    def _loop(self) -> None:
        while True:
            with self._condition:
                while (
                    self._running
                    and self._request_sequence <= self._processed_sequence
                ):
                    self._condition.wait()

                if not self._running:
                    break

                sequence = self._request_sequence
                frame = None if self._frame is None else self._frame.copy()
                predicted = self._predicted

            if frame is None:
                continue

            h, w = frame.shape[:2]
            ai_mask = np.zeros((h, w), dtype=np.uint8)
            ai_candidates: list[Observation] = []

            try:
                for detector in self.detectors:
                    _, detector_mask, detector_candidates = detector.detect(
                        frame,
                        predicted,
                    )
                    if detector_mask is not None:
                        ai_mask = cv2.max(ai_mask, detector_mask)
                    ai_candidates.extend(detector_candidates)

                ai_obs = best_observation(ai_candidates, predicted, w, h)
                result = AIDetectionResult(
                    sequence=sequence,
                    observation=ai_obs,
                    mask=ai_mask,
                    candidates=ai_candidates,
                )
            except Exception as error:
                print(f"AI worker failed: {error}")
                result = AIDetectionResult(
                    sequence=sequence,
                    observation=None,
                    mask=ai_mask,
                    candidates=[],
                )

            with self._condition:
                self._processed_sequence = sequence
                self._result = result

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ============================================================
# CLASSIC HS-HISTOGRAM + CONTOUR DETECTOR
# ============================================================

class ColourBallDetector:
    def __init__(self, model_path: Path) -> None:
        self.histogram: Optional[np.ndarray] = None
        if model_path.exists():
            try:
                data = np.load(model_path)
                histogram = data["histogram"].astype(np.float32)
                cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
                self.histogram = histogram
                print(f"Loaded colour model: {model_path}")
            except (OSError, KeyError, ValueError) as error:
                print(f"Colour model invalid; using fixed HSV thresholds: {error}")
        else:
            print("Colour model not found; using fixed HSV thresholds")

    def learn(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            raise ValueError("sample box is empty")

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]

        sample_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            sample_mask,
            (width // 2, height // 2),
            max(3, int(min(width, height) * 0.42)),
            255,
            -1,
        )
        broad_mask = cv2.inRange(hsv, COLOUR_SAMPLE_HSV_LOWER, COLOUR_SAMPLE_HSV_UPPER)
        sample_mask = cv2.bitwise_and(sample_mask, broad_mask)

        if cv2.countNonZero(sample_mask) < 40:
            raise ValueError(
                "not enough tennis-ball-coloured pixels in the centre sample box"
            )

        histogram = cv2.calcHist(
            [hsv],
            [0, 1],
            sample_mask,
            [COLOUR_HISTOGRAM_H_BINS, COLOUR_HISTOGRAM_S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        self.histogram = histogram.astype(np.float32)
        np.savez_compressed(COLOUR_MODEL_PATH, histogram=self.histogram)
        print(f"Saved colour model: {COLOUR_MODEL_PATH}")

    @staticmethod
    def _shape_metrics(contour: np.ndarray) -> tuple[float, float, float, float]:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / max(hull_area, 1e-6)

        x, y, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1.0, min(w, h))
        equivalent_diameter = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
        return circularity, solidity, aspect, equivalent_diameter

    @staticmethod
    def _robust_diameter(contour: np.ndarray) -> float:
        area = cv2.contourArea(contour)
        equivalent = 2.0 * math.sqrt(max(area, 1.0) / math.pi)
        _, _, w, h = cv2.boundingRect(contour)
        (_, _), radius = cv2.minEnclosingCircle(contour)

        measurements = [
            equivalent,
            float(min(w, h)),
        ]

        enclosing = 2.0 * float(radius)
        if enclosing <= equivalent * 1.45:
            measurements.append(enclosing)

        if len(contour) >= 5:
            (_, _), axes, _ = cv2.fitEllipse(contour)
            ellipse_diameter = (float(axes[0]) + float(axes[1])) / 2.0
            if equivalent * 0.65 <= ellipse_diameter <= equivalent * 1.45:
                measurements.append(ellipse_diameter)

        return float(np.median(np.array(measurements, dtype=np.float32)))

    def _candidate_mask(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        fixed_mask = cv2.inRange(hsv, COLOUR_HSV_LOWER, COLOUR_HSV_UPPER)

        if self.histogram is not None:
            probability = cv2.calcBackProject(
                [hsv],
                [0, 1],
                self.histogram,
                [0, 180, 0, 256],
                1,
            )
            _, mask = cv2.threshold(
                probability,
                COLOUR_BACKPROJECT_THRESHOLD,
                255,
                cv2.THRESH_BINARY,
            )
            mask = cv2.bitwise_or(mask.astype(np.uint8), fixed_mask)
            probability = cv2.max(probability, fixed_mask // 2)
        else:
            mask = fixed_mask
            probability = mask.copy()

        mask = mask.astype(np.uint8)
        mask[saturation < COLOUR_MIN_SATURATION] = 0
        mask[value < COLOUR_MIN_VALUE] = 0

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (COLOUR_KERNEL_SIZE, COLOUR_KERNEL_SIZE),
        )
        mask = cv2.medianBlur(mask, COLOUR_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask, probability

    @staticmethod
    def _ring_mask(shape: tuple[int, int], x: float, y: float, radius: float) -> np.ndarray:
        ring = np.zeros(shape, dtype=np.uint8)
        centre = (round(x), round(y))
        cv2.circle(ring, centre, max(1, round(radius)), 255, 2)
        return cv2.dilate(
            ring,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    def _hough_observations(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        probability: np.ndarray,
        total_mask_pixels: int,
    ) -> list[Observation]:
        h, w = frame.shape[:2]
        max_radius = max(
            HOUGH_MIN_RADIUS,
            int(min(h, w) * HOUGH_MAX_RADIUS_FRAME_FRACTION),
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DISTANCE,
            param1=HOUGH_CANNY_THRESHOLD,
            param2=HOUGH_ACCUMULATOR_THRESHOLD,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=max_radius,
        )
        if circles is None:
            return []

        edges = cv2.Canny(blurred, 55, 140)
        observations: list[Observation] = []

        for x, y, radius in np.round(circles[0]).astype(np.float32):
            diameter = 2.0 * float(radius)
            if diameter < MIN_BALL_DIAMETER:
                continue

            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(
                circle_mask,
                (round(float(x)), round(float(y))),
                max(1, round(float(radius) * 0.90)),
                255,
                -1,
            )
            circle_area = max(1, cv2.countNonZero(circle_mask))
            colour_fraction = (
                cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
                / circle_area
            )
            inner_mask = cv2.bitwise_and(mask, circle_mask)
            inner_contours, _ = cv2.findContours(
                inner_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not inner_contours:
                continue

            colour_contour = max(inner_contours, key=cv2.contourArea)
            colour_diameter = self._robust_diameter(colour_contour)
            if not (diameter * 0.55 <= colour_diameter <= diameter * 1.25):
                continue

            ring = self._ring_mask((h, w), float(x), float(y), float(radius))
            ring_area = max(1, cv2.countNonZero(ring))
            edge_fraction = (
                cv2.countNonZero(cv2.bitwise_and(edges, ring))
                / ring_area
            )

            if (
                colour_fraction < HOUGH_MIN_COLOUR_FRACTION
                or edge_fraction < HOUGH_MIN_EDGE_FRACTION
            ):
                continue

            support_pixels = cv2.countNonZero(cv2.bitwise_and(mask, circle_mask))
            colour_share = support_pixels / max(1, total_mask_pixels)
            colour_score = cv2.mean(probability, mask=circle_mask)[0] / 255.0
            confidence = min(
                0.92,
                0.20
                + 0.34 * min(1.0, colour_fraction / 0.45)
                + 0.22 * min(1.0, edge_fraction / 0.12)
                + 0.16 * colour_score,
            )
            observations.append(Observation(
                x=float(x),
                y=float(y),
                diameter=diameter,
                confidence=confidence,
                source="colour",
                mask=circle_mask,
                circularity=1.0,
                solidity=max(0.0, min(1.0, colour_fraction / 0.45)),
                aspect_ratio=1.0,
                colour_share=colour_share,
            ))

        return observations

    def detect(
        self,
        frame: np.ndarray,
        predicted: Optional[tuple[float, float, float]],
        use_hough: bool = True,
    ) -> tuple[Optional[Observation], np.ndarray, list[Observation]]:
        h, w = frame.shape[:2]
        max_diameter = min(h, w) * MAX_BALL_FRAME_FRACTION
        mask, probability = self._candidate_mask(frame)
        total_mask_pixels = cv2.countNonZero(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        observations: list[Observation] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < COLOUR_MIN_AREA:
                continue

            circularity, solidity, aspect, equivalent_diameter = self._shape_metrics(contour)
            if (
                circularity < COLOUR_MIN_CIRCULARITY
                or solidity < COLOUR_MIN_SOLIDITY
                or aspect > COLOUR_MAX_ASPECT_RATIO
            ):
                continue

            diameter = self._robust_diameter(contour)
            if diameter < MIN_BALL_DIAMETER or diameter > max_diameter:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue

            x = moments["m10"] / moments["m00"]
            y = moments["m01"] / moments["m00"]

            contour_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            support_pixels = cv2.countNonZero(cv2.bitwise_and(mask, contour_mask))
            colour_share = support_pixels / max(1, total_mask_pixels)
            colour_score = cv2.mean(probability, mask=contour_mask)[0] / 255.0
            size_score = min(1.0, equivalent_diameter / 24.0)
            confidence = min(
                0.95,
                0.18
                + 0.42 * colour_score
                + 0.22 * min(1.0, circularity)
                + 0.12 * min(1.0, solidity)
                + 0.06 * size_score,
            )

            observations.append(Observation(
                x=float(x),
                y=float(y),
                diameter=diameter,
                confidence=confidence,
                source="colour",
                mask=contour_mask,
                circularity=circularity,
                solidity=solidity,
                aspect_ratio=aspect,
                colour_share=colour_share,
            ))

        if use_hough:
            observations.extend(
                self._hough_observations(
                    frame,
                    mask,
                    probability,
                    total_mask_pixels,
                )
            )

        minimum_share = (
            COLOUR_MIN_MASK_SHARE_TRACKED
            if predicted is not None
            else COLOUR_MIN_MASK_SHARE_UNPREDICTED
        )
        observations = [
            obs for obs in observations
            if obs.colour_share >= minimum_share
        ]

        if not observations:
            return None, mask, observations

        def score(obs: Observation) -> float:
            value = obs.confidence
            value += 0.20 * min(1.0, obs.circularity)
            value += 0.12 * min(1.0, obs.solidity)
            value -= 0.08 * max(0.0, obs.aspect_ratio - 1.0)

            if predicted is not None:
                px, py, pd = predicted
                diagonal = math.hypot(w, h)
                distance = math.hypot(obs.x - px, obs.y - py)
                gate = max(pd * 8.0, diagonal * 0.06)
                if distance > max(gate, diagonal * MAX_ASSOCIATION_DISTANCE_FRACTION):
                    return -999.0
                ratio = max(obs.diameter, pd) / max(MIN_BALL_DIAMETER, min(obs.diameter, pd))
                if ratio > MAX_DIAMETER_RATIO:
                    return -999.0
                value += 0.45 * max(0.0, 1.0 - distance / max(gate, 1.0))
                value += 0.18 * max(0.0, 1.0 - abs(math.log(max(ratio, 1e-6))) / math.log(MAX_DIAMETER_RATIO))

            return value

        selected = max(observations, key=score)
        return (selected if score(selected) > -100 else None), mask, observations


# ============================================================
# PYRAMIDAL LUCAS-KANADE OPTICAL-FLOW FALLBACK
# ============================================================

class OpticalFlowFallback:
    def __init__(self) -> None:
        self.previous_gray: Optional[np.ndarray] = None
        self.points: Optional[np.ndarray] = None
        self.last_obs: Optional[Observation] = None

    def reset(self) -> None:
        self.previous_gray = None
        self.points = None
        self.last_obs = None

    def seed(self, frame: np.ndarray, obs: Observation) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        radius = max(5, int(obs.diameter * 0.65))
        mask = np.zeros_like(gray)
        cv2.circle(mask, (round(obs.x), round(obs.y)), radius, 255, -1)

        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=FLOW_MAX_CORNERS,
            qualityLevel=FLOW_QUALITY,
            minDistance=FLOW_MIN_DISTANCE,
            mask=mask,
            blockSize=5,
        )

        self.previous_gray = gray
        self.points = points
        self.last_obs = obs

    def update(self, frame: np.ndarray) -> tuple[Optional[Observation], np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = np.zeros_like(gray)

        if self.previous_gray is None or self.points is None or self.last_obs is None:
            self.previous_gray = gray
            return None, display

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            self.previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )

        if next_points is None or back_points is None:
            self.previous_gray = gray
            self.points = None
            return None, display

        fb = np.linalg.norm(self.points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
        valid = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & (fb < FLOW_FB_ERROR)
        )

        old_good = self.points.reshape(-1, 2)[valid]
        new_good = next_points.reshape(-1, 2)[valid]

        if len(new_good) < FLOW_MIN_GOOD_POINTS:
            self.previous_gray = gray
            self.points = None
            return None, display

        displacement = np.median(new_good - old_good, axis=0)
        h, w = gray.shape
        if np.linalg.norm(displacement) > math.hypot(w, h) * FLOW_MAX_JUMP_FRACTION:
            self.previous_gray = gray
            self.points = None
            return None, display

        new_x = self.last_obs.x + float(displacement[0])
        new_y = self.last_obs.y + float(displacement[1])

        for p in new_good:
            cv2.circle(display, tuple(np.round(p).astype(int)), 2, 255, -1)

        obs = Observation(
            x=new_x,
            y=new_y,
            diameter=self.last_obs.diameter,
            confidence=0.45,
            source="flow",
        )

        self.previous_gray = gray
        self.points = new_good.reshape(-1, 1, 2).astype(np.float32)
        self.last_obs = obs
        return obs, display


# ============================================================
# SIMPLE FRAME-DIFFERENCE MOTION EVIDENCE
# ============================================================

class MotionEvidence:
    def __init__(self) -> None:
        self.previous_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.previous_gray = None

    def update(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if MOTION_BLUR_SIZE > 1:
            gray = cv2.GaussianBlur(
                gray,
                (MOTION_BLUR_SIZE, MOTION_BLUR_SIZE),
                0,
            )

        if self.previous_gray is None:
            self.previous_gray = gray
            return np.zeros_like(gray)

        difference = cv2.absdiff(gray, self.previous_gray)
        self.previous_gray = gray

        _, mask = cv2.threshold(
            difference,
            MOTION_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask


# ============================================================
# HELPERS
# ============================================================

def load_focal_length() -> Optional[float]:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
        value = float(data["focal_length_pixels"])
        print(f"Loaded focal calibration: {value:.2f}px")
        return value
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        print("Calibration file invalid; recalibrate with K")
        return None


def save_focal_length(value: float) -> None:
    CALIBRATION_FILE.write_text(json.dumps(
        {
            "focal_length_pixels": value,
            "ball_diameter_metres": BALL_DIAMETER_METRES,
            "calibration_distance_metres": CALIBRATION_DISTANCE_METRES,
        },
        indent=2,
    ))
    print(f"Saved focal calibration: {value:.2f}px")


def sample_box_for_frame(frame: np.ndarray) -> tuple[int, int, int, int]:
    height, width = frame.shape[:2]
    side = round(min(width, height) * SAMPLE_BOX_FRACTION)
    centre_x, centre_y = width // 2, height // 2
    return (
        max(0, centre_x - side // 2),
        max(0, centre_y - side // 2),
        min(width, centre_x + side // 2),
        min(height, centre_y + side // 2),
    )


def add_motion_scores(observations: list[Observation], motion_mask: np.ndarray) -> None:
    for obs in observations:
        if obs.mask is None:
            radius = max(2, round(obs.diameter * 0.55))
            mask = np.zeros_like(motion_mask)
            cv2.circle(mask, (round(obs.x), round(obs.y)), radius, 255, -1)
        else:
            mask = obs.mask

        area = max(1, cv2.countNonZero(mask))
        moving = cv2.countNonZero(cv2.bitwise_and(motion_mask, mask))
        obs.motion_score = moving / area


class DistanceEstimator:
    def __init__(self, focal_length_pixels: Optional[float]) -> None:
        self.focal_length_pixels = focal_length_pixels
        self.raw_history: deque[float] = deque(maxlen=DISTANCE_MEDIAN_WINDOW)
        self.filtered_distance: Optional[float] = None

    def reset(self) -> None:
        self.raw_history.clear()
        self.filtered_distance = None

    def set_focal_length(self, focal_length_pixels: float) -> None:
        self.focal_length_pixels = focal_length_pixels
        self.reset()

    def update(self, obs: Observation) -> Optional[DistanceEstimate]:
        if self.focal_length_pixels is None or obs.source == "flow":
            return None
        if obs.diameter < MIN_BALL_DIAMETER:
            return None

        raw_distance = (
            self.focal_length_pixels
            * BALL_DIAMETER_METRES
            / max(obs.diameter, 1.0)
        )

        if len(self.raw_history) >= 3:
            median_distance = float(np.median(np.array(self.raw_history, dtype=np.float32)))
            max_change = median_distance * DISTANCE_MAX_JUMP_FRACTION
            raw_distance = WiFiServoController.clamp(
                raw_distance,
                median_distance - max_change,
                median_distance + max_change,
            )

        self.raw_history.append(raw_distance)
        median_distance = float(np.median(np.array(self.raw_history, dtype=np.float32)))

        if self.filtered_distance is None:
            self.filtered_distance = median_distance
        else:
            self.filtered_distance = (
                DISTANCE_ALPHA * median_distance
                + (1.0 - DISTANCE_ALPHA) * self.filtered_distance
            )

        return DistanceEstimate(
            raw_metres=raw_distance,
            filtered_metres=self.filtered_distance,
            diameter_pixels=obs.diameter,
            source=obs.source,
        )


class AcquisitionGate:
    def __init__(self) -> None:
        self.hits = 0
        self.previous: Optional[Observation] = None

    def reset(self) -> None:
        self.hits = 0
        self.previous = None

    def update(self, obs: Observation) -> int:
        if self.previous is None:
            self.hits = 1
            self.previous = obs
            return self.hits

        centre_distance = math.hypot(obs.x - self.previous.x, obs.y - self.previous.y)
        diameter_gate = max(obs.diameter, self.previous.diameter) * ACQUIRE_MAX_JUMP_DIAMETERS
        ratio = max(obs.diameter, self.previous.diameter) / max(
            MIN_BALL_DIAMETER,
            min(obs.diameter, self.previous.diameter),
        )

        if centre_distance <= diameter_gate and ratio <= ACQUIRE_MAX_DIAMETER_RATIO:
            self.hits += 1
        else:
            self.hits = 1

        self.previous = obs
        return self.hits


def observations_are_consistent(
    first: Observation,
    second: Observation,
    width: int,
    height: int,
) -> bool:
    centre_distance = math.hypot(first.x - second.x, first.y - second.y)
    diameter_gate = max(first.diameter, second.diameter) * 2.2
    frame_gate = math.hypot(width, height) * 0.045
    ratio = max(first.diameter, second.diameter) / max(
        MIN_BALL_DIAMETER,
        min(first.diameter, second.diameter),
    )
    return centre_distance <= max(diameter_gate, frame_gate) and ratio <= 1.9


def observation_matches_prediction(
    obs: Observation,
    predicted: Optional[tuple[float, float, float]],
    width: int,
    height: int,
) -> bool:
    if predicted is None:
        return True

    px, py, pd = predicted
    distance = math.hypot(obs.x - px, obs.y - py)
    frame_gate = math.hypot(width, height) * 0.08
    diameter_gate = max(pd, obs.diameter) * 5.0
    return distance <= max(frame_gate, diameter_gate)


def update_servo_tracking(
    controller: WiFiServoController,
    centering: BallCenteringController,
    aim: Optional[tuple[float, float, float]],
    frame: np.ndarray,
    now: float,
    last_command_time: float,
    tracking_enabled: bool,
) -> float:
    if not tracking_enabled or aim is None:
        centering.reset()
        return last_command_time

    if now - last_command_time < COMMAND_INTERVAL_SECONDS:
        return last_command_time

    pan_step, tilt_step = centering.steps(
        aim[0],
        aim[1],
        frame.shape[1],
        frame.shape[0],
        now,
    )
    if pan_step or tilt_step:
        controller.move_relative(pan_step, tilt_step)
    return now


def fuse_observations(first: Observation, second: Observation) -> Observation:
    first_weight = max(0.05, first.confidence)
    second_weight = max(0.05, second.confidence)
    total_weight = first_weight + second_weight

    mask = None
    if first.mask is not None and second.mask is not None:
        mask = cv2.max(first.mask, second.mask)
    elif first.mask is not None:
        mask = first.mask
    elif second.mask is not None:
        mask = second.mask

    return Observation(
        x=(first.x * first_weight + second.x * second_weight) / total_weight,
        y=(first.y * first_weight + second.y * second_weight) / total_weight,
        diameter=(
            first.diameter * first_weight
            + second.diameter * second_weight
        ) / total_weight,
        confidence=min(0.98, max(first.confidence, second.confidence) + 0.08),
        source="fusion",
        mask=mask,
        circularity=max(first.circularity, second.circularity),
        solidity=max(first.solidity, second.solidity),
        aspect_ratio=min(first.aspect_ratio, second.aspect_ratio),
        colour_share=max(first.colour_share, second.colour_share),
        motion_score=max(first.motion_score, second.motion_score),
    )


def measurement_score(
    obs: Observation,
    predicted: Optional[tuple[float, float, float]],
    width: int,
    height: int,
) -> float:
    value = obs.confidence
    if obs.source in {"ai", "coco_ai"}:
        value += 0.08
    elif obs.source == "fusion":
        value += 0.16

    value += 0.12 * min(1.0, obs.circularity)
    value += 0.08 * min(1.0, obs.solidity)
    value += 0.10 * min(1.0, obs.motion_score / 0.20)
    value -= 0.05 * max(0.0, obs.aspect_ratio - 1.0)

    if predicted is None:
        return value

    px, py, pd = predicted
    diagonal = math.hypot(width, height)
    distance = math.hypot(obs.x - px, obs.y - py)
    gate = max(pd * 8.0, diagonal * 0.06)
    if distance > max(gate, diagonal * MAX_ASSOCIATION_DISTANCE_FRACTION):
        return -999.0

    ratio = max(obs.diameter, pd) / max(
        MIN_BALL_DIAMETER,
        min(obs.diameter, pd),
    )
    if ratio > MAX_DIAMETER_RATIO:
        return -999.0

    value += 0.42 * max(0.0, 1.0 - distance / max(gate, 1.0))
    value += 0.16 * max(
        0.0,
        1.0 - abs(math.log(max(ratio, 1e-6))) / math.log(MAX_DIAMETER_RATIO),
    )
    return value


def best_observation(
    observations: list[Observation],
    predicted: Optional[tuple[float, float, float]],
    width: int,
    height: int,
) -> Optional[Observation]:
    if not observations:
        return None

    selected = max(
        observations,
        key=lambda obs: measurement_score(obs, predicted, width, height),
    )
    return (
        selected
        if measurement_score(selected, predicted, width, height) > -100
        else None
    )


def colour_only_can_acquire(obs: Observation) -> bool:
    return (
        obs.confidence >= COLOUR_ONLY_ACQUIRE_MIN_CONFIDENCE
        and obs.colour_share >= COLOUR_ONLY_ACQUIRE_MIN_SHARE
        and obs.motion_score >= COLOUR_ONLY_ACQUIRE_MIN_MOTION
    )


def coco_only_can_acquire(obs: Observation) -> bool:
    return (
        obs.confidence >= COCO_ONLY_ACQUIRE_MIN_CONFIDENCE
        and obs.diameter >= COCO_ONLY_ACQUIRE_MIN_DIAMETER
        and obs.motion_score >= COCO_ONLY_ACQUIRE_MIN_MOTION
    )


def can_fast_lock(obs: Observation) -> bool:
    if obs.source == "fusion":
        return obs.confidence >= FAST_LOCK_FUSION_CONFIDENCE

    if obs.source in {"ai", "coco_ai"}:
        return (
            obs.confidence >= FAST_LOCK_AI_CONFIDENCE
            and obs.diameter >= COCO_ONLY_ACQUIRE_MIN_DIAMETER
        )

    if obs.source == "colour":
        return (
            obs.confidence >= FAST_LOCK_COLOUR_CONFIDENCE
            and obs.colour_share >= FAST_LOCK_COLOUR_SHARE
        )

    return False


def choose_measurement(
    ai_obs: Optional[Observation],
    colour_obs: Optional[Observation],
    predicted: Optional[tuple[float, float, float]],
    width: int,
    height: int,
) -> Optional[Observation]:
    if ai_obs is not None and colour_obs is not None:
        if observations_are_consistent(ai_obs, colour_obs, width, height):
            return fuse_observations(ai_obs, colour_obs)

        if ai_obs.source == "coco_ai" and predicted is None:
            if colour_obs is not None and colour_only_can_acquire(colour_obs):
                return colour_obs
            if coco_only_can_acquire(ai_obs):
                return ai_obs
            return None

        observations = [ai_obs, colour_obs]
        selected = max(
            observations,
            key=lambda obs: measurement_score(obs, predicted, width, height),
        )
        return (
            selected
            if measurement_score(selected, predicted, width, height) > -100
            else None
        )

    if ai_obs is not None and ai_obs.source == "coco_ai" and predicted is None:
        return ai_obs if coco_only_can_acquire(ai_obs) else None

    if colour_obs is not None and predicted is None:
        if not colour_only_can_acquire(colour_obs):
            return None

    return ai_obs if ai_obs is not None else colour_obs


class BallCenteringController:
    def __init__(self) -> None:
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.previous_filtered_error_x = 0.0
        self.previous_filtered_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.previous_pan_step = 0.0
        self.previous_tilt_step = 0.0
        self.previous_time: Optional[float] = None
        self.initialised = False

    def reset(self) -> None:
        self.filtered_error_x = 0.0
        self.filtered_error_y = 0.0
        self.previous_filtered_error_x = 0.0
        self.previous_filtered_error_y = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.previous_pan_step = 0.0
        self.previous_tilt_step = 0.0
        self.previous_time = None
        self.initialised = False

    @staticmethod
    def _axis_step(
        error: float,
        previous_error: float,
        integral: float,
        dt: float,
        dead_zone: float,
        proportional_gain: float,
        derivative_gain: float,
        integral_gain: float,
        minimum: float,
        maximum: float,
        direction: float,
    ) -> tuple[float, float]:
        if abs(error) <= dead_zone:
            return 0.0, integral * 0.80

        effective_error = math.copysign(abs(error) - dead_zone, error)
        integral += effective_error * dt
        integral = WiFiServoController.clamp(
            integral,
            -PID_INTEGRAL_LIMIT,
            PID_INTEGRAL_LIMIT,
        )

        derivative = (error - previous_error) / max(dt, 1e-3)
        derivative = WiFiServoController.clamp(
            derivative,
            -PID_DERIVATIVE_LIMIT,
            PID_DERIVATIVE_LIMIT,
        )

        control = (
            proportional_gain * effective_error
            + integral_gain * integral
            - derivative_gain * derivative
        )
        magnitude = min(maximum, abs(control))
        if magnitude < minimum:
            magnitude = minimum

        return direction * math.copysign(magnitude, control), integral

    @staticmethod
    def _limit_step_change(
        step: float,
        previous_step: float,
        maximum_change: float,
    ) -> float:
        return WiFiServoController.clamp(
            step,
            previous_step - maximum_change,
            previous_step + maximum_change,
        )

    def steps(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        now: float,
    ) -> tuple[float, float]:
        error_x = (x - width / 2) / (width / 2)
        error_y = (y - height / 2) / (height / 2)
        dt = (
            COMMAND_INTERVAL_SECONDS
            if self.previous_time is None
            else max(1e-3, min(0.12, now - self.previous_time))
        )

        if not self.initialised:
            self.filtered_error_x = error_x
            self.filtered_error_y = error_y
            self.previous_filtered_error_x = error_x
            self.previous_filtered_error_y = error_y
            self.initialised = True
        else:
            self.previous_filtered_error_x = self.filtered_error_x
            self.previous_filtered_error_y = self.filtered_error_y
            self.filtered_error_x = (
                ERROR_FILTER_ALPHA * error_x
                + (1.0 - ERROR_FILTER_ALPHA) * self.filtered_error_x
            )
            self.filtered_error_y = (
                ERROR_FILTER_ALPHA * error_y
                + (1.0 - ERROR_FILTER_ALPHA) * self.filtered_error_y
            )

        if abs(error_x) <= DEAD_ZONE_X:
            self.filtered_error_x = 0.0
            self.integral_x = 0.0
            self.previous_pan_step = 0.0
        if abs(error_y) <= DEAD_ZONE_Y:
            self.filtered_error_y = 0.0
            self.integral_y = 0.0
            self.previous_tilt_step = 0.0

        pan_step, self.integral_x = self._axis_step(
            self.filtered_error_x,
            self.previous_filtered_error_x,
            self.integral_x,
            dt,
            DEAD_ZONE_X,
            PAN_GAIN,
            PAN_D_GAIN,
            PAN_I_GAIN,
            MINIMUM_PAN_STEP,
            MAXIMUM_PAN_STEP,
            PAN_DIRECTION,
        )
        tilt_step, self.integral_y = self._axis_step(
            self.filtered_error_y,
            self.previous_filtered_error_y,
            self.integral_y,
            dt,
            DEAD_ZONE_Y,
            TILT_GAIN,
            TILT_D_GAIN,
            TILT_I_GAIN,
            MINIMUM_TILT_STEP,
            MAXIMUM_TILT_STEP,
            TILT_DIRECTION,
        )
        pan_step = self._limit_step_change(
            pan_step,
            self.previous_pan_step,
            MAXIMUM_PAN_STEP_CHANGE,
        )
        tilt_step = self._limit_step_change(
            tilt_step,
            self.previous_tilt_step,
            MAXIMUM_TILT_STEP_CHANGE,
        )

        self.previous_pan_step = pan_step
        self.previous_tilt_step = tilt_step
        self.previous_time = now
        return pan_step, tilt_step


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    camera = LatestFrameCamera(CAMERA_SOURCE)
    controller: Optional[WiFiServoController] = None
    ai_worker: Optional[AsyncAIDetector] = None

    try:
        ai_detectors: list[AIBallDetector] = []
        try:
            ai_detectors.append(AIBallDetector(
                MODEL_PATH,
                accepted_names={"tennis_ball", "ball", BALL_CLASS_NAME.lower()},
                source_name="ai",
                confidence=DETECTION_CONFIDENCE,
                inference_size=INFERENCE_SIZE,
                accept_class_zero=True,
            ))
        except RuntimeError as error:
            print(error)

        try:
            coco_model_path = (
                COCO_MODEL_PATH
                if COCO_MODEL_PATH.exists()
                else COCO_STRONG_MODEL_PATH
            )
            ai_detectors.append(AIBallDetector(
                coco_model_path,
                accepted_names={"sports_ball"},
                source_name="coco_ai",
                confidence=COCO_DETECTION_CONFIDENCE,
                inference_size=COCO_INFERENCE_SIZE,
                accept_class_zero=False,
            ))
        except RuntimeError as error:
            print(error)

        ai_worker = AsyncAIDetector(ai_detectors)
        colour_detector = ColourBallDetector(COLOUR_MODEL_PATH)
        controller = WiFiServoController(GIGA_HOST, GIGA_PORT)
        kalman = BallKalman()
        flow = OpticalFlowFallback()
        motion = MotionEvidence()
        centering = BallCenteringController()
        distance_estimator = DistanceEstimator(load_focal_length())

        camera.start()
        sequence = -1
        frame_index = 0
        acquisition = AcquisitionGate()
        ai_misses = 0
        total_misses = 0
        tracking_enabled = True
        locked = False
        last_command_time = 0.0
        last_frame_time: Optional[float] = None
        process_fps = 0.0
        last_distance: Optional[DistanceEstimate] = None
        last_ai_result_sequence = -1
        trail: deque[tuple[int, int]] = deque(maxlen=40)

        ai_worker.start()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if SHOW_DIAGNOSTIC_WINDOW:
            cv2.namedWindow(DIAGNOSTIC_WINDOW, cv2.WINDOW_NORMAL)

        while True:
            ok, frame, sequence = camera.read(sequence)
            if not ok or frame is None:
                continue

            frame_index += 1
            now = time.monotonic()
            if last_frame_time is not None:
                frame_dt = max(1.0 / 240.0, now - last_frame_time)
                instant_fps = 1.0 / frame_dt
                process_fps = (
                    instant_fps
                    if process_fps == 0.0
                    else 0.90 * process_fps + 0.10 * instant_fps
                )
            last_frame_time = now

            kalman.predict(now)
            predicted = kalman.future(0.0)
            motion_mask = motion.update(frame)
            flow_obs, flow_debug = flow.update(frame)

            ai_obs: Optional[Observation] = None
            ai_mask = (
                np.zeros(frame.shape[:2], dtype=np.uint8)
                if SHOW_DIAGNOSTIC_WINDOW
                else np.zeros((1, 1), dtype=np.uint8)
            )
            ai_candidates: list[Observation] = []

            ai_interval = (
                AI_LOCKED_EVERY_N_FRAMES
                if locked
                else AI_ACQUIRE_EVERY_N_FRAMES
            )
            if ai_worker is not None and frame_index % ai_interval == 0:
                ai_worker.submit(sequence, frame, predicted)

            if ai_worker is not None:
                ai_result = ai_worker.latest()
                if (
                    ai_result is not None
                    and ai_result.sequence > last_ai_result_sequence
                    and sequence - ai_result.sequence <= AI_RESULT_MAX_AGE_FRAMES
                ):
                    ai_obs = ai_result.observation
                    if SHOW_DIAGNOSTIC_WINDOW:
                        ai_mask = ai_result.mask
                    ai_candidates = ai_result.candidates
                    last_ai_result_sequence = ai_result.sequence

            selected: Optional[Observation] = None
            aim: Optional[tuple[float, float, float]] = None
            fast_control_done = False
            flow_is_plausible = (
                locked
                and flow_obs is not None
                and observation_matches_prediction(
                    flow_obs,
                    predicted,
                    frame.shape[1],
                    frame.shape[0],
                )
            )
            if flow_is_plausible and flow_obs is not None:
                selected = flow_obs
                state = kalman.correct(selected)
                trail.append((round(state[0]), round(state[1])))
                aim = kalman.future(PREDICTION_SECONDS)
                last_command_time = update_servo_tracking(
                    controller,
                    centering,
                    aim,
                    frame,
                    now,
                    last_command_time,
                    tracking_enabled,
                )
                fast_control_done = True

            colour_obs: Optional[Observation] = None
            colour_mask = (
                np.zeros(frame.shape[:2], dtype=np.uint8)
                if SHOW_DIAGNOSTIC_WINDOW
                else np.zeros((1, 1), dtype=np.uint8)
            )
            colour_candidates: list[Observation] = []
            colour_interval = (
                COLOUR_LOCKED_EVERY_N_FRAMES
                if locked
                else COLOUR_ACQUIRE_EVERY_N_FRAMES
            )
            should_run_colour = (
                not flow_is_plausible
                or frame_index % colour_interval == 0
            )
            if should_run_colour:
                colour_obs, colour_mask, colour_candidates = colour_detector.detect(
                    frame,
                    predicted,
                    use_hough=not flow_is_plausible,
                )
                add_motion_scores(colour_candidates, motion_mask)
                colour_obs = best_observation(
                    colour_candidates,
                    predicted,
                    frame.shape[1],
                    frame.shape[0],
                )

            measured_obs = choose_measurement(
                ai_obs,
                colour_obs,
                predicted,
                frame.shape[1],
                frame.shape[0],
            )

            if flow_is_plausible and flow_obs is not None:
                if measured_obs is not None:
                    flow.seed(frame, measured_obs)
                    if observations_are_consistent(
                        flow_obs,
                        measured_obs,
                        frame.shape[1],
                        frame.shape[0],
                    ):
                        kalman.correct(measured_obs)
                    ai_misses = 0
                    total_misses = 0
                else:
                    ai_misses += 1
                    total_misses += 1
            elif measured_obs is not None:
                selected = measured_obs
                flow.seed(frame, measured_obs)
                ai_misses = 0
                total_misses = 0
                acquisition_hits = (
                    ACQUIRE_HITS
                    if can_fast_lock(measured_obs)
                    else acquisition.update(measured_obs)
                )
            else:
                ai_misses += 1
                total_misses += 1
                acquisition.reset()
                acquisition_hits = 0

            if not locked and acquisition_hits >= ACQUIRE_HITS:
                locked = True
                print("Target locked")
                if selected is not None:
                    kalman.correct(selected)
                    aim = kalman.future(PREDICTION_SECONDS)
                    last_command_time = update_servo_tracking(
                        controller,
                        centering,
                        aim,
                        frame,
                        now,
                        last_command_time,
                        tracking_enabled,
                    )

            if ai_misses > MAX_AI_MISSES:
                # Force AI on every frame until reacquired.
                pass

            if total_misses > MAX_TOTAL_MISSES:
                if locked:
                    print("Target lost")
                locked = False
                acquisition.reset()
                kalman.reset()
                flow.reset()
                motion.reset()
                centering.reset()
                trail.clear()
                distance_estimator.reset()
                last_distance = None

            if selected is not None and not fast_control_done:
                state = kalman.correct(selected)
                if (
                    selected.source == "flow"
                    and measured_obs is not None
                    and observations_are_consistent(
                        selected,
                        measured_obs,
                        frame.shape[1],
                        frame.shape[0],
                    )
                ):
                    state = kalman.correct(measured_obs)
                if locked:
                    trail.append((round(state[0]), round(state[1])))

            if not fast_control_done:
                predict_only = (
                    locked
                    and selected is None
                    and total_misses > 0
                    and total_misses <= PREDICT_ONLY_MAX_MISSES
                )
                aim_seconds = (
                    OFFSCREEN_PREDICTION_SECONDS
                    if predict_only
                    else PREDICTION_SECONDS
                )
                aim = kalman.future(aim_seconds) if locked else None
                last_command_time = update_servo_tracking(
                    controller,
                    centering,
                    aim,
                    frame,
                    now,
                    last_command_time,
                    tracking_enabled,
                )

            # Distance uses detector measurements only, never flow-only diameter.
            distance_obs = measured_obs if measured_obs is not None else selected
            if distance_obs is not None and distance_obs.source != "flow":
                estimate = distance_estimator.update(distance_obs)
                if estimate is not None:
                    last_distance = estimate

            if DRAW_CANDIDATES:
                for obs in ai_candidates:
                    radius = max(3, round(obs.diameter / 2))
                    cv2.circle(frame, (round(obs.x), round(obs.y)), radius, (120, 120, 120), 1)
                for obs in colour_candidates:
                    radius = max(3, round(obs.diameter / 2))
                    cv2.circle(frame, (round(obs.x), round(obs.y)), radius, (80, 160, 255), 1)

            if selected is not None:
                colour = (0, 255, 0) if selected.source == "ai" else (0, 200, 255)
                cv2.circle(
                    frame,
                    (round(selected.x), round(selected.y)),
                    max(4, round(selected.diameter / 2)),
                    colour,
                    2,
                )
                cv2.putText(
                    frame,
                    f"{selected.source.upper()} {selected.confidence:.2f}",
                    (round(selected.x + selected.diameter / 2 + 5), round(selected.y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    colour,
                    2,
                )

            if aim is not None:
                cv2.drawMarker(
                    frame,
                    (round(aim[0]), round(aim[1])),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    20,
                    2,
                )

            for i in range(1, len(trail)):
                cv2.line(frame, trail[i - 1], trail[i], (0, 255, 255), 1)

            h, w = frame.shape[:2]
            sample_box = sample_box_for_frame(frame)
            cv2.rectangle(
                frame,
                (sample_box[0], sample_box[1]),
                (sample_box[2], sample_box[3]),
                (255, 0, 255),
                1,
            )
            dx, dy = round(w * DEAD_ZONE_X / 2), round(h * DEAD_ZONE_Y / 2)
            cv2.rectangle(frame, (w // 2 - dx, h // 2 - dy), (w // 2 + dx, h // 2 + dy), (255, 255, 0), 1)

            status = "LOCKED" if locked else "ACQUIRING"
            if locked and selected is None and total_misses > 0:
                status = "PREDICTING"
            if not tracking_enabled:
                status = "PAUSED"
            cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(
                frame,
                (
                    f"Pan {controller.pan:.0f} Tilt {controller.tilt:.0f}  "
                    f"Cam {camera.fps:.1f} FPS Proc {process_fps:.1f}"
                ),
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            if last_distance is not None:
                cv2.putText(
                    frame,
                    (
                        f"Distance {last_distance.filtered_metres:.2f} m  "
                        f"{last_distance.source} d={last_distance.diameter_pixels:.0f}px"
                    ),
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "B learn ball | Space track | K calibrate | R reset | A/D/W/S manual | C centre | Q quit",
                (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            cv2.imshow(WINDOW_NAME, frame)
            if SHOW_DIAGNOSTIC_WINDOW:
                diagnostics = np.hstack((ai_mask, colour_mask, motion_mask, flow_debug))
                cv2.imshow(DIAGNOSTIC_WINDOW, diagnostics)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                tracking_enabled = not tracking_enabled
            elif key in (ord("r"), ord("R")):
                locked = False
                acquisition.reset()
                ai_misses = total_misses = 0
                kalman.reset()
                flow.reset()
                motion.reset()
                centering.reset()
                distance_estimator.reset()
                last_distance = None
                trail.clear()
            elif key in (ord("b"), ord("B")):
                try:
                    colour_detector.learn(frame, sample_box_for_frame(frame))
                    locked = False
                    acquisition.reset()
                    kalman.reset()
                    flow.reset()
                    motion.reset()
                    centering.reset()
                    distance_estimator.reset()
                    last_distance = None
                    trail.clear()
                except ValueError as error:
                    print(f"Colour learning failed: {error}")
            elif key in (ord("k"), ord("K")):
                calibration_obs = selected if selected is not None and selected.source != "flow" else measured_obs
                if calibration_obs is None or calibration_obs.source == "flow":
                    print("Calibration failed: no detector ball measurement")
                else:
                    focal_length = (
                        calibration_obs.diameter
                        * CALIBRATION_DISTANCE_METRES
                        / BALL_DIAMETER_METRES
                    )
                    distance_estimator.set_focal_length(focal_length)
                    last_distance = None
                    save_focal_length(focal_length)
            elif key in (ord("c"), ord("C")):
                tracking_enabled = False
                centering.reset()
                controller.centre()
            elif key in (ord("a"), ord("A")):
                tracking_enabled = False
                centering.reset()
                controller.move_relative(-MANUAL_PAN_STEP, 0)
            elif key in (ord("d"), ord("D")):
                tracking_enabled = False
                centering.reset()
                controller.move_relative(MANUAL_PAN_STEP, 0)
            elif key in (ord("w"), ord("W")):
                tracking_enabled = False
                centering.reset()
                controller.move_relative(0, MANUAL_TILT_STEP)
            elif key in (ord("s"), ord("S")):
                tracking_enabled = False
                centering.reset()
                controller.move_relative(0, -MANUAL_TILT_STEP)

    except KeyboardInterrupt:
        print("Stopped")
    except Exception as error:
        print(f"ERROR: {error}")
    finally:
        if ai_worker is not None:
            ai_worker.stop()
        camera.stop()
        if controller is not None:
            controller.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

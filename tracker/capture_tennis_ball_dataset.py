from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np


CAMERA_SOURCE: Union[str, int] = "http://192.168.0.96:81/stream"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "captured_dataset"
POSITIVE_DIR = OUTPUT_ROOT / "ball_present"
NEGATIVE_DIR = OUTPUT_ROOT / "no_ball"

WINDOW_NAME = "Tennis-ball dataset capture"

AUTO_CAPTURE_INTERVAL_SECONDS = 0.25
MINIMUM_FRAME_DIFFERENCE = 7.0
DIFFERENCE_CHECK_SIZE = (160, 120)
JPEG_QUALITY = 95


def open_camera(source: Union[str, int]) -> cv2.VideoCapture:
    print(f"Opening camera: {source}")
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "Could not open the camera stream. "
            "Check CAMERA_SOURCE and confirm the stream works in a browser."
        )

    return capture


def comparison_image(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(
        frame,
        DIFFERENCE_CHECK_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def frame_difference(
    current: np.ndarray,
    previous: np.ndarray | None,
) -> float:
    if previous is None:
        return 999.0

    difference = cv2.absdiff(current, previous)
    return float(np.mean(difference))


def next_index(directory: Path) -> int:
    highest = -1

    for path in directory.glob("*.jpg"):
        try:
            index = int(path.stem.split("_")[-1])
            highest = max(highest, index)
        except ValueError:
            continue

    return highest + 1


def save_frame(
    frame: np.ndarray,
    directory: Path,
    prefix: str,
    index: int,
) -> Path:
    path = directory / f"{prefix}_{index:06d}.jpg"

    success = cv2.imwrite(
        str(path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )

    if not success:
        raise RuntimeError(f"Could not save image: {path}")

    return path


def main() -> None:
    POSITIVE_DIR.mkdir(parents=True, exist_ok=True)
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)

    positive_index = next_index(POSITIVE_DIR)
    negative_index = next_index(NEGATIVE_DIR)

    capture = open_camera(CAMERA_SOURCE)

    auto_mode: str | None = None
    last_auto_capture_time = 0.0

    last_saved_positive: np.ndarray | None = None
    last_saved_negative: np.ndarray | None = None

    positive_count = len(list(POSITIVE_DIR.glob("*.jpg")))
    negative_count = len(list(NEGATIVE_DIR.glob("*.jpg")))

    print()
    print("Controls:")
    print("  P  save frame with a tennis ball")
    print("  N  save frame with no tennis ball")
    print("  A  toggle automatic ball-present capture")
    print("  G  toggle automatic no-ball capture")
    print("  Space  stop automatic capture")
    print("  Q or Esc  quit")
    print()

    try:
        while True:
            ok, frame = capture.read()

            if not ok or frame is None:
                print("Camera stream lost; reconnecting")
                capture.release()
                time.sleep(0.5)
                capture = open_camera(CAMERA_SOURCE)
                continue

            now = time.monotonic()
            current_comparison = comparison_image(frame)
            auto_saved = False

            if (
                auto_mode is not None
                and now - last_auto_capture_time >= AUTO_CAPTURE_INTERVAL_SECONDS
            ):
                if auto_mode == "positive":
                    difference = frame_difference(
                        current_comparison,
                        last_saved_positive,
                    )

                    if difference >= MINIMUM_FRAME_DIFFERENCE:
                        path = save_frame(
                            frame,
                            POSITIVE_DIR,
                            "ball",
                            positive_index,
                        )
                        positive_index += 1
                        positive_count += 1
                        last_saved_positive = current_comparison.copy()
                        print(
                            f"Saved positive: {path.name} "
                            f"(difference {difference:.1f})"
                        )
                        auto_saved = True

                elif auto_mode == "negative":
                    difference = frame_difference(
                        current_comparison,
                        last_saved_negative,
                    )

                    if difference >= MINIMUM_FRAME_DIFFERENCE:
                        path = save_frame(
                            frame,
                            NEGATIVE_DIR,
                            "no_ball",
                            negative_index,
                        )
                        negative_index += 1
                        negative_count += 1
                        last_saved_negative = current_comparison.copy()
                        print(
                            f"Saved negative: {path.name} "
                            f"(difference {difference:.1f})"
                        )
                        auto_saved = True

                last_auto_capture_time = now

            display = frame.copy()

            mode_text = (
                "MANUAL"
                if auto_mode is None
                else (
                    "AUTO: BALL PRESENT"
                    if auto_mode == "positive"
                    else "AUTO: NO BALL"
                )
            )

            cv2.putText(
                display,
                mode_text,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display,
                f"Ball images: {positive_count}  No-ball images: {negative_count}",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                display,
                "P ball | N no-ball | A auto-ball | G auto-no-ball | Space stop | Q quit",
                (10, display.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1,
            )

            if auto_saved:
                cv2.rectangle(
                    display,
                    (2, 2),
                    (display.shape[1] - 3, display.shape[0] - 3),
                    (0, 255, 0),
                    4,
                )

            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                break

            if key in (ord("p"), ord("P")):
                path = save_frame(
                    frame,
                    POSITIVE_DIR,
                    "ball",
                    positive_index,
                )
                positive_index += 1
                positive_count += 1
                last_saved_positive = current_comparison.copy()
                print(f"Saved positive: {path.name}")

            elif key in (ord("n"), ord("N")):
                path = save_frame(
                    frame,
                    NEGATIVE_DIR,
                    "no_ball",
                    negative_index,
                )
                negative_index += 1
                negative_count += 1
                last_saved_negative = current_comparison.copy()
                print(f"Saved negative: {path.name}")

            elif key in (ord("a"), ord("A")):
                auto_mode = None if auto_mode == "positive" else "positive"
                print(
                    "Automatic ball-present capture "
                    + ("enabled" if auto_mode == "positive" else "disabled")
                )

            elif key in (ord("g"), ord("G")):
                auto_mode = None if auto_mode == "negative" else "negative"
                print(
                    "Automatic no-ball capture "
                    + ("enabled" if auto_mode == "negative" else "disabled")
                )

            elif key == ord(" "):
                auto_mode = None
                print("Automatic capture stopped")

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print()
        print("Capture complete")
        print(f"Ball-present images: {positive_count}")
        print(f"No-ball images: {negative_count}")
        print(f"Saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
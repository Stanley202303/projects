#!/usr/bin/env python3
"""Pico Geometry Level Studio.

A Tkinter desktop editor for the Pico 2 W Geometry Runner.
Levels are saved as editable JSON projects and exported/uploaded as compact
GDL1 binary files.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import struct
import sys
import threading
import time
import tkinter as tk
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

GRID = 30
GROUND_Y = 282

# The Pico game ground is y=282, which is not divisible by 30.
# Horizontal construction lines therefore use the ground as their
# vertical origin: ..., 222, 252, 282, 312, ...
GRID_Y_OFFSET = GROUND_Y % GRID

# Editor-only vertical design range. The Pico world can already store
# signed y coordinates, so this simply exposes much more space above
# the normal 320 px playfield.
EDITOR_WORLD_TOP = -900
EDITOR_FLOOR_MARGIN = 18
PLAYER_X = 88
PLAYER_SIZE = 25
MAX_OBJECTS = 128
MAX_LENGTH = 32700

MAGIC = b"GDL1"
FORMAT_VERSION = 1
HEADER_STRUCT = struct.Struct("<4sBBHIH20s")
OBJECT_STRUCT = struct.Struct("<IhHHBB")

TYPE_SPIKE = 0
TYPE_BLOCK = 1
TYPE_PAD = 2
TYPE_ORB = 3
TYPE_PIT = 4
TYPE_PINK_ORB = 5
TYPE_BLUE_ORB = 6
TYPE_SHORT_SPIKE = 7
TYPE_CEILING_SPIKE = 8
TYPE_SHORT_CEILING_SPIKE = 9

FLOOR_SPIKE_TYPES = {
    TYPE_SPIKE,
    TYPE_SHORT_SPIKE,
}

CEILING_SPIKE_TYPES = {
    TYPE_CEILING_SPIKE,
    TYPE_SHORT_CEILING_SPIKE,
}

SPIKE_TYPES = FLOOR_SPIKE_TYPES | CEILING_SPIKE_TYPES

ORB_TYPES = {
    TYPE_ORB,
    TYPE_PINK_ORB,
    TYPE_BLUE_ORB,
}

ORB_COLOURS = {
    TYPE_ORB: "#ffff00",
    TYPE_PINK_ORB: "#ff55cc",
    TYPE_BLUE_ORB: "#3388ff",
}

# Tight, deliberate activation area shared by all orb colours.
# The test is elliptical rather than rectangular.
ORB_ACTIVATION_REACH_X = 40.0
ORB_ACTIVATION_REACH_Y = 44.0
ORB_INPUT_BUFFER_SECONDS = 0.10

TYPE_NAMES = {
    TYPE_SPIKE: "spike",
    TYPE_BLOCK: "block",
    TYPE_PAD: "pad",
    TYPE_ORB: "yellow_orb",
    TYPE_PIT: "pit",
    TYPE_PINK_ORB: "pink_orb",
    TYPE_BLUE_ORB: "blue_orb",
    TYPE_SHORT_SPIKE: "short_spike",
    TYPE_CEILING_SPIKE: "ceiling_spike",
    TYPE_SHORT_CEILING_SPIKE: "short_ceiling_spike",
}
NAME_TO_TYPE = {value: key for key, value in TYPE_NAMES.items()}

ACCENTS = [
    ("Cyan", "#00ffff"),
    ("Magenta", "#ff00ff"),
    ("Orange", "#ff8800"),
    ("Green", "#00ff66"),
    ("Yellow", "#ffff00"),
    ("Red", "#ff3344"),
    ("Sky", "#88ccff"),
    ("Blue", "#3388ff"),
]


@dataclass
class LevelObject:
    x: int
    base_y: int
    width: int
    height: int
    type: int
    flags: int = 0

    @property
    def top(self) -> int:
        return self.base_y - self.height

    @property
    def right(self) -> int:
        return self.x + self.width


@dataclass
class LevelProject:
    name: str = "MY LEVEL"
    speed: float = 210.0
    length: int = 6000
    accent: int = 0
    objects: list[LevelObject] = None

    def __post_init__(self) -> None:
        if self.objects is None:
            self.objects = []

    def to_json_dict(self) -> dict:
        return {
            "format": "PicoGeometryProject1",
            "name": self.name,
            "speed": self.speed,
            "length": self.length,
            "accent": self.accent,
            "objects": [asdict(item) for item in self.objects],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "LevelProject":
        if data.get("format") != "PicoGeometryProject1":
            raise ValueError("Unsupported project format")
        objects = [LevelObject(**item) for item in data.get("objects", [])]
        project = cls(
            name=str(data.get("name", "MY LEVEL")),
            speed=float(data.get("speed", 210.0)),
            length=int(data.get("length", 6000)),
            accent=int(data.get("accent", 0)),
            objects=objects,
        )

        normalise_project_pits(
            project
        )

        return project


def normalise_project_pits(
    project: LevelProject,
) -> int:
    """Force every pit onto the Geometry Runner ground line."""
    corrected = 0

    for item in project.objects:
        if item.type != TYPE_PIT:
            continue

        if (
            item.base_y != GROUND_Y
            or item.height != 4
        ):
            corrected += 1

        item.base_y = GROUND_Y
        item.height = 4

    return corrected


def encode_gdl(project: LevelProject) -> bytes:
    normalise_project_pits(
        project
    )

    errors, _warnings = validate_project(project)
    if errors:
        raise ValueError("; ".join(errors))

    name_bytes = project.name.encode("ascii", errors="replace")[:20]
    name_bytes = name_bytes.ljust(20, b"\0")
    speed_tenths = int(round(project.speed * 10.0))

    output = bytearray(
        HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            project.accent % len(ACCENTS),
            speed_tenths,
            project.length,
            len(project.objects),
            name_bytes,
        )
    )

    for item in project.objects:
        output.extend(
            OBJECT_STRUCT.pack(
                item.x,
                item.base_y,
                item.width,
                item.height,
                item.type,
                item.flags,
            )
        )

    return bytes(output)


def decode_gdl(data: bytes) -> LevelProject:
    if len(data) < HEADER_STRUCT.size:
        raise ValueError("File is too short")

    magic, version, accent, speed_tenths, length, count, raw_name = \
        HEADER_STRUCT.unpack_from(data, 0)

    if magic != MAGIC:
        raise ValueError("Not a GDL1 level")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported GDL version {version}")

    expected = HEADER_STRUCT.size + count * OBJECT_STRUCT.size
    if len(data) != expected:
        raise ValueError(f"File size mismatch: expected {expected}, got {len(data)}")

    objects = []
    offset = HEADER_STRUCT.size
    for _ in range(count):
        x, base_y, width, height, item_type, flags = OBJECT_STRUCT.unpack_from(data, offset)
        offset += OBJECT_STRUCT.size
        objects.append(LevelObject(x, base_y, width, height, item_type, flags))

    project = LevelProject(
        name=raw_name.split(b"\0", 1)[0].decode("ascii", errors="replace") or "CUSTOM",
        speed=speed_tenths / 10.0,
        length=length,
        accent=accent,
        objects=objects,
    )
    errors, _warnings = validate_project(project)
    if errors:
        raise ValueError("; ".join(errors))
    return project


def validate_project(project: LevelProject) -> tuple[list[str], list[str]]:
    normalise_project_pits(
        project
    )

    errors: list[str] = []
    warnings: list[str] = []

    if not project.name.strip():
        errors.append("Level name is empty")
    if len(project.name.encode("ascii", errors="replace")) > 20:
        warnings.append("Level name will be shortened to 20 characters")
    if not 80.0 <= project.speed <= 350.0:
        errors.append("Speed must be between 80 and 350 px/s")
    if not 300 <= project.length <= MAX_LENGTH:
        errors.append(f"Length must be between 300 and {MAX_LENGTH}")
    if len(project.objects) == 0:
        errors.append("The level has no objects")
    if len(project.objects) > MAX_OBJECTS:
        errors.append(f"Too many objects: {len(project.objects)} / {MAX_OBJECTS}")

    for index, item in enumerate(project.objects, start=1):
        if item.type not in TYPE_NAMES:
            errors.append(f"Object {index}: invalid type")
        if not 0 <= item.x <= MAX_LENGTH:
            errors.append(f"Object {index}: x is outside the supported range")
        if item.x + item.width > project.length + 600:
            warnings.append(f"Object {index}: extends beyond the finish")
        if not -200 <= item.base_y <= 600:
            errors.append(f"Object {index}: base_y is outside the supported range")
        if not 1 <= item.width <= 20000:
            errors.append(f"Object {index}: invalid width")
        if not 1 <= item.height <= 600:
            errors.append(f"Object {index}: invalid height")

    first_hazard = min(
        (item.x for item in project.objects if item.type in SPIKE_TYPES | {TYPE_PIT}),
        default=None,
    )
    if first_hazard is not None and first_hazard < 90:
        warnings.append("A hazard begins very close to the player start")

    sorted_x = sorted({item.x for item in project.objects})
    if len(sorted_x) > 1:
        max_gap = max(b - a for a, b in zip(sorted_x, sorted_x[1:]))
        if max_gap > 480:
            warnings.append(f"There is a {max_gap}px stretch with no new object")

    if not any(item.type == TYPE_PIT for item in project.objects):
        warnings.append("No pits: the ground remains solid everywhere")

    return errors, warnings


class SerialUploader:
    def __init__(self) -> None:
        try:
            import serial  # type: ignore
            from serial.tools import list_ports  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyserial is not installed. Run: python3 -m pip install pyserial"
            ) from exc
        self.serial_module = serial
        self.list_ports_module = list_ports

    def ports(self) -> list[str]:
        return [port.device for port in self.list_ports_module.comports()]

    @staticmethod
    def _readline(port, deadline: float) -> str:
        while time.monotonic() < deadline:
            raw = port.readline()
            if raw:
                return raw.decode("utf-8", errors="replace").strip()
        raise TimeoutError("Timed out waiting for the Pico")

    def connect(self, device: str):
        port = self.serial_module.Serial(device, 115200, timeout=0.2, write_timeout=3)
        time.sleep(0.8)
        port.reset_input_buffer()
        port.write(b"HELLO\n")
        port.flush()
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            line = self._readline(port, deadline)
            if line.startswith("PICOLEVEL 1 "):
                parts = line.split()

                if len(parts) < 4 or parts[3] != "UPLOAD":
                    port.close()
                    raise RuntimeError(
                        "The Pico was found, but it is not on the "
                        "LEVEL UPLOAD screen. Select LEVEL UPLOAD "
                        "from the Pico main menu, then try again."
                    )

                return port

        port.close()
        raise RuntimeError(
            "This serial port did not answer as a Level Studio Pico. "
            "Flash the included pencilcase.ino and select LEVEL UPLOAD."
        )

    @staticmethod
    def _raise_protocol_error(line: str) -> None:
        if line == "ERR MODE":
            raise RuntimeError(
                "Select LEVEL UPLOAD on the Pico main menu and keep "
                "that screen open."
            )

        if line == "ERR FS":
            raise RuntimeError(
                "The Pico filesystem is unavailable. In Arduino IDE, "
                "choose a Flash Size option that includes a filesystem, "
                "then upload pencilcase.ino again."
            )

        raise RuntimeError(line)

    def upload(self, device: str, slot: int, payload: bytes, progress=None) -> str:
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        port = self.connect(device)
        try:
            command = f"PUT {slot} {len(payload)} {crc:08x}\n".encode("ascii")
            port.write(command)
            port.flush()
            deadline = time.monotonic() + 4.0
            while True:
                line = self._readline(port, deadline)
                if line == "READY":
                    break
                if line.startswith("ERR"):
                    self._raise_protocol_error(line)

            chunk_size = 128
            for offset in range(0, len(payload), chunk_size):
                chunk = payload[offset:offset + chunk_size]
                port.write(chunk)
                if progress:
                    progress(offset + len(chunk), len(payload))
            port.flush()

            deadline = time.monotonic() + 8.0
            while True:
                line = self._readline(port, deadline)
                if line.startswith("OK "):
                    return line
                if line.startswith("ERR"):
                    self._raise_protocol_error(line)
        finally:
            port.close()

    def delete(self, device: str, slot: int) -> str:
        port = self.connect(device)
        try:
            port.write(f"DEL {slot}\n".encode("ascii"))
            port.flush()
            deadline = time.monotonic() + 4.0
            while True:
                line = self._readline(port, deadline)
                if line.startswith("OK DEL"):
                    return line
                if line.startswith("ERR"):
                    self._raise_protocol_error(line)
        finally:
            port.close()

    def list_slots(self, device: str) -> list[str]:
        port = self.connect(device)
        try:
            port.write(b"LIST\n")
            port.flush()
            result = []
            deadline = time.monotonic() + 4.0
            while True:
                line = self._readline(port, deadline)
                if line == "END":
                    return result
                if line.startswith("SLOT "):
                    result.append(line)
                elif line.startswith("ERR"):
                    self._raise_protocol_error(line)
        finally:
            port.close()


class PlaytestWindow(tk.Toplevel):
    GRAVITY = 1080.0
    JUMP_SPEED = -410.0
    PAD_SPEED = -575.0
    YELLOW_ORB_SPEED = -485.0
    PINK_ORB_SPEED = -315.0
    BLUE_ORB_FLIP_SPEED = 360.0
    TERMINAL = 760.0
    WORLD_TOP_LIMIT = -260.0
    ORB_PULSE_SECONDS = 0.24

    def __init__(self, parent: tk.Misc, project: LevelProject) -> None:
        super().__init__(parent)
        self.title(f"Playtest — {project.name}")
        self.resizable(False, False)
        self.project = copy.deepcopy(project)
        self.canvas = tk.Canvas(self, width=480, height=320, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.status = ttk.Label(self, text="Up = jump | Space = activate nearest orb | R = restart | Esc = close")
        self.status.pack(fill="x")
        self.bind("<Up>", lambda _event: self.request_jump(False))
        self.bind("<space>", lambda _event: self.request_jump(True))
        self.bind("r", lambda _event: self.reset())
        self.bind("<Escape>", lambda _event: self.destroy())
        self.focus_force()
        self.running = True
        self.last_time = time.monotonic()
        self.accumulator = 0.0
        self.reset()
        self.after(8, self.tick)

    def reset(self) -> None:
        self.distance = 0.0
        self.y = GROUND_Y - PLAYER_SIZE
        self.vy = 0.0
        self.grounded = True
        self.jump_buffer = 0.0
        self.orb_buffer = 0.0
        self.last_pad = None
        self.gravity_direction = 1
        self.orb_pulse_item = None
        self.orb_pulse_timer = 0.0
        self.crashed = False
        self.complete = False
        self.camera_y = 0.0
        self.status.configure(text="Up = jump | Space = activate nearest orb | R = restart | Esc = close")
        self.draw()

    def request_jump(self, button: bool) -> None:
        if self.crashed or self.complete:
            self.reset()
            return
        self.jump_buffer = 0.15
        if button:
            self.orb_buffer = ORB_INPUT_BUFFER_SECONDS

    def _screen_x(self, item: LevelObject) -> float:
        return PLAYER_X + item.x - self.distance

    def step(self, dt: float) -> None:
        if self.crashed or self.complete:
            return

        self.jump_buffer = max(0.0, self.jump_buffer - dt)
        self.orb_buffer = max(0.0, self.orb_buffer - dt)
        self.orb_pulse_timer = max(0.0, self.orb_pulse_timer - dt)
        if self.orb_pulse_timer <= 0.0:
            self.orb_pulse_item = None

        if self.jump_buffer > 0.0 and self.grounded:
            self.vy = self.JUMP_SPEED * self.gravity_direction
            self.grounded = False
            self.jump_buffer = 0.0

        self.distance += self.project.speed * dt
        if self.distance >= self.project.length:
            self.complete = True
            self.status.configure(text="LEVEL COMPLETE — R or Space to restart")
            return

        old_y = self.y
        previous_top = old_y
        previous_bottom = old_y + PLAYER_SIZE

        self.vy += self.GRAVITY * self.gravity_direction * dt
        self.vy = max(-self.TERMINAL, min(self.TERMINAL, self.vy))
        self.y += self.vy * dt

        current_top = self.y
        current_bottom = self.y + PLAYER_SIZE

        ground_available = self.gravity_direction > 0
        if ground_available:
            for item in self.project.objects:
                if item.type != TYPE_PIT:
                    continue
                sx = self._screen_x(item)
                if (
                    PLAYER_X + PLAYER_SIZE - 4 > sx
                    and PLAYER_X + 4 < sx + item.width
                ):
                    ground_available = False
                    break

        if self.gravity_direction > 0:
            support_y = (
                GROUND_Y - PLAYER_SIZE
                if ground_available
                else 1_000_000.0
            )
        else:
            support_y = -1_000_000.0

        found_support = False

        for item in self.project.objects:
            if item.type != TYPE_BLOCK:
                continue

            sx = self._screen_x(item)
            if not (
                PLAYER_X + PLAYER_SIZE - 3 > sx
                and PLAYER_X + 3 < sx + item.width
            ):
                continue

            block_top = item.top
            block_bottom = item.base_y

            if self.gravity_direction > 0:
                crossed_support = (
                    self.vy >= 0
                    and previous_bottom <= block_top + 4
                    and current_bottom >= block_top
                )
                on_support = (
                    self.grounded
                    and abs(previous_bottom - block_top) < 6
                )

                if crossed_support or on_support:
                    candidate = block_top - PLAYER_SIZE
                    if not found_support or candidate < support_y:
                        support_y = candidate
                        found_support = True
                    continue
            else:
                crossed_support = (
                    self.vy <= 0
                    and previous_top >= block_bottom - 4
                    and current_top <= block_bottom
                )
                on_support = (
                    self.grounded
                    and abs(previous_top - block_bottom) < 6
                )

                if crossed_support or on_support:
                    candidate = block_bottom
                    if not found_support or candidate > support_y:
                        support_y = candidate
                        found_support = True
                    continue

            vertical_overlap = (
                self.y + PLAYER_SIZE - 4 > block_top + 4
                and self.y + 4 < block_bottom
            )
            if vertical_overlap:
                self.crash()
                return

        if self.gravity_direction > 0:
            reached_support = self.y >= support_y
        else:
            reached_support = self.y <= support_y

        if reached_support:
            self.y = support_y
            self.vy = 0.0
            self.grounded = True
        else:
            self.grounded = False

        if self.gravity_direction > 0:
            if (
                not ground_available
                and not found_support
                and self.y + PLAYER_SIZE >= GROUND_Y
            ):
                self.crash()
                return
        elif not found_support and self.y <= self.WORLD_TOP_LIMIT:
            self.crash()
            return

        active_pad = None
        for index, item in enumerate(self.project.objects):
            if item.type != TYPE_PAD:
                continue
            sx = self._screen_x(item)
            overlap = PLAYER_X + PLAYER_SIZE - 4 > sx and PLAYER_X + 4 < sx + item.width
            if self.gravity_direction > 0:
                on_surface = abs((self.y + PLAYER_SIZE) - item.base_y) < 8
            else:
                on_surface = abs(self.y - item.top) < 8

            if overlap and on_surface:
                active_pad = index
                if active_pad != self.last_pad:
                    self.vy = self.PAD_SPEED * self.gravity_direction
                    self.grounded = False
                break
        self.last_pad = active_pad

        if self.orb_buffer > 0.0:
            best = None
            best_score = float("inf")
            pcx = PLAYER_X + PLAYER_SIZE / 2
            pcy = self.y + PLAYER_SIZE / 2

            for item in self.project.objects:
                if item.type not in ORB_TYPES:
                    continue

                sx = self._screen_x(item)
                ocx = sx + item.width / 2
                ocy = item.base_y - item.height / 2
                dx = abs(pcx - ocx)
                dy = abs(pcy - ocy)

                normalised_distance = (
                    (dx / ORB_ACTIVATION_REACH_X) ** 2
                    +
                    (dy / ORB_ACTIVATION_REACH_Y) ** 2
                )

                if normalised_distance <= 1.0:
                    if normalised_distance < best_score:
                        best = item
                        best_score = normalised_distance

            if best is not None:
                if best.type == TYPE_ORB:
                    self.vy = (
                        self.YELLOW_ORB_SPEED
                        * self.gravity_direction
                    )
                elif best.type == TYPE_PINK_ORB:
                    self.vy = (
                        self.PINK_ORB_SPEED
                        * self.gravity_direction
                    )
                else:
                    self.gravity_direction *= -1
                    self.vy = (
                        self.BLUE_ORB_FLIP_SPEED
                        * self.gravity_direction
                    )

                self.grounded = False
                self.orb_pulse_item = best
                self.orb_pulse_timer = self.ORB_PULSE_SECONDS
                self.orb_buffer = 0.0
                self.jump_buffer = 0.0

        for item in self.project.objects:
            if item.type not in SPIKE_TYPES:
                continue

            sx = self._screen_x(item)
            left = max(PLAYER_X + 5, sx)
            right = min(PLAYER_X + PLAYER_SIZE - 5, sx + item.width)
            if left >= right:
                continue

            player_top = self.y + 4
            player_bottom = self.y + PLAYER_SIZE - 3

            for sample in range(5):
                sample_x = left + (right - left) * sample / 4
                relative = (sample_x - sx) / item.width
                triangle_height = item.height * (
                    1 - abs(2 * relative - 1)
                )

                if item.type in FLOOR_SPIKE_TYPES:
                    surface_y = item.base_y - triangle_height
                    hit = (
                        player_bottom >= surface_y
                        and player_top < item.base_y
                    )
                else:
                    surface_y = item.base_y + triangle_height
                    hit = (
                        player_top <= surface_y
                        and player_bottom > item.base_y
                    )

                if hit:
                    self.crash()
                    return

        desired_camera = max(0.0, 170.0 - self.y)
        self.camera_y += (desired_camera - self.camera_y) * min(1.0, dt * 8)

    def crash(self) -> None:
        self.crashed = True
        self.status.configure(text="CRASHED — R or Space to restart")

    def draw(self) -> None:
        self.canvas.delete("all")
        accent = ACCENTS[self.project.accent % len(ACCENTS)][1]
        camera = self.camera_y
        self.canvas.create_line(0, GROUND_Y + camera, 480, GROUND_Y + camera, fill="white")

        for item in self.project.objects:
            sx = self._screen_x(item)
            if sx > 520 or sx + item.width < -40:
                continue
            top = item.top + camera
            base = item.base_y + camera
            if item.type == TYPE_PIT:
                self.canvas.create_line(sx, GROUND_Y + camera, sx + item.width, GROUND_Y + camera, fill="black", width=5)
            elif item.type == TYPE_BLOCK:
                self.canvas.create_rectangle(sx, top, sx + item.width, base, outline="white", fill="black")
                self.canvas.create_line(sx + 2, top + 2, sx + item.width - 2, top + 2, fill=accent)
            elif item.type in FLOOR_SPIKE_TYPES:
                self.canvas.create_polygon(
                    sx,
                    base,
                    sx + item.width / 2,
                    top,
                    sx + item.width,
                    base,
                    fill="white",
                    outline=accent,
                )
            elif item.type in CEILING_SPIKE_TYPES:
                self.canvas.create_polygon(
                    sx,
                    base,
                    sx + item.width / 2,
                    base + item.height,
                    sx + item.width,
                    base,
                    fill="white",
                    outline=accent,
                )
            elif item.type == TYPE_PAD:
                self.canvas.create_rectangle(sx, top, sx + item.width, base, fill="yellow", outline="white")
            elif item.type in ORB_TYPES:
                orb_colour = ORB_COLOURS[item.type]
                self.canvas.create_oval(
                    sx,
                    top,
                    sx + item.width,
                    base,
                    fill=orb_colour,
                    outline="white",
                    width=2,
                )
                self.canvas.create_oval(
                    sx + 8,
                    top + 8,
                    sx + item.width - 8,
                    base - 8,
                    fill="black",
                    outline="",
                )

                if (
                    item is self.orb_pulse_item
                    and self.orb_pulse_timer > 0.0
                ):
                    progress = 1.0 - (
                        self.orb_pulse_timer
                        / self.ORB_PULSE_SECONDS
                    )
                    radius = 15 + int(progress * 12)
                    centre_x = sx + item.width / 2
                    centre_y = top + item.height / 2
                    self.canvas.create_oval(
                        centre_x - radius,
                        centre_y - radius,
                        centre_x + radius,
                        centre_y + radius,
                        outline=orb_colour,
                        width=2,
                    )

        py = self.y + camera
        self.canvas.create_rectangle(
            PLAYER_X,
            py,
            PLAYER_X + PLAYER_SIZE,
            py + PLAYER_SIZE,
            fill="#00ffff",
            outline="white",
        )

        indicator_y = (
            py + PLAYER_SIZE - 4
            if self.gravity_direction > 0
            else py + 4
        )
        self.canvas.create_line(
            PLAYER_X + 5,
            indicator_y,
            PLAYER_X + PLAYER_SIZE - 5,
            indicator_y,
            fill="black",
            width=2,
        )

        percent = min(100, int(self.distance * 100 / self.project.length))
        self.canvas.create_text(430, 16, text=f"{percent}%", fill="white")

    def tick(self) -> None:
        if not self.winfo_exists():
            return
        now = time.monotonic()
        dt = min(0.05, now - self.last_time)
        self.last_time = now
        self.accumulator += dt
        while self.accumulator >= 1 / 120:
            self.step(1 / 120)
            self.accumulator -= 1 / 120
        self.draw()
        self.after(16, self.tick)


class LevelStudio(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Pico Geometry Level Studio")
        self.geometry("1280x760")
        self.minsize(960, 640)

        self.project = LevelProject()
        self.project_path: Optional[Path] = None
        self.tool = tk.StringVar(value="select")
        self.zoom = tk.DoubleVar(value=1.0)
        self.show_grid = tk.BooleanVar(value=True)
        self.name_var = tk.StringVar(value=self.project.name)
        self.speed_var = tk.DoubleVar(value=self.project.speed)
        self.length_var = tk.IntVar(value=self.project.length)
        self.accent_var = tk.IntVar(value=self.project.accent)
        self.slot_var = tk.IntVar(value=1)
        self.port_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.selected_index: Optional[int] = None
        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_preview: Optional[int] = None
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []
        self.uploader: Optional[SerialUploader] = None
        self.upload_window: Optional[tk.Toplevel] = None
        self.upload_port_box: Optional[ttk.Combobox] = None
        self.upload_status_var = tk.StringVar(value="Not connected")
        self._resize_redraw_pending = False

        self._build_ui()
        self._bind_keys()
        self.refresh_ports()
        self.redraw()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill="x")

        for label, command in [
            ("New", self.new_project),
            ("Open", self.open_project),
            ("Save", self.save_project),
            ("Save As", self.save_project_as),
            ("Import GDL", self.import_gdl),
            ("Export GDL", self.export_gdl),
            ("Playtest", self.playtest),
            ("Validate", self.show_validation),
            ("Send to Pico", self.open_upload_dialog),
        ]:
            ttk.Button(toolbar, text=label, command=command).pack(side="left", padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Delete", command=self.delete_selected).pack(side="left", padx=2)

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main, padding=8)
        right = ttk.Frame(main)
        main.add(left, weight=0)
        main.add(right, weight=1)

        ttk.Label(left, text="Level", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        self._label_entry(left, "Name", self.name_var)
        self._label_spin(left, "Speed px/s", self.speed_var, 80, 350, 1)
        self._label_spin(left, "Length px", self.length_var, 300, MAX_LENGTH, 30)

        ttk.Label(left, text="Accent").pack(anchor="w", pady=(8, 0))
        accent_box = ttk.Combobox(left, state="readonly", values=[name for name, _ in ACCENTS], width=18)
        accent_box.current(self.accent_var.get())
        accent_box.pack(fill="x")
        accent_box.bind("<<ComboboxSelected>>", lambda _e: self._set_accent(accent_box.current()))

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(left, text="Tools", font=("TkDefaultFont", 12, "bold")).pack(anchor="w")
        tools = [
            ("Select / move", "select"),
            ("Block rectangle", "block"),
            ("Normal spike", "spike"),
            ("Short spike", "short_spike"),
            ("Ceiling spike", "ceiling_spike"),
            ("Short ceiling spike", "short_ceiling_spike"),
            ("Jump pad", "pad"),
            ("Yellow orb — strong", "orb"),
            ("Pink orb — weak", "pink_orb"),
            ("Blue orb — flip gravity", "blue_orb"),
            ("Pit", "pit"),
            ("Erase", "erase"),
        ]
        for label, value in tools:
            ttk.Radiobutton(left, text=label, variable=self.tool, value=value).pack(anchor="w")

        ttk.Checkbutton(left, text="Show editor grid", variable=self.show_grid, command=self.redraw).pack(anchor="w", pady=(8, 0))
        ttk.Label(left, text="Zoom").pack(anchor="w", pady=(8, 0))
        zoom_box = ttk.Combobox(
            left,
            state="readonly",
            values=[
                "0.35",
                "0.50",
                "0.75",
                "1.00",
                "1.25",
                "1.50",
            ],
            width=8,
        )
        zoom_box.set("1.00")
        zoom_box.pack(anchor="w")
        zoom_box.bind("<<ComboboxSelected>>", lambda _e: self._set_zoom(float(zoom_box.get())))

        ttk.Separator(left).pack(fill="x", pady=10)
        ttk.Label(
            left,
            text="Upload to Pico",
            font=("TkDefaultFont", 12, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            left,
            text=(
                "On the Pico main menu, open LEVEL UPLOAD first. "
                "Keep that screen open during transfer."
            ),
            wraplength=220,
            foreground="#aa2200",
        ).pack(anchor="w", pady=(0, 6))

        ttk.Label(
            left,
            text="Serial port (or use Send to Pico above)",
        ).pack(anchor="w")
        self.port_box = ttk.Combobox(
            left,
            textvariable=self.port_var,
            width=23,
            state="normal",
        )
        self.port_box.pack(fill="x")
        ttk.Button(left, text="Refresh ports", command=self.refresh_ports).pack(fill="x", pady=2)
        self._label_spin(left, "Slot (1–8)", self.slot_var, 1, 8, 1)
        ttk.Button(left, text="Upload current level", command=self.upload_level).pack(fill="x", pady=(6, 2))
        ttk.Button(left, text="Read slot list", command=self.read_slot_list).pack(fill="x", pady=2)
        ttk.Button(left, text="Delete custom slot", command=self.delete_slot).pack(fill="x", pady=2)

        ttk.Label(left, textvariable=self.status_var, wraplength=220).pack(anchor="w", pady=12)

        canvas_frame = ttk.Frame(right)
        canvas_frame.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="black",
            highlightthickness=0,
        )
        hbar = ttk.Scrollbar(
            canvas_frame,
            orient="horizontal",
            command=self.canvas.xview,
        )
        self.canvas.configure(
            xscrollcommand=hbar.set,
        )
        self.canvas.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        hbar.grid(
            row=1,
            column=0,
            sticky="ew",
        )
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self.canvas_press)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.canvas.bind("<Button-3>", self.canvas_right_click)
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)
        self.canvas.bind("<Configure>", self._schedule_resize_redraw)

    def open_upload_dialog(self) -> None:
        """Open a dedicated, always-visible Pico transfer window."""
        if (
            self.upload_window is not None
            and self.upload_window.winfo_exists()
        ):
            self.upload_window.deiconify()
            self.upload_window.lift()
            self.upload_window.focus_force()
            self.refresh_ports()
            return

        window = tk.Toplevel(self)
        self.upload_window = window
        window.title("Send Level to Pico 2 W")
        window.geometry("540x520")
        window.minsize(500, 470)
        window.transient(self)

        outer = ttk.Frame(window, padding=16)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Send Level to Pico 2 W",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(anchor="w")

        ttk.Label(
            outer,
            text=(
                "On the Pico, select LEVEL UPLOAD and leave that screen "
                "open. Then select its USB serial port below."
            ),
            wraplength=500,
        ).pack(anchor="w", pady=(6, 14))

        port_frame = ttk.LabelFrame(
            outer,
            text="1. Serial port",
            padding=10,
        )
        port_frame.pack(fill="x")

        self.upload_port_box = ttk.Combobox(
            port_frame,
            textvariable=self.port_var,
            state="normal",
            width=48,
        )
        self.upload_port_box.pack(fill="x")

        ttk.Label(
            port_frame,
            text=(
                "On macOS this normally looks like "
                "/dev/cu.usbmodemXXXX. You may type the path manually."
            ),
            wraplength=480,
        ).pack(anchor="w", pady=(4, 8))

        port_buttons = ttk.Frame(port_frame)
        port_buttons.pack(fill="x")

        ttk.Button(
            port_buttons,
            text="Refresh ports",
            command=self.refresh_ports,
        ).pack(side="left")

        ttk.Button(
            port_buttons,
            text="Find Pico automatically",
            command=self.auto_detect_pico,
        ).pack(side="left", padx=(8, 0))

        slot_frame = ttk.LabelFrame(
            outer,
            text="2. Level slot",
            padding=10,
        )
        slot_frame.pack(fill="x", pady=(12, 0))

        ttk.Label(
            slot_frame,
            text="Choose a slot from 1 to 8:",
        ).pack(side="left")

        ttk.Spinbox(
            slot_frame,
            textvariable=self.slot_var,
            from_=1,
            to=8,
            increment=1,
            width=6,
        ).pack(side="left", padx=(10, 0))

        action_frame = ttk.LabelFrame(
            outer,
            text="3. Transfer",
            padding=10,
        )
        action_frame.pack(fill="x", pady=(12, 0))

        ttk.Button(
            action_frame,
            text="UPLOAD CURRENT LEVEL",
            command=self.upload_level,
        ).pack(fill="x", ipady=6)

        secondary = ttk.Frame(action_frame)
        secondary.pack(fill="x", pady=(8, 0))

        ttk.Button(
            secondary,
            text="Read slot list",
            command=self.read_slot_list,
        ).pack(side="left", expand=True, fill="x")

        ttk.Button(
            secondary,
            text="Delete custom slot",
            command=self.delete_slot,
        ).pack(
            side="left",
            expand=True,
            fill="x",
            padx=(8, 0),
        )

        status_frame = ttk.LabelFrame(
            outer,
            text="Connection and transfer status",
            padding=10,
        )
        status_frame.pack(fill="both", expand=True, pady=(12, 0))

        ttk.Label(
            status_frame,
            textvariable=self.upload_status_var,
            wraplength=470,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=470,
            justify="left",
        ).pack(anchor="w", pady=(8, 0))

        ttk.Separator(outer).pack(fill="x", pady=(12, 8))

        ttk.Label(
            outer,
            text=(
                "No ports shown? Install pyserial, use a data-capable USB "
                "cable, reconnect the Pico, and press Refresh ports."
            ),
            wraplength=500,
        ).pack(anchor="w")

        ttk.Button(
            outer,
            text="Close",
            command=window.destroy,
        ).pack(anchor="e", pady=(12, 0))

        def on_close() -> None:
            self.upload_port_box = None
            self.upload_window = None
            window.destroy()

        window.protocol(
            "WM_DELETE_WINDOW",
            on_close,
        )

        self.refresh_ports()
        window.lift()
        window.focus_force()

    def _set_port_values(
        self,
        ports: list[str],
    ) -> None:
        """Update both the sidebar and dialog port selectors."""
        self.port_box["values"] = ports

        if (
            self.upload_port_box is not None
            and self.upload_port_box.winfo_exists()
        ):
            self.upload_port_box["values"] = ports

    def auto_detect_pico(self) -> None:
        """Probe likely USB ports and select the Level Studio Pico."""
        try:
            uploader = self._get_uploader()
            ports = uploader.ports()
        except Exception as exc:
            self.upload_status_var.set(str(exc))
            messagebox.showerror(
                "Cannot list serial ports",
                str(exc),
            )
            return

        if not ports:
            message = (
                "No serial ports were found. Check the USB cable, reconnect "
                "the Pico, and make sure pyserial is installed."
            )
            self.upload_status_var.set(message)
            messagebox.showerror(
                "No serial ports",
                message,
            )
            return

        ordered_ports = sorted(
            ports,
            key=lambda port: (
                0
                if (
                    "usbmodem" in port.lower()
                    or "usbserial" in port.lower()
                    or "acm" in port.lower()
                )
                else 1,
                port,
            ),
        )

        self.upload_status_var.set(
            f"Checking {len(ordered_ports)} port(s)…"
        )
        self.update_idletasks()

        failures: list[str] = []

        for device in ordered_ports:
            self.upload_status_var.set(
                f"Checking {device}…"
            )
            self.update_idletasks()

            try:
                connection = uploader.connect(
                    device
                )
                connection.close()

                self.port_var.set(device)
                self.upload_status_var.set(
                    f"Pico found on {device}"
                )
                self.status_var.set(
                    f"Connected Pico: {device}"
                )

                messagebox.showinfo(
                    "Pico found",
                    (
                        f"Level Studio Pico detected on:\n\n{device}\n\n"
                        "Choose a slot and click UPLOAD CURRENT LEVEL."
                    ),
                )
                return
            except Exception as exc:
                failures.append(
                    f"{device}: {exc}"
                )

        message = (
            "No connected port answered as a Pico on the LEVEL UPLOAD "
            "screen.\n\nOn the Pico choose LEVEL UPLOAD, then try again."
        )

        self.upload_status_var.set(message)

        if failures:
            message += (
                "\n\nChecked:\n"
                + "\n".join(failures[:5])
            )

        messagebox.showerror(
            "Pico not detected",
            message,
        )

    @staticmethod
    def _label_entry(parent, label: str, variable) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        ttk.Entry(parent, textvariable=variable, width=22).pack(fill="x")

    @staticmethod
    def _label_spin(parent, label: str, variable, minimum, maximum, increment) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        ttk.Spinbox(parent, textvariable=variable, from_=minimum, to=maximum, increment=increment, width=18).pack(fill="x")

    def _bind_keys(self) -> None:
        self.bind("<Delete>", lambda _e: self.delete_selected())
        self.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.bind("<Control-z>", lambda _e: self.undo())
        self.bind("<Control-y>", lambda _e: self.redo())
        self.bind("<Control-s>", lambda _e: self.save_project())
        self.bind("<Left>", lambda _e: self.move_selected(-GRID, 0))
        self.bind("<Right>", lambda _e: self.move_selected(GRID, 0))
        self.bind("<Up>", lambda _e: self.move_selected(0, -GRID))
        self.bind("<Down>", lambda _e: self.move_selected(0, GRID))

    def sync_project_fields(self) -> None:
        normalise_project_pits(
            self.project
        )

        self.project.name = self.name_var.get().strip() or "MY LEVEL"
        self.project.speed = float(self.speed_var.get())
        self.project.length = int(self.length_var.get())
        self.project.accent = int(self.accent_var.get())

    def _set_accent(self, value: int) -> None:
        self.accent_var.set(value)
        self.project.accent = value
        self.redraw()

    def _set_zoom(self, value: float) -> None:
        self.zoom.set(value)
        self.redraw()

    def editor_floor_canvas_y(self) -> float:
        """Return the fixed screen-space y coordinate of the floor."""
        canvas_height = max(
            1,
            self.canvas.winfo_height(),
        )
        return max(
            EDITOR_FLOOR_MARGIN,
            canvas_height -
            EDITOR_FLOOR_MARGIN,
        )

    def world_to_canvas(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Map world coordinates with the floor pinned to the bottom."""
        scale = self.zoom.get()
        floor_canvas_y = self.editor_floor_canvas_y()

        return (
            x * scale,
            floor_canvas_y +
            (y - GROUND_Y) * scale,
        )

    def canvas_to_world(
        self,
        event,
    ) -> tuple[int, int]:
        """Convert an editor pointer position back into world coordinates."""
        scale = self.zoom.get()
        floor_canvas_y = self.editor_floor_canvas_y()

        x = (
            self.canvas.canvasx(event.x)
            / scale
        )
        y = (
            GROUND_Y +
            (
                event.y -
                floor_canvas_y
            ) / scale
        )

        return int(x), int(y)

    def _schedule_resize_redraw(
        self,
        _event=None,
    ) -> None:
        """Coalesce Configure events into one redraw."""
        if self._resize_redraw_pending:
            return

        self._resize_redraw_pending = True
        self.after_idle(
            self._finish_resize_redraw
        )

    def _finish_resize_redraw(self) -> None:
        self._resize_redraw_pending = False
        self.redraw()

    @staticmethod
    def snap(value: float) -> int:
        """Snap an x coordinate or movement delta to the 30 px grid."""
        return int(math.floor(value / GRID) * GRID)

    @staticmethod
    def snap_y_cell_top(value: float) -> int:
        """Return the top edge of the ground-aligned grid cell at y.

        Grid rows are measured from GROUND_Y rather than from y=0.
        When the pointer is exactly on a horizontal grid line, choose
        the cell immediately above that line. This makes clicking the
        floor place a spike with its base exactly on the floor.
        """
        clamped_value = max(
            EDITOR_WORLD_TOP,
            min(
                GROUND_Y,
                value,
            ),
        )

        relative = (
            clamped_value -
            GROUND_Y
        ) / GRID

        cell_index = math.floor(
            relative -
            1.0e-9
        )

        return int(
            GROUND_Y +
            cell_index * GRID
        )

    def push_undo(self) -> None:
        self.sync_project_fields()
        self.undo_stack.append(copy.deepcopy(self.project.to_json_dict()))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.project.to_json_dict()))
        self.project = LevelProject.from_json_dict(self.undo_stack.pop())
        self.load_fields_from_project()

    def redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.project.to_json_dict()))
        self.project = LevelProject.from_json_dict(self.redo_stack.pop())
        self.load_fields_from_project()

    def load_fields_from_project(self) -> None:
        corrected_pits = normalise_project_pits(
            self.project
        )

        self.name_var.set(self.project.name)
        self.speed_var.set(self.project.speed)
        self.length_var.set(self.project.length)
        self.accent_var.set(self.project.accent)
        self.selected_index = None

        if corrected_pits:
            self.status_var.set(
                f"Grounded {corrected_pits} pit"
                + (
                    ""
                    if corrected_pits == 1
                    else "s"
                )
            )

        self.redraw()

    def redraw(self) -> None:
        self.sync_project_fields()
        self.canvas.delete("all")
        scale = self.zoom.get()
        width = max(
            self.project.length + 480,
            1200,
        )
        canvas_height = max(
            1,
            self.canvas.winfo_height(),
        )

        # Horizontal scrolling remains available. Vertically, the canvas
        # is viewport-based so the floor never leaves the bottom edge.
        self.canvas.configure(
            scrollregion=(
                0,
                0,
                width * scale,
                canvas_height,
            )
        )

        floor_canvas_y = self.editor_floor_canvas_y()
        visible_world_top = (
            GROUND_Y -
            floor_canvas_y / scale
        )

        if self.show_grid.get():
            for x in range(
                0,
                width + GRID,
                GRID,
            ):
                cx, _ = self.world_to_canvas(
                    x,
                    GROUND_Y,
                )
                self.canvas.create_line(
                    cx,
                    0,
                    cx,
                    floor_canvas_y,
                    fill="#171717",
                )

            first_grid_y = (
                GROUND_Y +
                math.floor(
                    (
                        max(
                            EDITOR_WORLD_TOP,
                            visible_world_top,
                        ) -
                        GROUND_Y
                    ) / GRID
                ) * GRID
            )

            y = first_grid_y
            while y <= GROUND_Y:
                _, cy = self.world_to_canvas(
                    0,
                    y,
                )

                if 0 <= cy <= floor_canvas_y:
                    line_colour = (
                        "#303030"
                        if y == GROUND_Y
                        else "#171717"
                    )
                    self.canvas.create_line(
                        0,
                        cy,
                        width * scale,
                        cy,
                        fill=line_colour,
                    )

                y += GRID

        _, ground = self.world_to_canvas(
            0,
            GROUND_Y,
        )
        self.canvas.create_line(
            0,
            ground,
            width * scale,
            ground,
            fill="white",
            width=2,
        )

        finish_x, _ = self.world_to_canvas(
            self.project.length,
            GROUND_Y,
        )
        self.canvas.create_line(
            finish_x,
            0,
            finish_x,
            ground,
            fill="#00ff66",
            width=2,
        )
        self.canvas.create_text(
            finish_x + 5,
            12,
            text="FINISH",
            fill="#00ff66",
            anchor="nw",
        )

        accent = ACCENTS[self.project.accent % len(ACCENTS)][1]
        for index, item in enumerate(self.project.objects):
            selected = index == self.selected_index
            self.draw_object(item, accent, selected)

        sx, sy = self.world_to_canvas(0, GROUND_Y - PLAYER_SIZE)
        self.canvas.create_rectangle(sx, sy, sx + PLAYER_SIZE * scale, sy + PLAYER_SIZE * scale, fill="#00ffff", outline="white")
        self.canvas.create_text(
            6,
            6,
            text=(
                f"Objects: {len(self.project.objects)}/{MAX_OBJECTS}"
                f"   Floor fixed at y={GROUND_Y}"
                f"   Visible top ≈ {int(visible_world_top)}"
                f"   World limit {EDITOR_WORLD_TOP}"
            ),
            fill="white",
            anchor="nw",
        )

    def draw_object(self, item: LevelObject, accent: str, selected: bool = False) -> None:
        x1, y1 = self.world_to_canvas(item.x, item.top)
        x2, y2 = self.world_to_canvas(item.x + item.width, item.base_y)
        outline = "#ff5555" if selected else "white"
        width = 3 if selected else 1

        if item.type == TYPE_BLOCK:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline=outline, width=width)
            self.canvas.create_line(x1 + 2, y1 + 2, x2 - 2, y1 + 2, fill=accent)
        elif item.type in FLOOR_SPIKE_TYPES:
            self.canvas.create_polygon(
                x1,
                y2,
                (x1 + x2) / 2,
                y1,
                x2,
                y2,
                fill="white",
                outline=accent,
                width=width,
            )
        elif item.type in CEILING_SPIKE_TYPES:
            tip_y = y2 + item.height * self.zoom.get()
            self.canvas.create_polygon(
                x1,
                y2,
                (x1 + x2) / 2,
                tip_y,
                x2,
                y2,
                fill="white",
                outline=accent,
                width=width,
            )
        elif item.type == TYPE_PAD:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="yellow", outline=outline, width=width)
        elif item.type in ORB_TYPES:
            orb_colour = ORB_COLOURS[item.type]
            self.canvas.create_oval(
                x1,
                y1,
                x2,
                y2,
                fill=orb_colour,
                outline=outline,
                width=width,
            )
            inset = 8 * self.zoom.get()
            self.canvas.create_oval(
                x1 + inset,
                y1 + inset,
                x2 - inset,
                y2 - inset,
                fill="black",
                outline="",
            )

            if selected:
                centre_x = (x1 + x2) / 2
                centre_y = (y1 + y2) / 2
                scale = self.zoom.get()

                self.canvas.create_oval(
                    centre_x -
                    ORB_ACTIVATION_REACH_X *
                    scale,
                    centre_y -
                    ORB_ACTIVATION_REACH_Y *
                    scale,
                    centre_x +
                    ORB_ACTIVATION_REACH_X *
                    scale,
                    centre_y +
                    ORB_ACTIVATION_REACH_Y *
                    scale,
                    outline="#55ff88",
                    width=1,
                    dash=(5, 4),
                )
        elif item.type == TYPE_PIT:
            # Use the floor-pinned coordinate transform, exactly like
            # the white ground line.
            _, ground_canvas_y = self.world_to_canvas(
                0,
                GROUND_Y,
            )

            self.canvas.create_line(
                x1,
                ground_canvas_y,
                x2,
                ground_canvas_y,
                fill=(
                    "#ff3333"
                    if selected
                    else "black"
                ),
                width=max(
                    5,
                    width,
                ),
            )

            self.canvas.create_text(
                (x1 + x2) / 2,
                ground_canvas_y - 12,
                text="PIT",
                fill="#ff6666",
            )

    def object_at(self, x: int, y: int) -> Optional[int]:
        for index in range(len(self.project.objects) - 1, -1, -1):
            item = self.project.objects[index]
            if item.type == TYPE_PIT:
                if item.x <= x <= item.right and abs(y - GROUND_Y) <= 16:
                    return index
            elif item.type in CEILING_SPIKE_TYPES:
                if (
                    item.x <= x <= item.right
                    and item.base_y <= y <= item.base_y + item.height
                ):
                    return index
            elif item.x <= x <= item.right and item.top <= y <= item.base_y:
                return index
        return None

    def canvas_press(self, event) -> None:
        x, y = self.canvas_to_world(event)
        tool = self.tool.get()
        if tool == "select":
            self.selected_index = self.object_at(x, y)
            self.drag_start = (x, y)
            self.redraw()
            return
        if tool == "erase":
            index = self.object_at(x, y)
            if index is not None:
                self.push_undo()
                del self.project.objects[index]
                self.selected_index = None
                self.redraw()
            return
        self.drag_start = (x, y)
        if tool in {
            "spike",
            "short_spike",
            "ceiling_spike",
            "short_ceiling_spike",
            "pad",
            "orb",
            "pink_orb",
            "blue_orb",
        }:
            self.push_undo()
            cell_x = self.snap(x)
            cell_y = self.snap_y_cell_top(y)

            if tool == "spike":
                item = LevelObject(
                    cell_x,
                    cell_y + GRID,
                    GRID,
                    GRID,
                    TYPE_SPIKE,
                )
            elif tool == "short_spike":
                item = LevelObject(
                    cell_x,
                    cell_y + GRID,
                    GRID,
                    GRID // 2,
                    TYPE_SHORT_SPIKE,
                )
            elif tool == "ceiling_spike":
                item = LevelObject(
                    cell_x,
                    cell_y,
                    GRID,
                    GRID,
                    TYPE_CEILING_SPIKE,
                )
            elif tool == "short_ceiling_spike":
                item = LevelObject(
                    cell_x,
                    cell_y,
                    GRID,
                    GRID // 2,
                    TYPE_SHORT_CEILING_SPIKE,
                )
            elif tool == "pad":
                item = LevelObject(
                    cell_x,
                    cell_y + GRID,
                    GRID,
                    7,
                    TYPE_PAD,
                )
            else:
                orb_type = {
                    "orb": TYPE_ORB,
                    "pink_orb": TYPE_PINK_ORB,
                    "blue_orb": TYPE_BLUE_ORB,
                }[tool]
                item = LevelObject(
                    cell_x + 3,
                    cell_y + 27,
                    24,
                    24,
                    orb_type,
                )
            self.project.objects.append(item)
            self.selected_index = len(self.project.objects) - 1
            self.redraw()
            self.drag_start = None

    def canvas_drag(self, event) -> None:
        if self.drag_start is None:
            return
        tool = self.tool.get()
        if tool not in {"block", "pit", "select"}:
            return
        if self.drag_preview is not None:
            self.canvas.delete(self.drag_preview)
            self.drag_preview = None
        x0, y0 = self.drag_start
        x1, y1 = self.canvas_to_world(event)
        if tool == "select" and self.selected_index is not None:
            return
        if tool == "pit":
            left = min(self.snap(x0), self.snap(x1))
            right = max(self.snap(x0), self.snap(x1)) + GRID
            cx1, cy = self.world_to_canvas(left, GROUND_Y)
            cx2, _ = self.world_to_canvas(right, GROUND_Y)
            self.drag_preview = self.canvas.create_line(cx1, cy, cx2, cy, fill="#ff5555", width=6)
        else:
            left = min(self.snap(x0), self.snap(x1))
            right = max(self.snap(x0), self.snap(x1)) + GRID
            top = min(
                self.snap_y_cell_top(y0),
                self.snap_y_cell_top(y1),
            )
            bottom = max(
                self.snap_y_cell_top(y0),
                self.snap_y_cell_top(y1),
            ) + GRID
            cx1, cy1 = self.world_to_canvas(left, top)
            cx2, cy2 = self.world_to_canvas(right, bottom)
            self.drag_preview = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#ff5555", width=2)

    def canvas_release(self, event) -> None:
        if self.drag_preview is not None:
            self.canvas.delete(self.drag_preview)
            self.drag_preview = None
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = self.canvas_to_world(event)
        tool = self.tool.get()
        if tool == "select" and self.selected_index is not None:
            dx = self.snap(x1 - x0)
            dy = self.snap(y1 - y0)
            if dx or dy:
                self.push_undo()
                item = self.project.objects[self.selected_index]
                item.x = max(0, item.x + dx)
                if item.type != TYPE_PIT:
                    item.base_y += dy
            self.drag_start = None
            self.redraw()
            return
        if tool in {"block", "pit"}:
            self.push_undo()
            left = min(self.snap(x0), self.snap(x1))
            right = max(self.snap(x0), self.snap(x1)) + GRID
            if tool == "pit":
                # The vertical pointer position is intentionally ignored.
                # Every pit is anchored exactly to the floor.
                item = LevelObject(
                    left,
                    GROUND_Y,
                    right - left,
                    4,
                    TYPE_PIT,
                )
            else:
                top = min(
                    self.snap_y_cell_top(y0),
                    self.snap_y_cell_top(y1),
                )
                bottom = max(
                    self.snap_y_cell_top(y0),
                    self.snap_y_cell_top(y1),
                ) + GRID
                item = LevelObject(left, bottom, right - left, bottom - top, TYPE_BLOCK)
            self.project.objects.append(item)
            self.selected_index = len(self.project.objects) - 1
        self.drag_start = None
        self.redraw()

    def canvas_right_click(self, event) -> None:
        x, y = self.canvas_to_world(event)
        index = self.object_at(x, y)
        if index is not None:
            self.push_undo()
            del self.project.objects[index]
            self.selected_index = None
            self.redraw()

    def mouse_wheel(self, event) -> None:
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def move_selected(self, dx: int, dy: int) -> None:
        if self.selected_index is None:
            return
        self.push_undo()
        item = self.project.objects[self.selected_index]
        item.x = max(0, item.x + dx)
        if item.type != TYPE_PIT:
            item.base_y += dy
        self.redraw()

    def delete_selected(self) -> None:
        if self.selected_index is None:
            return
        self.push_undo()
        del self.project.objects[self.selected_index]
        self.selected_index = None
        self.redraw()

    def new_project(self) -> None:
        if not messagebox.askyesno("New level", "Discard the current level and start again?"):
            return
        self.project = LevelProject()
        self.project_path = None
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.load_fields_from_project()

    def open_project(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Geometry project", "*.json"), ("All files", "*")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.project = LevelProject.from_json_dict(data)
            self.project_path = Path(path)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.load_fields_from_project()
            self.status_var.set(f"Opened {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        self.sync_project_fields()
        self.project_path.write_text(json.dumps(self.project.to_json_dict(), indent=2), encoding="utf-8")
        self.status_var.set(f"Saved {self.project_path.name}")

    def save_project_as(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Geometry project", "*.json")])
        if not path:
            return
        self.project_path = Path(path)
        self.save_project()

    def export_gdl(self) -> None:
        self.sync_project_fields()
        try:
            payload = encode_gdl(self.project)
        except Exception as exc:
            messagebox.showerror("Cannot export", str(exc))
            return
        path = filedialog.asksaveasfilename(defaultextension=".gdl", filetypes=[("Pico Geometry level", "*.gdl")])
        if not path:
            return
        Path(path).write_bytes(payload)
        self.status_var.set(f"Exported {len(payload)} bytes")

    def import_gdl(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Pico Geometry level", "*.gdl")])
        if not path:
            return
        try:
            self.project = decode_gdl(Path(path).read_bytes())
            self.project_path = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.load_fields_from_project()
            self.status_var.set(f"Imported {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Import failed", str(exc))

    def show_validation(self) -> None:
        self.sync_project_fields()
        errors, warnings = validate_project(self.project)
        lines = []
        if errors:
            lines.append("ERRORS:\n- " + "\n- ".join(errors))
        if warnings:
            lines.append("WARNINGS:\n- " + "\n- ".join(warnings))
        if not lines:
            lines.append("No structural problems found. Use Playtest for timing and difficulty.")
        messagebox.showinfo("Validation", "\n\n".join(lines))

    def playtest(self) -> None:
        self.sync_project_fields()
        errors, _warnings = validate_project(self.project)
        if errors:
            messagebox.showerror("Cannot playtest", "\n".join(errors))
            return
        PlaytestWindow(self, self.project)

    def _get_uploader(self) -> SerialUploader:
        if self.uploader is None:
            self.uploader = SerialUploader()
        return self.uploader

    def refresh_ports(self) -> None:
        try:
            uploader = self._get_uploader()
            ports = uploader.ports()
            self._set_port_values(
                ports
            )

            if (
                ports
                and self.port_var.get() not in ports
            ):
                preferred = next(
                    (
                        port
                        for port in ports
                        if (
                            "usbmodem" in port.lower()
                            or "usbserial" in port.lower()
                            or "acm" in port.lower()
                        )
                    ),
                    ports[0],
                )
                self.port_var.set(
                    preferred
                )

            if ports:
                message = (
                    f"Found {len(ports)} serial port(s). "
                    f"Selected: {self.port_var.get()}"
                )
            else:
                message = (
                    "No serial ports found. Reconnect the Pico with a "
                    "data-capable USB cable and press Refresh ports."
                )

            self.status_var.set(message)
            self.upload_status_var.set(message)

        except Exception as exc:
            message = str(exc)
            self.status_var.set(message)
            self.upload_status_var.set(message)

    def upload_level(self) -> None:
        self.sync_project_fields()
        device = self.port_var.get().strip()
        if not device:
            messagebox.showerror("No port", "Choose the Pico serial port first")
            return
        try:
            payload = encode_gdl(self.project)
        except Exception as exc:
            messagebox.showerror("Cannot upload", str(exc))
            return
        slot = int(self.slot_var.get())
        self.status_var.set("Uploading…")
        self.upload_status_var.set(
            f"Uploading to {device}, slot {slot}…"
        )
        self.update_idletasks()
        try:
            result = self._get_uploader().upload(
                device,
                slot,
                payload,
                progress=lambda sent, total: (
                    self.status_var.set(
                        f"Uploading {sent}/{total} bytes"
                    ),
                    self.upload_status_var.set(
                        f"Uploading {sent}/{total} bytes to slot {slot}"
                    ),
                    self.update_idletasks(),
                ),
            )
            self.status_var.set(
                f"Uploaded to slot {slot}: {result}"
            )
            self.upload_status_var.set(
                f"SUCCESS: level stored in Pico slot {slot}"
            )
            messagebox.showinfo(
                "Upload complete",
                (
                    f"Level stored in Pico slot {slot}.\n\n"
                    "Return to the Pico main menu, open Geometry Runner, "
                    "and choose the same slot."
                ),
            )
        except Exception as exc:
            self.status_var.set("Upload failed")
            self.upload_status_var.set(
                f"UPLOAD FAILED: {exc}"
            )
            messagebox.showerror("Upload failed", str(exc))

    def read_slot_list(self) -> None:
        device = self.port_var.get().strip()
        if not device:
            messagebox.showerror("No port", "Choose the Pico serial port first")
            return
        try:
            lines = self._get_uploader().list_slots(device)
            messagebox.showinfo("Pico level slots", "\n".join(lines))
        except Exception as exc:
            messagebox.showerror("Could not read slots", str(exc))

    def delete_slot(self) -> None:
        device = self.port_var.get().strip()
        slot = int(self.slot_var.get())
        if not device:
            messagebox.showerror("No port", "Choose the Pico serial port first")
            return
        if not messagebox.askyesno("Delete custom level", f"Delete the custom file in slot {slot}? The built-in level will return."):
            return
        try:
            result = self._get_uploader().delete(device, slot)
            self.status_var.set(result)
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))


def command_line_upload(args) -> int:
    payload = Path(args.file).read_bytes()
    decode_gdl(payload)
    uploader = SerialUploader()
    print(uploader.upload(args.port, args.slot, payload))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pico Geometry Level Studio")
    subparsers = parser.add_subparsers(dest="command")
    upload_parser = subparsers.add_parser("upload", help="Upload a .gdl file without opening the GUI")
    upload_parser.add_argument("file")
    upload_parser.add_argument("--port", required=True)
    upload_parser.add_argument("--slot", type=int, choices=range(1, 9), required=True)
    args = parser.parse_args()
    if args.command == "upload":
        return command_line_upload(args)
    app = LevelStudio()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
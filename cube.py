#!/usr/bin/env python3
"""
Double-clickable Wi-Fi UDP 3D IMU viewer.

Expected UDP CSV:
time_ms,ax,ay,az,gx,gy,gz,mx,my,mz,temp
"""

import math
import socket
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


def normalize(v):
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-9:
        return None
    return v / n


def quat_multiply(q, r):
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = r
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1,
    ], dtype=float)


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


@dataclass
class MahonyFilter:
    kp: float = 1.5
    ki: float = 0.02

    def __post_init__(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.integral = np.zeros(3)
        self.mag_reference_world = None

    def update(self, gyro, accel, mag, dt):
        if dt <= 0.0 or dt > 0.2:
            return self.q

        a = normalize(accel)
        m = normalize(mag)

        if a is None:
            return self._integrate_gyro(gyro, dt)

        R = quat_to_matrix(self.q)
        gravity_body_est = R.T @ np.array([0.0, 0.0, 1.0])
        error = np.cross(a, gravity_body_est)

        if m is not None:
            if self.mag_reference_world is None:
                self.mag_reference_world = normalize(R @ m)

            if self.mag_reference_world is not None:
                mag_body_est = normalize(R.T @ self.mag_reference_world)
                if mag_body_est is not None:
                    error += np.cross(m, mag_body_est)

        self.integral += self.ki * error * dt
        corrected_gyro = gyro + self.kp * error + self.integral
        return self._integrate_gyro(corrected_gyro, dt)

    def _integrate_gyro(self, gyro, dt):
        omega = np.array([0.0, *gyro])
        q_dot = 0.5 * quat_multiply(self.q, omega)
        self.q += q_dot * dt
        self.q /= np.linalg.norm(self.q)
        return self.q


class UDPIMU:
    def __init__(self, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(("0.0.0.0", port))
        self.socket.setblocking(False)
        self.last_device_time = None
        self.last_sender = None

    def read_latest(self):
        latest = None

        while True:
            try:
                data, sender = self.socket.recvfrom(2048)
            except BlockingIOError:
                break

            self.last_sender = sender
            parsed = self.parse_packet(data.decode("utf-8", errors="ignore").strip())
            if parsed is not None:
                latest = parsed

        return latest

    def parse_packet(self, packet):
        fields = packet.split(",")
        if len(fields) < 11:
            return None

        try:
            values = [float(x) for x in fields[:11]]
        except ValueError:
            return None

        device_time_ms = values[0]
        accel = np.array(values[1:4], dtype=float)
        gyro = np.array(values[4:7], dtype=float)
        mag = np.array(values[7:10], dtype=float)
        temperature = values[10]

        if self.last_device_time is None:
            dt = 0.02
        else:
            dt = (device_time_ms - self.last_device_time) / 1000.0
            if dt <= 0 or dt > 0.2:
                dt = 0.02

        self.last_device_time = device_time_ms
        return dt, accel, gyro, mag, temperature


def make_box_vertices():
    x, y, z = 1.3, 0.65, 0.25
    return np.array([
        [-x, -y, -z], [ x, -y, -z], [ x,  y, -z], [-x,  y, -z],
        [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
    ], dtype=float)


EDGES = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7),
]


def show_viewer(port):
    try:
        imu = UDPIMU(port)
    except OSError as exc:
        messagebox.showerror(
            "UDP listener failed",
            f"Could not listen on UDP port {port}.\n\n{exc}"
        )
        return

    fusion = MahonyFilter()
    vertices = make_box_vertices()

    fig = plt.figure("Live racket orientation over Wi-Fi")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("World X")
    ax.set_ylabel("World Y")
    ax.set_zlabel("World Z")

    lines = [ax.plot([], [], [], linewidth=2)[0] for _ in EDGES]
    status = ax.text2D(
        0.02, 0.96,
        f"Listening on UDP port {port}...",
        transform=ax.transAxes
    )

    ax.plot([0, 1.5], [0, 0], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 1.5], [0, 0], linewidth=1)
    ax.plot([0, 0], [0, 0], [0, 1.5], linewidth=1)

    def update(_frame):
        sample = imu.read_latest()
        if sample is None:
            return [*lines, status]

        dt, accel, gyro, mag, temperature = sample

        # Apply axis remapping here if required.
        # accel = np.array([accel[1], -accel[0], accel[2]])
        # gyro  = np.array([gyro[1],  -gyro[0],  gyro[2]])
        # mag   = np.array([mag[1],   -mag[0],   mag[2]])

        q = fusion.update(gyro, accel, mag, dt)
        R = quat_to_matrix(q)
        rotated = (R @ vertices.T).T

        for line, (i, j) in zip(lines, EDGES):
            points = rotated[[i, j]]
            line.set_data(points[:, 0], points[:, 1])
            line.set_3d_properties(points[:, 2])

        gyro_deg_s = np.linalg.norm(gyro) * 180.0 / math.pi
        sender = imu.last_sender[0] if imu.last_sender else "unknown"

        status.set_text(
            f"ESP32 {sender} | dt {dt*1000:.1f} ms | "
            f"|gyro| {gyro_deg_s:.1f} deg/s | "
            f"T {temperature:.1f} C"
        )
        return [*lines, status]

    animation = FuncAnimation(
        fig,
        update,
        interval=20,
        blit=False,
        cache_frame_data=False
    )

    def on_close(_event):
        imu.socket.close()

    fig.canvas.mpl_connect("close_event", on_close)
    plt.show()


class Launcher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Racket IMU Wi-Fi Viewer")
        self.root.resizable(False, False)

        frame = ttk.Frame(self.root, padding=16)
        frame.grid()

        ttk.Label(frame, text="UDP listening port").grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )

        self.port_var = tk.StringVar(value="5005")
        ttk.Entry(frame, textvariable=self.port_var, width=16).grid(
            row=1, column=0, sticky="ew"
        )

        ttk.Button(
            frame,
            text="Start Wi-Fi viewer",
            command=self.start
        ).grid(row=2, column=0, pady=(14, 0), sticky="ew")

        ttk.Label(
            frame,
            text="ESP32 and computer must be on the same network."
        ).grid(row=3, column=0, pady=(10, 0), sticky="w")

    def start(self):
        try:
            port = int(self.port_var.get())
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            messagebox.showwarning(
                "Invalid port",
                "Enter a UDP port between 1 and 65535."
            )
            return

        self.root.withdraw()
        try:
            show_viewer(port)
        finally:
            self.root.deiconify()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Launcher().run()
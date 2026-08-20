from __future__ import annotations

import argparse
import math
import os
from typing import Optional


def _prompt_text(label: str, default: Optional[str] = None, required: bool = False) -> str:
    while True:
        suffix = f" [{default}]" if default not in (None, "") else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("A value is required.")


def _prompt_float(label: str, default: str) -> str:
    while True:
        value = _prompt_text(label, default=default, required=True)
        try:
            float(value)
            return value
        except ValueError:
            print("Enter a valid number.")


def _prompt_int(label: str, default: str) -> str:
    while True:
        value = _prompt_text(label, default=default, required=True)
        try:
            int(float(value))
            return str(int(float(value)))
        except ValueError:
            print("Enter a valid integer.")


def _interactive_sources(default_sources: Optional[list[str]] = None) -> list[str]:
    default_sources = default_sources or []
    if default_sources and default_sources != ["run"]:
        return default_sources
    first = _prompt_text("Onshape URL or STL for object 1", required=True)
    second = _prompt_text("Onshape URL or STL for object 2 (collision, optional)", default="", required=False)
    return [source for source in (first, second) if source]


def _interactive_env_for_sources(sources: list[str]) -> dict[str, str]:
    print("Interactive CFD runner")
    print("Press Enter to accept the value shown in brackets.")
    print("")

    default_velocity = os.environ.get("VELOCITY", "0.0" if len(sources) == 2 else "-100.0")
    velocity = _prompt_float("Velocity (m/s)", default_velocity)
    iterations = _prompt_int("CFD iterations", os.environ.get("CFD_ITERATIONS", "70"))
    surface_min = _prompt_int("Surface refinement min", os.environ.get("SURFACE_REFINEMENT_MIN", "2"))
    surface_max = _prompt_int("Surface refinement max", os.environ.get("SURFACE_REFINEMENT_MAX", "3"))
    region_refinement = _prompt_int("Region refinement", os.environ.get("REGION_REFINEMENT", "3"))
    collision_speed = ""
    collision_gap = ""
    frame_interval_default = os.environ.get("MOTION_DT", "0.02")
    if len(sources) == 2:
        collision_speed = _prompt_float("Collision impact speed (m/s)", os.environ.get("COLLISION_CONVERGENCE_SPEED_MPS", "1.0"))
        collision_gap = _prompt_float("Initial collision gap (m)", os.environ.get("COLLISION_INITIAL_GAP_M", "0.05"))
    frame_interval = _prompt_float(
        "Frame time interval / simulation time step (s)",
        frame_interval_default,
    )
    if float(frame_interval) <= 0.0:
        raise ValueError("Frame time interval must be greater than zero")
    dynamic_steps_default = os.environ.get("ASSEMBLY_DYNAMIC_STEPS", "3")
    if len(sources) == 2 and "ASSEMBLY_DYNAMIC_STEPS" not in os.environ:
        travel_steps = math.ceil(
            max(float(collision_gap), 0.0)
            / max(float(collision_speed) * float(frame_interval), 1e-12)
        )
        dynamic_steps_default = str(max(3, travel_steps + 10))
    dynamic_steps = _prompt_int("Assembly dynamic steps", dynamic_steps_default)
    case_name = _prompt_text("Case folder name", default=os.environ.get("CASE_NAME", "actual_model_case"), required=True)

    print("")
    print("Run summary")
    for i, source in enumerate(sources, start=1):
        print(f"  source {i}: {source}")
    print(f"  velocity: {velocity} m/s")
    print(f"  CFD iterations: {iterations}")
    print(f"  assembly dynamic steps: {dynamic_steps}")
    print(f"  frame time interval: {frame_interval} s")
    print(f"  total simulated duration: {int(dynamic_steps) * float(frame_interval):.8g} s")
    print(f"  surface refinement: ({surface_min}, {surface_max})")
    print(f"  region refinement: {region_refinement}")
    if len(sources) == 2:
        print(f"  collision impact speed: {collision_speed} m/s")
        print(f"  initial collision gap: {collision_gap} m")
    print(f"  case folder: {case_name}")
    print("")

    confirm = _prompt_text("Start simulation? (y/n)", default="y", required=True).strip().lower()
    if confirm not in {"y", "yes"}:
        raise SystemExit(130)

    env = {
        "VELOCITY": velocity,
        "CFD_ITERATIONS": iterations,
        "ASSEMBLY_DYNAMIC_STEPS": dynamic_steps,
        "MOTION_DT": frame_interval,
        "SURFACE_REFINEMENT_MIN": surface_min,
        "SURFACE_REFINEMENT_MAX": surface_max,
        "REGION_REFINEMENT": region_refinement,
        "CASE_NAME": case_name,
        # The config layer blocks per-variable CFD overrides unless this is enabled.
        "ALLOW_CFD_ENV_OVERRIDES": "1",
    }
    if len(sources) == 2:
        env["COLLISION_CONVERGENCE_SPEED_MPS"] = collision_speed
        env["COLLISION_INITIAL_GAP_M"] = collision_gap
    return env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cfd_motion",
        description="Run the CFD motion solver from an Onshape URL or local STL.",
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="One Onshape URL/local STL, or two sources for a prescribed collision case.",
    )
    parser.add_argument(
        "--collision-speed",
        type=float,
        default=None,
        help="Total closing speed in m/s when two sources are supplied. Defaults to 1.0 m/s.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Simulated seconds between output frames. This also sets the motion integration time step.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use the interactive runner even when a source argument is provided.",
    )
    parser.add_argument(
        "--structural-solver",
        choices=("current", "openradioss"),
        default="current",
        help="Use the current coupled CFD motion path (default) or the standalone OpenRadioss explicit structural backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.interactive or not args.sources or args.sources == ["run"]:
        sources = _interactive_sources(args.sources)
        env_updates = _interactive_env_for_sources(sources)
        os.environ.update(env_updates)
    else:
        sources = args.sources

    if len(sources) == 2 and args.collision_speed is not None:
        os.environ["COLLISION_CONVERGENCE_SPEED_MPS"] = str(max(args.collision_speed, 0.0))
    elif len(sources) == 2 and "COLLISION_CONVERGENCE_SPEED_MPS" not in os.environ:
        os.environ["COLLISION_CONVERGENCE_SPEED_MPS"] = "1.0"
    if len(sources) == 2 and "VELOCITY" not in os.environ:
        os.environ["VELOCITY"] = "0.0"
    if args.frame_interval is not None:
        if args.frame_interval <= 0.0:
            raise SystemExit("--frame-interval must be greater than zero")
        os.environ["MOTION_DT"] = str(args.frame_interval)

    from .runner import run_openradioss_sources, run_source, run_sources

    if args.structural_solver == "openradioss":
        return run_openradioss_sources(sources)

    if len(sources) == 1:
        return run_source(sources[0])
    if len(sources) == 2:
        return run_sources(sources)

    print("Provide either one source or two sources.")
    return 2


__all__ = ["main"]

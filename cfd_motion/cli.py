from __future__ import annotations

import argparse
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


def _interactive_source_and_env() -> tuple[str, dict[str, str]]:
    print("Interactive CFD runner")
    print("Press Enter to accept the value shown in brackets.")
    print("")

    source = _prompt_text("Onshape URL", required=True)
    velocity = _prompt_float("Velocity (m/s)", os.environ.get("VELOCITY", "-100.0"))
    iterations = _prompt_int("CFD iterations", os.environ.get("CFD_ITERATIONS", "70"))
    dynamic_steps = _prompt_int("Assembly dynamic steps", os.environ.get("ASSEMBLY_DYNAMIC_STEPS", "3"))
    surface_min = _prompt_int("Surface refinement min", os.environ.get("SURFACE_REFINEMENT_MIN", "2"))
    surface_max = _prompt_int("Surface refinement max", os.environ.get("SURFACE_REFINEMENT_MAX", "3"))
    region_refinement = _prompt_int("Region refinement", os.environ.get("REGION_REFINEMENT", "3"))
    case_name = _prompt_text("Case folder name", default=os.environ.get("CASE_NAME", "actual_model_case"), required=True)

    print("")
    print("Run summary")
    print(f"  source: {source}")
    print(f"  velocity: {velocity} m/s")
    print(f"  CFD iterations: {iterations}")
    print(f"  assembly dynamic steps: {dynamic_steps}")
    print(f"  surface refinement: ({surface_min}, {surface_max})")
    print(f"  region refinement: {region_refinement}")
    print(f"  case folder: {case_name}")
    print("")

    confirm = _prompt_text("Start simulation? (y/n)", default="y", required=True).strip().lower()
    if confirm not in {"y", "yes"}:
        raise SystemExit(130)

    env = {
        "VELOCITY": velocity,
        "CFD_ITERATIONS": iterations,
        "ASSEMBLY_DYNAMIC_STEPS": dynamic_steps,
        "SURFACE_REFINEMENT_MIN": surface_min,
        "SURFACE_REFINEMENT_MAX": surface_max,
        "REGION_REFINEMENT": region_refinement,
        "CASE_NAME": case_name,
        # The config layer blocks per-variable CFD overrides unless this is enabled.
        "ALLOW_CFD_ENV_OVERRIDES": "1",
    }
    return source, env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m cfd_motion",
        description="Run the CFD motion solver from an Onshape URL or local STL.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Onshape URL or local STL path. Omit to use the interactive runner.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use the interactive runner even when a source argument is provided.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.interactive or args.source in (None, "", "run"):
        source, env_updates = _interactive_source_and_env()
        os.environ.update(env_updates)
    else:
        source = args.source

    from .runner import run_source

    return run_source(source)


__all__ = ["main"]

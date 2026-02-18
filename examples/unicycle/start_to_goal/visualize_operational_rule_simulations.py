"""Visualize operational-rule simulations for unicycle static/dynamic obstacle systems.

Reads a row from D_paired_comparison and re-simulates both controllers to produce:
1) A professional static trajectory figure (PNG)
2) Optional animation (GIF)

The visualization includes:
- Legacy and evolved unicycle trajectories
- Obstacle geometry (and motion path for dynamic case)
- Start/end states
- First failure position where h(x(t), t) < 0 (if any)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array, random
from matplotlib import patches
from matplotlib.animation import FuncAnimation, PillowWriter

import cbfkit.simulation.simulator as sim
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor
from examples.unicycle.start_to_goal.generate_operational_rule_datasets import (
    DYNAMIC_SYSTEM,
    STATIC_SYSTEM,
    make_evolved_controller,
    make_nominal_controller,
    make_state_from_input,
    obstacle_velocity_from_input,
    safety_function,
)


def load_dataset_row(csv_path: Path, index: int) -> Dict[str, str]:
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if index < 0 or index >= len(rows):
        raise IndexError(f"Index {index} out of range for {csv_path} ({len(rows)} rows)")
    return rows[index]


def infer_system(system_name: str):
    if system_name == STATIC_SYSTEM.name:
        return STATIC_SYSTEM
    if system_name == DYNAMIC_SYSTEM.name:
        return DYNAMIC_SYSTEM
    raise ValueError(f"Unsupported system: {system_name}")


def row_to_vector(row: Dict[str, str], feature_names: List[str]) -> np.ndarray:
    return np.array([float(row[name]) for name in feature_names], dtype=float)


def simulate_trace(
    system_name: str,
    input_vec: np.ndarray,
    controller_kind: str,
    dt: float,
    tf: float,
    kp_pos: float,
    kp_theta: float,
    cbf_alpha: float,
    a_limit: float,
    omega_limit: float,
    seed: int,
    cbf_controller_kind: str = "vanilla",
    disturbance_norm_bound: float = 0.1,
    disturbance_norm: str = "2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    system = infer_system(system_name)
    dynamics = unicycle.plant()
    n_steps = int(tf / dt)
    x0 = make_state_from_input(input_vec)
    desired_state = jnp.array([4.0, 0.0, 0.0, 0.0])
    nominal_controller = make_nominal_controller(dynamics, desired_state, kp_pos, kp_theta)

    if controller_kind == "legacy":
        controller = nominal_controller
    elif controller_kind == "evolved":
        evolved_controller = make_evolved_controller(
            system,
            input_vec,
            dynamics,
            nominal_controller,
            jnp.array([a_limit, omega_limit]),
            cbf_alpha,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
        )

        def nonterminating_evolved_controller(t: float, x: Array):
            u, data = evolved_controller(t, x)
            data = dict(data)
            if bool(data.get("error", False)):
                u_nom, _ = nominal_controller(t, x)
                u = jnp.clip(u_nom, -jnp.array([a_limit, omega_limit]), jnp.array([a_limit, omega_limit]))
            data["error"] = False
            return u, data

        controller = nonterminating_evolved_controller
    else:
        raise ValueError(f"Unsupported controller kind: {controller_kind}")

    states, _u, _z, _p, _dkeys, _dvalues = sim.execute(
        x0=x0,
        dt=dt,
        num_steps=n_steps,
        dynamics=dynamics,
        integrator=integrator,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        key=random.PRNGKey(seed),
        verbose=False,
    )

    traj = np.vstack([np.asarray(x0), np.asarray(states)])
    ts = np.arange(traj.shape[0]) * dt

    h_fun = safety_function(system, input_vec)
    h_vals = np.array([h_fun(float(t), jnp.asarray(x)) for t, x in zip(ts, traj)])
    fail_indices = np.where(h_vals < 0.0)[0]
    fail_idx = int(fail_indices[0]) if len(fail_indices) > 0 else -1

    return traj, ts, h_vals, fail_idx


def obstacle_position(input_vec: np.ndarray, is_dynamic: bool, t: float) -> np.ndarray:
    if not is_dynamic:
        return np.array([0.0, 0.0], dtype=float)
    speed = float(input_vec[4])
    heading = float(input_vec[5])
    return np.array([speed * np.cos(heading) * t, speed * np.sin(heading) * t], dtype=float)


def make_paired_static_plot(
    out_path: Path,
    system_name: str,
    input_vec: np.ndarray,
    legacy_traj: np.ndarray,
    evolved_traj: np.ndarray,
    ts: np.ndarray,
    legacy_fail_idx: int,
    evolved_fail_idx: int,
) -> None:
    is_dynamic = system_name == DYNAMIC_SYSTEM.name
    radius = float(input_vec[2])
    goal = np.array([4.0, 0.0], dtype=float)
    obs_positions = np.array([obstacle_position(input_vec, is_dynamic, float(t)) for t in ts])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(legacy_traj[:, 0], legacy_traj[:, 1], color="#005f73", linewidth=2.2, label="legacy trajectory")
    ax.plot(evolved_traj[:, 0], evolved_traj[:, 1], color="#0a9396", linewidth=2.2, label="evolved trajectory")
    ax.scatter(legacy_traj[0, 0], legacy_traj[0, 1], color="#001219", s=70, marker="o", label="start")
    ax.scatter(goal[0], goal[1], color="#ee9b00", s=100, marker="*", label="goal")

    if legacy_fail_idx >= 0:
        ax.scatter(
            legacy_traj[legacy_fail_idx, 0],
            legacy_traj[legacy_fail_idx, 1],
            color="#bb3e03",
            s=110,
            marker="x",
            linewidth=2.2,
            label=f"legacy failure t={ts[legacy_fail_idx]:.2f}s",
        )
    if evolved_fail_idx >= 0:
        ax.scatter(
            evolved_traj[evolved_fail_idx, 0],
            evolved_traj[evolved_fail_idx, 1],
            color="#9b2226",
            s=110,
            marker="x",
            linewidth=2.2,
            label=f"evolved failure t={ts[evolved_fail_idx]:.2f}s",
        )

    if is_dynamic:
        ax.plot(obs_positions[:, 0], obs_positions[:, 1], "--", color="#ae2012", alpha=0.5, label="obstacle path")
    obs_circle = patches.Circle((obs_positions[0, 0], obs_positions[0, 1]), radius, edgecolor="#ae2012", facecolor="#ee9b00", alpha=0.2)
    ax.add_patch(obs_circle)

    all_x = np.hstack([legacy_traj[:, 0], evolved_traj[:, 0], obs_positions[:, 0], [goal[0]]])
    all_y = np.hstack([legacy_traj[:, 1], evolved_traj[:, 1], obs_positions[:, 1], [goal[1]]])
    ax.set_xlim((float(np.min(all_x) - 1.5), float(np.max(all_x) + 1.5)))
    ax.set_ylim((float(np.min(all_y) - 1.5), float(np.max(all_y) + 1.5)))
    ax.set_title(f"{system_name} | paired legacy vs evolved", fontsize=12, pad=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def make_paired_animation(
    out_path: Path,
    system_name: str,
    input_vec: np.ndarray,
    legacy_traj: np.ndarray,
    evolved_traj: np.ndarray,
    ts: np.ndarray,
    legacy_fail_idx: int,
    evolved_fail_idx: int,
    fps: int,
) -> None:
    is_dynamic = system_name == DYNAMIC_SYSTEM.name
    radius = float(input_vec[2])
    goal = np.array([4.0, 0.0], dtype=float)
    dt = float(ts[1] - ts[0]) if len(ts) > 1 else 0.02
    n_frames = max(len(legacy_traj), len(evolved_traj))
    ts_anim = np.arange(n_frames) * dt
    obs_positions = np.array([obstacle_position(input_vec, is_dynamic, float(t)) for t in ts_anim])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    all_x = np.hstack([legacy_traj[:, 0], evolved_traj[:, 0], obs_positions[:, 0], [goal[0]]])
    all_y = np.hstack([legacy_traj[:, 1], evolved_traj[:, 1], obs_positions[:, 1], [goal[1]]])
    ax.set_xlim((float(np.min(all_x) - 1.5), float(np.max(all_x) + 1.5)))
    ax.set_ylim((float(np.min(all_y) - 1.5), float(np.max(all_y) + 1.5)))

    ax.scatter(goal[0], goal[1], color="#ee9b00", s=100, marker="*", label="goal")
    if is_dynamic:
        ax.plot(obs_positions[:, 0], obs_positions[:, 1], "--", color="#ae2012", alpha=0.5, label="obstacle path")

    (legacy_line,) = ax.plot([], [], color="#005f73", linewidth=2.2, label="legacy trajectory")
    legacy_dot = ax.scatter([], [], color="#005f73", s=65, marker="o", label="legacy robot")
    (evolved_line,) = ax.plot([], [], color="#0a9396", linewidth=2.2, label="evolved trajectory")
    evolved_dot = ax.scatter([], [], color="#0a9396", s=65, marker="s", label="evolved robot")
    obs_circle = patches.Circle((obs_positions[0, 0], obs_positions[0, 1]), radius, edgecolor="#ae2012", facecolor="#ee9b00", alpha=0.25)
    ax.add_patch(obs_circle)
    legacy_fail_marker = ax.scatter([], [], color="#bb3e03", s=120, marker="x", linewidth=2.5, label="legacy failure")
    evolved_fail_marker = ax.scatter([], [], color="#9b2226", s=120, marker="x", linewidth=2.5, label="evolved failure")

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=11, va="top")
    ax.legend(loc="best")

    def update(frame: int):
        legacy_idx = min(frame, len(legacy_traj) - 1)
        evolved_idx = min(frame, len(evolved_traj) - 1)

        legacy_line.set_data(legacy_traj[: legacy_idx + 1, 0], legacy_traj[: legacy_idx + 1, 1])
        legacy_dot.set_offsets(np.array([[legacy_traj[legacy_idx, 0], legacy_traj[legacy_idx, 1]]]))
        evolved_line.set_data(evolved_traj[: evolved_idx + 1, 0], evolved_traj[: evolved_idx + 1, 1])
        evolved_dot.set_offsets(np.array([[evolved_traj[evolved_idx, 0], evolved_traj[evolved_idx, 1]]]))
        obs_circle.center = (obs_positions[frame, 0], obs_positions[frame, 1])

        if legacy_fail_idx >= 0 and frame >= legacy_fail_idx:
            legacy_fail_marker.set_offsets(np.array([[legacy_traj[legacy_fail_idx, 0], legacy_traj[legacy_fail_idx, 1]]]))
        else:
            legacy_fail_marker.set_offsets(np.empty((0, 2)))
        if evolved_fail_idx >= 0 and frame >= evolved_fail_idx:
            evolved_fail_marker.set_offsets(np.array([[evolved_traj[evolved_fail_idx, 0], evolved_traj[evolved_fail_idx, 1]]]))
        else:
            evolved_fail_marker.set_offsets(np.empty((0, 2)))

        time_text.set_text(f"t = {ts_anim[frame]:.2f} s")
        return (
            legacy_line,
            legacy_dot,
            evolved_line,
            evolved_dot,
            obs_circle,
            legacy_fail_marker,
            evolved_fail_marker,
            time_text,
        )

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="examples/unicycle/start_to_goal/results/operational_rule_datasets",
    )
    parser.add_argument("--system", type=str, choices=[STATIC_SYSTEM.name, DYNAMIC_SYSTEM.name], required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else dataset_root / "visualizations" / args.system
    )

    metadata_path = dataset_root / args.system / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    dataset_name = "D_paired_comparison.csv"
    paired_path = dataset_root / args.system / dataset_name
    if not paired_path.exists():
        raise FileNotFoundError(
            f"Paired dataset not found: {paired_path}. "
            "Generate with --paired-index-datasets first."
        )
    row = load_dataset_row(paired_path, args.index)
    print(f"Loaded row {args.index} from {paired_path}:")
    for key, value in row.items():
        print(f"  {key}: {value}")
    feat_names = [f["name"] for f in metadata["input_features"]]
    input_vec = row_to_vector(row, feat_names)

    dt = float(metadata["simulation"]["time_step_s"])
    tf = float(metadata["simulation"]["total_duration_s"])
    kp_pos = float(metadata["controller_parameters"]["legacy"]["Kp_pos"])
    kp_theta = float(metadata["controller_parameters"]["legacy"]["Kp_theta"])
    cbf_alpha = float(metadata["controller_parameters"]["evolved"]["cbf_alpha_linear_class_k"])
    cbf_controller_kind = metadata["controller_parameters"]["evolved"].get(
        "cbf_controller_kind", "vanilla"
    )
    disturbance_norm_bound = float(
        metadata["controller_parameters"]["evolved"].get("disturbance_norm_bound", 0.1)
    )
    disturbance_norm = str(
        metadata["controller_parameters"]["evolved"].get("disturbance_norm", "2")
    )
    a_limit = float(metadata["controller_parameters"]["legacy"]["control_limits"][0])
    omega_limit = float(metadata["controller_parameters"]["legacy"]["control_limits"][1])

    legacy_traj, ts, legacy_h_vals, legacy_fail_idx = simulate_trace(
        system_name=args.system,
        input_vec=input_vec,
        controller_kind="legacy",
        dt=dt,
        tf=tf,
        kp_pos=kp_pos,
        kp_theta=kp_theta,
        cbf_alpha=cbf_alpha,
        a_limit=a_limit,
        omega_limit=omega_limit,
        seed=args.seed,
        cbf_controller_kind=cbf_controller_kind,
        disturbance_norm_bound=disturbance_norm_bound,
        disturbance_norm=disturbance_norm,
    )
    evolved_traj, _, evolved_h_vals, evolved_fail_idx = simulate_trace(
        system_name=args.system,
        input_vec=input_vec,
        controller_kind="evolved",
        dt=dt,
        tf=tf,
        kp_pos=kp_pos,
        kp_theta=kp_theta,
        cbf_alpha=cbf_alpha,
        a_limit=a_limit,
        omega_limit=omega_limit,
        seed=args.seed + 1,
        cbf_controller_kind=cbf_controller_kind,
        disturbance_norm_bound=disturbance_norm_bound,
        disturbance_norm=disturbance_norm,
    )

    stem = f"paired_idx{args.index:04d}"
    fig_path = out_dir / f"{stem}.png"
    make_paired_static_plot(
        fig_path,
        args.system,
        input_vec,
        legacy_traj,
        evolved_traj,
        ts,
        legacy_fail_idx,
        evolved_fail_idx,
    )
    # if args.animate:
    if True:
        gif_path = out_dir / f"{stem}.gif"
        make_paired_animation(
            gif_path,
            args.system,
            input_vec,
            legacy_traj,
            evolved_traj,
            ts,
            legacy_fail_idx,
            evolved_fail_idx,
            args.fps,
        )
        print(f"Saved figure: {fig_path}")
        print(f"Saved paired animation: {gif_path}")
    else:
        print(f"Saved figure: {fig_path}")

    case_id = row.get("case_id", str(args.index))
    print(f"Case ID: {case_id} (row index {args.index}, source {dataset_name})")
    if "legacy_label" in row and "evolved_label" in row:
        print(f"Paired labels from dataset: legacy={row['legacy_label']} evolved={row['evolved_label']}")
    print(f"Legacy min h: {np.min(legacy_h_vals):.6f}")
    print(f"Evolved min h: {np.min(evolved_h_vals):.6f}")
    if legacy_fail_idx >= 0:
        print(
            f"Legacy first failure at t={ts[legacy_fail_idx]:.3f}s, "
            f"position=({legacy_traj[legacy_fail_idx,0]:.3f}, {legacy_traj[legacy_fail_idx,1]:.3f})"
        )
    else:
        print("Legacy has no failure point (safe wrt h >= 0).")
    if evolved_fail_idx >= 0:
        print(
            f"Evolved first failure at t={ts[evolved_fail_idx]:.3f}s, "
            f"position=({evolved_traj[evolved_fail_idx,0]:.3f}, {evolved_traj[evolved_fail_idx,1]:.3f})"
        )
    else:
        print("Evolved has no failure point (safe wrt h >= 0).")


if __name__ == "__main__":
    main()

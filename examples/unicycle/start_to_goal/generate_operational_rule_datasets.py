"""Generate operational-rule datasets for unicycle systems with static and dynamic obstacles.

Creates, for each system:
- D_legacy: nominal controller (no CBF), mixed Pass/Fail outcomes.
- D_evolved: CBF-QP controller, filtered to include only Pass outcomes.

Each row stores the full operational input vector and an outcome label.
Metadata is saved with feature definitions, bounds, units, controller parameters,
simulation settings, and random seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SAFE_RUNTIME_FLAG = "--safe-runtime"
SAFE_RUNTIME_SENTINEL = "CBFKIT_SAFE_RUNTIME_APPLIED"


def _maybe_enable_safe_runtime() -> None:
    """Relaunch the script with conservative CPU/XLA settings."""
    if SAFE_RUNTIME_FLAG not in sys.argv:
        return
    if os.environ.get(SAFE_RUNTIME_SENTINEL) == "1":
        return

    env = os.environ.copy()
    env[SAFE_RUNTIME_SENTINEL] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_PLATFORM_NAME", "cpu")
    env.setdefault("JAX_DISABLE_JIT", "1")
    env.setdefault(
        "XLA_FLAGS",
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    )
    print(
        "[runtime] Enabling safe runtime mode "
        "(OMP/BLAS threads=1, CPU-only backend, JAX JIT disabled)."
    )
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


_maybe_enable_safe_runtime()

import jax.numpy as jnp
import numpy as np
from jax import Array, jacfwd, jacrev, random

import cbfkit.simulation.simulator as sim
from cbfkit.utils import logger as sim_logger
import cbfkit.systems.unicycle.models.accel_unicycle as unicycle
from cbfkit.controllers.model_based.cbf_clf_controllers import (
    robust_cbf_clf_qp_controller,
    vanilla_cbf_clf_qp_controller,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.barrier_conditions import (
    zeroing_barriers,
)
from cbfkit.controllers.model_based.cbf_clf_controllers.utils.certificate_packager import (
    certificate_package,
    concatenate_certificates,
)
from cbfkit.estimators import naive as estimator
from cbfkit.integration import forward_euler as integrator
from cbfkit.sensors import perfect as sensor


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    meaning: str
    units: str
    bounds: Tuple[float, float]


@dataclass(frozen=True)
class SystemSpec:
    name: str
    features: Tuple[FeatureSpec, ...]
    sampling_strategy: str


N_STATES = 4
CBF_VELOCITY_WEIGHT = 1.0
INITIAL_OBSTACLE_CLEARANCE_M = 0.05
TIME_LOG_EVERY_CASES = 5


def moving_circle_cbf(
    obstacle_center: Array,
    obstacle_velocity: Array,
    radius: float,
    velocity_weight: float,
):
    """Velocity-aware CBF for accel-unicycle.

    h_cbf(x,t) = ||p-p_obs(t)||^2 - r^2 + k_v * v * <p-p_obs(t), [cos(theta), sin(theta)]>
    """

    def func(state_and_time: Array) -> Array:
        x, y, _v, _th, t = state_and_time
        ox = obstacle_center[0] + obstacle_velocity[0] * t
        oy = obstacle_center[1] + obstacle_velocity[1] * t
        dx = x - ox
        dy = y - oy
        heading_x = jnp.cos(_th)
        heading_y = jnp.sin(_th)
        dist_term = dx**2 + dy**2 - radius**2
        velocity_term = velocity_weight * _v * (dx * heading_x + dy * heading_y)
        return dist_term + velocity_term

    return func


def moving_circle_cbf_grad(
    obstacle_center: Array,
    obstacle_velocity: Array,
    radius: float,
    velocity_weight: float,
):
    jacobian = jacfwd(
        moving_circle_cbf(
            obstacle_center,
            obstacle_velocity,
            radius,
            velocity_weight,
        )
    )

    def func(state_and_time: Array) -> Array:
        return jacobian(state_and_time)

    return func


def moving_circle_cbf_hess(
    obstacle_center: Array,
    obstacle_velocity: Array,
    radius: float,
    velocity_weight: float,
):
    hessian = jacrev(
        jacfwd(
            moving_circle_cbf(
                obstacle_center,
                obstacle_velocity,
                radius,
                velocity_weight,
            )
        )
    )

    def func(state_and_time: Array) -> Array:
        return hessian(state_and_time)

    return func


velocity_aware_obstacle_ca = certificate_package(
    moving_circle_cbf,
    moving_circle_cbf_grad,
    moving_circle_cbf_hess,
    N_STATES,
)


STATIC_SYSTEM = SystemSpec(
    name="unicycle_static_obstacle",
    sampling_strategy="uniform_random",
    features=(
        FeatureSpec(
            "initial_distance_to_obstacle",
            "Distance from unicycle COM to obstacle center at t=0",
            "m",
            (0.7, 2.6),
        ),
        FeatureSpec("initial_speed", "Initial forward speed v(0)", "m/s", (0.0, 2.5)),
        FeatureSpec("obstacle_radius", "Circular obstacle radius", "m", (0.35, 1.0)),
        FeatureSpec(
            "initial_heading_error",
            "Initial heading error relative to +x direction toward goal",
            "rad",
            (-math.pi, math.pi),
        ),
    ),
)

DYNAMIC_SYSTEM = SystemSpec(
    name="unicycle_dynamic_obstacle",
    sampling_strategy="uniform_random",
    features=(
        FeatureSpec(
            "initial_distance_to_obstacle",
            "Distance from unicycle COM to obstacle center at t=0",
            "m",
            (0.9, 3.0),
        ),
        FeatureSpec("initial_speed", "Initial forward speed v(0)", "m/s", (0.0, 2.5)),
        FeatureSpec("obstacle_radius", "Circular obstacle radius", "m", (0.35, 0.9)),
        FeatureSpec(
            "initial_heading_error",
            "Initial heading error relative to +x direction toward goal",
            "rad",
            (-math.pi, math.pi),
        ),
        FeatureSpec("obstacle_speed", "Obstacle translational speed", "m/s", (0.0, 1.2)),
        FeatureSpec(
            "obstacle_heading",
            "Obstacle heading angle in world frame",
            "rad",
            (-math.pi, math.pi),
        ),
    ),
)


def feature_names(system: SystemSpec) -> List[str]:
    return [f.name for f in system.features]


def sample_inputs(system: SystemSpec, n_samples: int, seed: int) -> np.ndarray:
    """Sample operational input vectors with rejection for initial safety."""
    rng = np.random.default_rng(seed)
    lows = np.array([f.bounds[0] for f in system.features], dtype=float)
    highs = np.array([f.bounds[1] for f in system.features], dtype=float)

    samples: List[np.ndarray] = []
    while len(samples) < n_samples:
        candidate = rng.uniform(lows, highs)
        if initial_clearance_margin(candidate) > INITIAL_OBSTACLE_CLEARANCE_M:
            samples.append(candidate)

    return np.vstack(samples)


def make_state_from_input(vec: np.ndarray) -> jnp.ndarray:
    d0, v0, _r, heading_err = vec[0], vec[1], vec[2], vec[3]
    return jnp.array([-d0, 0.0, v0, heading_err])


def initial_clearance_margin(vec: np.ndarray) -> float:
    """At t=0, margin between robot COM distance to obstacle center and obstacle radius."""
    d0 = float(vec[0])
    radius = float(vec[2])
    return d0 - radius


def is_logical_initial_condition(vec: np.ndarray) -> bool:
    """Reject non-physical starts: inside/on obstacle or too close to boundary."""
    return initial_clearance_margin(vec) > INITIAL_OBSTACLE_CLEARANCE_M


def sample_single_input(rng: np.random.Generator, system: SystemSpec) -> np.ndarray:
    lows = np.array([f.bounds[0] for f in system.features], dtype=float)
    highs = np.array([f.bounds[1] for f in system.features], dtype=float)
    return rng.uniform(lows, highs)


def obstacle_velocity_from_input(system: SystemSpec, vec: np.ndarray) -> jnp.ndarray:
    if system.name == STATIC_SYSTEM.name:
        return jnp.array([0.0, 0.0])
    speed = vec[4]
    heading = vec[5]
    return jnp.array([speed * np.cos(heading), speed * np.sin(heading)])


def safety_function(system: SystemSpec, vec: np.ndarray) -> Callable[[float, Array], float]:
    radius = float(vec[2])
    obs_center = jnp.array([0.0, 0.0])
    obs_vel = obstacle_velocity_from_input(system, vec)

    def h_of_t_x(t: float, x: Array) -> float:
        ox = obs_center[0] + obs_vel[0] * t
        oy = obs_center[1] + obs_vel[1] * t
        return float((x[0] - ox) ** 2 + (x[1] - oy) ** 2 - radius**2)

    return h_of_t_x


def make_nominal_controller(dynamics, desired_state, kp_pos: float, kp_theta: float):
    return unicycle.controllers.proportional_controller(
        dynamics=dynamics,
        Kp_pos=kp_pos,
        Kp_theta=kp_theta,
        desired_state=desired_state,
    )


def make_evolved_controller(
    system: SystemSpec,
    vec: np.ndarray,
    dynamics,
    nominal_controller,
    control_limits: jnp.ndarray,
    alpha: float,
    cbf_controller_kind: str = "vanilla",
    disturbance_norm_bound: float = 0.0,
    disturbance_norm: str = "2",
):
    radius = float(vec[2])

    barrier = velocity_aware_obstacle_ca(
        certificate_conditions=zeroing_barriers.linear_class_k(alpha),
        obstacle_center=jnp.array([0.0, 0.0]),
        obstacle_velocity=obstacle_velocity_from_input(system, vec),
        radius=radius,
        velocity_weight=CBF_VELOCITY_WEIGHT,
    )

    barriers = concatenate_certificates(barrier)

    if cbf_controller_kind == "vanilla":
        return vanilla_cbf_clf_qp_controller(
            control_limits=control_limits,
            nominal_input=nominal_controller,
            dynamics_func=dynamics,
            barriers=barriers,
        )
    if cbf_controller_kind == "robust":
        if disturbance_norm_bound <= 0.0:
            raise ValueError("robust CBF controller requires --disturbance-norm-bound > 0.")
        disturbance_norm_value = 2 if disturbance_norm == "2" else jnp.inf
        return robust_cbf_clf_qp_controller(
            control_limits=control_limits,
            nominal_input=nominal_controller,
            dynamics_func=dynamics,
            barriers=barriers,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm_value,
        )
    raise ValueError(f"Unsupported --cbf-controller: {cbf_controller_kind}")


def run_single_simulation(
    system: SystemSpec,
    vec: np.ndarray,
    use_cbf: bool,
    dt: float,
    total_time: float,
    sim_seed: int,
    kp_pos: float,
    kp_theta: float,
    cbf_alpha: float,
    control_limits: jnp.ndarray,
    cbf_controller_kind: str,
    disturbance_norm_bound: float,
    disturbance_norm: str,
    progress_label: Optional[str] = None,
    progress_step_every: int = 0,
) -> Dict[str, object]:
    initial_margin = initial_clearance_margin(vec)
    if initial_margin <= 0.0:
        raise ValueError(
            "Invalid test case: robot starts inside or on obstacle. "
            f"(initial_distance_to_obstacle={float(vec[0]):.6f}, obstacle_radius={float(vec[2]):.6f})"
        )
    if initial_margin <= INITIAL_OBSTACLE_CLEARANCE_M:
        raise ValueError(
            "Invalid test case: initial clearance is below required minimum. "
            f"(margin={initial_margin:.6f} m, required>{INITIAL_OBSTACLE_CLEARANCE_M:.6f} m)"
        )

    dynamics = unicycle.plant()
    x0 = make_state_from_input(
        vec
    )  # initial state based on input vector (-distance, 0, initial speed, heading error)
    n_steps = int(total_time / dt)

    desired_state = jnp.array([4.0, 0.0, 0.0, 0.0])
    nominal_controller = make_nominal_controller(dynamics, desired_state, kp_pos, kp_theta)
    controller = nominal_controller

    if use_cbf:
        evolved_controller = make_evolved_controller(
            system,
            vec,
            dynamics,
            nominal_controller,
            control_limits,
            cbf_alpha,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
        )

        def nonterminating_evolved_controller(t: float, x: Array):
            u, data = evolved_controller(t, x)
            data = dict(data)
            qp_error = bool(data.get("error", False))

            if qp_error:
                # Keep simulation running after QP infeasibility by falling back to nominal control.
                u_nom, _ = nominal_controller(t, x)
                u = jnp.clip(u_nom, -control_limits, control_limits)

            data["qp_error"] = qp_error
            data["error"] = False
            return u, data

        controller = nonterminating_evolved_controller

    sim_logger.clear_log()
    start_time = time.perf_counter()
    def zero_perturbation(x, _u, _f, _g):
        def p(_subkey):
            return jnp.zeros(x.shape)

        return p

    simulate_iter = sim.simulator(
        dt=dt,
        num_steps=n_steps,
        dynamics=dynamics,
        integrator=integrator,
        controller=controller,
        sensor=sensor,
        estimator=estimator,
        perturbation=zero_perturbation,
        sigma=None,
        key=random.PRNGKey(sim_seed),
        verbose=False,
    )
    simulation_data = []
    step_wall_clock_cumulative_s: List[float] = []
    for sim_step in simulate_iter(x0):
        simulation_data.append(sim_step)
        elapsed_wall = time.perf_counter() - start_time
        step_wall_clock_cumulative_s.append(elapsed_wall)
        if progress_label and progress_step_every > 0:
            step_idx = len(simulation_data)
            if step_idx == 1 or step_idx % progress_step_every == 0:
                print(
                    f"[{system.name}] {progress_label}: "
                    f"step {step_idx}/{n_steps}, sim_time={step_idx * dt:.2f}s, "
                    f"wall_elapsed={elapsed_wall:.2f}s"
                )
    case_runtime_s = time.perf_counter() - start_time
    states, _u, _z, _p, dkeys, dvalues = sim.extract_and_log_data(None, tuple(simulation_data))
    sim_logger.clear_log()

    h = safety_function(system, vec)
    obs_center = jnp.array([0.0, 0.0])
    obs_vel = obstacle_velocity_from_input(system, vec)
    h_values = [h(0.0, x0)]
    h_values.extend(h((i + 1) * dt, xk) for i, xk in enumerate(states))
    min_h = float(min(h_values))
    distance_values = []
    dx_values = []
    dy_values = []
    traj_states = [x0]
    traj_states.extend(states)
    for i, xk in enumerate(traj_states):
        t = i * dt
        ox = float(obs_center[0] + obs_vel[0] * t)
        oy = float(obs_center[1] + obs_vel[1] * t)
        dx = float(xk[0] - ox)
        dy = float(xk[1] - oy)
        distance = float(math.hypot(dx, dy))
        dx_values.append(dx)
        dy_values.append(dy)
        distance_values.append(distance)
    min_distance = float(min(distance_values))
    min_dx = float(min(dx_values))
    min_dy = float(min(dy_values))

    step_logs: List[Dict[str, Any]] = []
    for i, xk in enumerate(states):
        t = (i + 1) * dt
        row: Dict[str, Any] = {
            "step": i + 1,
            "time_s": float(t),
            "state_x": float(xk[0]),
            "state_y": float(xk[1]),
            "state_v": float(xk[2]),
            "state_theta": float(xk[3]),
            "dx": dx_values[i + 1],
            "dy": dy_values[i + 1],
            "distance": distance_values[i + 1],
            "wall_time_cumulative_s": (
                float(step_wall_clock_cumulative_s[i])
                if i < len(step_wall_clock_cumulative_s)
                else float(case_runtime_s)
            ),
        }
        controller_stats: Dict[str, Any] = {}
        if i < len(dvalues):
            for k, v in zip(dkeys, dvalues[i]):
                key = str(k)
                if isinstance(v, (np.bool_, bool)):
                    controller_stats[key] = bool(v)
                elif isinstance(v, (np.integer, int)):
                    controller_stats[key] = int(v)
                elif isinstance(v, (np.floating, float)):
                    controller_stats[key] = float(v)
                else:
                    controller_stats[key] = str(v)
        row["controller_stats_json"] = json.dumps(controller_stats, ensure_ascii=True)
        step_logs.append(row)

    error = False
    if "qp_error" in dkeys:
        error_idx = dkeys.index("qp_error")
        error = any(bool(step_vals[error_idx]) for step_vals in dvalues)
    elif "error" in dkeys:
        error_idx = dkeys.index("error")
        error = any(bool(step_vals[error_idx]) for step_vals in dvalues)

    # Outcome label follows the operational safety rule only.
    passed = min_h >= 0.0

    return {
        "case_id": -1,
        "input_vector": vec.tolist(),
        "label": "Pass" if passed else "Fail",
        "min_h": min_h,
        "min_distance": min_distance,
        "min_dx": min_dx,
        "min_dy": min_dy,
        "controller_error": error,
        "case_runtime_s": case_runtime_s,
        "step_logs": step_logs,
    }


def run_batch_simulation(
    system: SystemSpec,
    input_vectors: np.ndarray,
    use_cbf: bool,
    dt: float,
    total_time: float,
    base_seed: int,
    kp_pos: float,
    kp_theta: float,
    cbf_alpha: float,
    control_limits: jnp.ndarray,
    print_every: int,
    cbf_controller_kind: str,
    disturbance_norm_bound: float,
    disturbance_norm: str,
    on_case_complete: Optional[
        Callable[[int, Dict[str, object], List[Dict[str, object]]], None]
    ] = None,
) -> List[Dict[str, object]]:
    """Batch wrapper requested by task: list of vectors -> pass/fail outcomes."""
    results: List[Dict[str, object]] = []
    controller_name = "evolved(CBF)" if use_cbf else "legacy(nominal)"
    batch_start = time.perf_counter()
    ## The input vector order contains the the samples from the candidate dataset
    for idx, vec in enumerate(input_vectors):
        result = run_single_simulation(
            system=system,
            vec=vec,
            use_cbf=use_cbf,
            dt=dt,
            total_time=total_time,
            sim_seed=base_seed + idx,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
        )
        result["case_id"] = idx
        results.append(result)
        if on_case_complete is not None:
            on_case_complete(idx, result, results)
        # Keep memory bounded in large runs: persist step logs externally, then drop them from RAM.
        results[-1].pop("step_logs", None)
        if (idx + 1) % TIME_LOG_EVERY_CASES == 0:
            elapsed = time.perf_counter() - batch_start
            avg = elapsed / float(idx + 1)
            print(
                f"[{system.name}] {controller_name} timing: "
                f"{idx + 1}/{len(input_vectors)} cases, elapsed={elapsed:.2f}s, avg={avg:.3f}s/case"
            )
        if (idx + 1) % print_every == 0 or (idx + 1) == len(input_vectors):
            print(
                f"[{system.name}] {controller_name} progress: "
                f"{idx + 1}/{len(input_vectors)} cases simulated"
            )

    return results


def is_failure_target_satisfied(
    failure_target: str,
    legacy_result: Dict[str, object],
    evolved_result: Dict[str, object],
) -> bool:
    legacy_failed = str(legacy_result["label"]) == "Fail"
    evolved_failed = str(evolved_result["label"]) == "Fail"

    if failure_target == "legacy":
        return legacy_failed
    if failure_target == "evolved":
        return evolved_failed
    if failure_target == "either":
        return legacy_failed or evolved_failed
    if failure_target == "both":
        return legacy_failed and evolved_failed
    raise ValueError(f"Unsupported failure target: {failure_target}")


def generate_failure_focused_cases(
    system: SystemSpec,
    n_samples: int,
    sampling_seed: int,
    legacy_seed: int,
    evolved_seed: int,
    dt: float,
    total_time: float,
    kp_pos: float,
    kp_theta: float,
    cbf_alpha: float,
    control_limits: jnp.ndarray,
    failure_target: str,
    max_attempts: int,
    random_sample_percentage: float,
    print_every: int,
    cbf_controller_kind: str,
    disturbance_norm_bound: float,
    disturbance_norm: str,
    on_case_accepted: Optional[
        Callable[
            [
                Dict[str, object],
                Dict[str, object],
                List[Dict[str, object]],
                List[Dict[str, object]],
            ],
            None,
        ]
    ] = None,
) -> Tuple[np.ndarray, List[Dict[str, object]], List[Dict[str, object]], Dict[str, int]]:
    rng = np.random.default_rng(sampling_seed)
    accepted_vectors: List[np.ndarray] = []
    accepted_legacy: List[Dict[str, object]] = []
    accepted_evolved: List[Dict[str, object]] = []

    logical_rejections = 0
    non_failure_rejections = 0
    attempts = 0
    accepted_failure_target = 0
    accepted_random_bucket = 0

    required_random = int(round(n_samples * (random_sample_percentage / 100.0)))
    required_failure_target = n_samples - required_random

    print(
        f"[{system.name}] failure-focused quotas: "
        f"failure_target_cases={required_failure_target}, random_cases={required_random}"
    )
    loop_start = time.perf_counter()

    while len(accepted_vectors) < n_samples and attempts < max_attempts:
        attempt_idx = attempts + 1
        detailed_attempt_log = attempt_idx <= 3 or (attempt_idx % print_every == 0)
        vec = sample_single_input(rng, system)
        if not is_logical_initial_condition(vec):
            logical_rejections += 1
            attempts += 1
            if detailed_attempt_log:
                print(
                    f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                    "rejected logical initial condition"
                )
            continue

        if detailed_attempt_log:
            print(
                f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                "running legacy controller simulation"
            )
        legacy_result = run_single_simulation(
            system=system,
            vec=vec,
            use_cbf=False,
            dt=dt,
            total_time=total_time,
            sim_seed=legacy_seed + attempts,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
            progress_label=f"attempt {attempt_idx} legacy" if attempt_idx == 1 else None,
            progress_step_every=25 if attempt_idx == 1 else 0,
        )
        if detailed_attempt_log:
            print(
                f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                f"legacy done label={legacy_result['label']}, "
                f"runtime={float(legacy_result['case_runtime_s']):.2f}s"
            )
            print(
                f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                "running evolved controller simulation"
            )
        evolved_result = run_single_simulation(
            system=system,
            vec=vec,
            use_cbf=True,
            dt=dt,
            total_time=total_time,
            sim_seed=evolved_seed + attempts,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
            progress_label=f"attempt {attempt_idx} evolved" if attempt_idx == 1 else None,
            progress_step_every=25 if attempt_idx == 1 else 0,
        )
        if detailed_attempt_log:
            print(
                f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                f"evolved done label={evolved_result['label']}, "
                f"runtime={float(evolved_result['case_runtime_s']):.2f}s"
            )

        matches_failure_target = is_failure_target_satisfied(
            failure_target, legacy_result, evolved_result
        )

        accept_case = False
        if matches_failure_target and accepted_failure_target < required_failure_target:
            accepted_failure_target += 1
            accept_case = True
        elif accepted_random_bucket < required_random:
            accepted_random_bucket += 1
            accept_case = True

        if not accept_case:
            if not matches_failure_target and accepted_failure_target < required_failure_target:
                non_failure_rejections += 1
            if detailed_attempt_log:
                print(
                    f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                    f"rejected after simulation (matches_failure_target={matches_failure_target})"
                )
            attempts += 1
            continue

        case_id = len(accepted_vectors)
        legacy_result["case_id"] = case_id
        evolved_result["case_id"] = case_id
        accepted_vectors.append(vec)
        accepted_legacy.append(legacy_result)
        accepted_evolved.append(evolved_result)
        if on_case_accepted is not None:
            on_case_accepted(legacy_result, evolved_result, accepted_legacy, accepted_evolved)
        # Keep memory bounded in large runs: persist step logs externally, then drop them from RAM.
        accepted_legacy[-1].pop("step_logs", None)
        accepted_evolved[-1].pop("step_logs", None)
        attempts += 1
        if detailed_attempt_log:
            print(
                f"[{system.name}] attempt {attempt_idx}/{max_attempts}: "
                f"accepted as case_id={case_id} "
                f"(accepted={len(accepted_vectors)}/{n_samples}, "
                f"failure_target={accepted_failure_target}/{required_failure_target}, "
                f"random={accepted_random_bucket}/{required_random})"
            )

        if len(accepted_vectors) % TIME_LOG_EVERY_CASES == 0:
            elapsed = time.perf_counter() - loop_start
            avg = elapsed / float(len(accepted_vectors))
            print(
                f"[{system.name}] failure-focused timing: accepted={len(accepted_vectors)}/{n_samples}, "
                f"elapsed={elapsed:.2f}s, avg={avg:.3f}s/accepted-case"
            )

        if attempts % print_every == 0 or len(accepted_vectors) == n_samples:
            print(
                f"[{system.name}] failure-focused progress: accepted={len(accepted_vectors)}/{n_samples}, "
                f"attempts={attempts}/{max_attempts}, logical_rejections={logical_rejections}, "
                f"non_failure_rejections={non_failure_rejections}, "
                f"accepted_failure_target={accepted_failure_target}/{required_failure_target}, "
                f"accepted_random={accepted_random_bucket}/{required_random}"
            )

    if len(accepted_vectors) < n_samples:
        raise RuntimeError(
            "Could not generate enough failure-focused cases. "
            f"accepted={len(accepted_vectors)}, requested={n_samples}, attempts={attempts}, "
            f"logical_rejections={logical_rejections}, non_failure_rejections={non_failure_rejections}, "
            "consider increasing --failure-focus-max-attempts or relaxing --failure-target."
        )

    return (
        np.vstack(accepted_vectors),
        accepted_legacy,
        accepted_evolved,
        {
            "attempts": attempts,
            "logical_rejections": logical_rejections,
            "non_failure_rejections": non_failure_rejections,
            "required_failure_target_cases": required_failure_target,
            "required_random_cases": required_random,
            "accepted_failure_target_cases": accepted_failure_target,
            "accepted_random_cases": accepted_random_bucket,
            "random_sample_percentage": random_sample_percentage,
        },
    )


def write_dataset_csv(
    out_csv: Path,
    system: SystemSpec,
    rows: Sequence[Dict[str, object]],
) -> None:
    names = feature_names(system)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                *names,
                "label",
                "min_distance",
                "min_dx",
                "min_dy",
                "controller_error",
                "case_runtime_s",
            ]
        )
        for row in rows:
            vec = row["input_vector"]
            writer.writerow(
                [
                    row["case_id"],
                    *vec,
                    row["label"],
                    row["min_distance"],
                    row["min_dx"],
                    row["min_dy"],
                    row["controller_error"],
                    row["case_runtime_s"],
                ]
            )


def write_step_logs_csv(
    out_csv: Path,
    rows: Sequence[Dict[str, object]],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "case_id",
        "label",
        "controller_error",
        "step",
        "time_s",
        "wall_time_cumulative_s",
        "state_x",
        "state_y",
        "state_v",
        "state_theta",
        "dx",
        "dy",
        "distance",
        "controller_stats_json",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            case_id = row["case_id"]
            label = row["label"]
            controller_error = row["controller_error"]
            for step in row.get("step_logs", []):
                writer.writerow(
                    {
                        "case_id": case_id,
                        "label": label,
                        "controller_error": controller_error,
                        **step,
                    }
                )


def append_step_logs_csv(
    out_csv: Path,
    case_row: Dict[str, object],
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "case_id",
        "label",
        "controller_error",
        "step",
        "time_s",
        "wall_time_cumulative_s",
        "state_x",
        "state_y",
        "state_v",
        "state_theta",
        "dx",
        "dy",
        "distance",
        "controller_stats_json",
    ]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for step in case_row.get("step_logs", []):
            writer.writerow(
                {
                    "case_id": case_row["case_id"],
                    "label": case_row["label"],
                    "controller_error": case_row["controller_error"],
                    **step,
                }
            )


def write_paired_comparison_csv(
    out_csv: Path,
    system: SystemSpec,
    legacy_rows: Sequence[Dict[str, object]],
    evolved_rows: Sequence[Dict[str, object]],
) -> None:
    names = feature_names(system)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(legacy_rows) != len(evolved_rows):
        raise ValueError("Paired comparison requires equal row counts for legacy and evolved runs.")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                *names,
                "legacy_label",
                "legacy_min_distance",
                "legacy_min_dx",
                "legacy_min_dy",
                "legacy_controller_error",
                "legacy_case_runtime_s",
                "evolved_label",
                "evolved_min_distance",
                "evolved_min_dx",
                "evolved_min_dy",
                "evolved_controller_error",
                "evolved_case_runtime_s",
            ]
        )

        for leg, evo in zip(legacy_rows, evolved_rows):
            if leg["case_id"] != evo["case_id"]:
                raise ValueError("Mismatched case_id between legacy and evolved paired rows.")
            writer.writerow(
                [
                    leg["case_id"],
                    *leg["input_vector"],
                    leg["label"],
                    leg["min_distance"],
                    leg["min_dx"],
                    leg["min_dy"],
                    leg["controller_error"],
                    leg["case_runtime_s"],
                    evo["label"],
                    evo["min_distance"],
                    evo["min_dx"],
                    evo["min_dy"],
                    evo["controller_error"],
                    evo["case_runtime_s"],
                ]
            )


def summarize_labels(rows: Sequence[Dict[str, object]]) -> Dict[str, int]:
    counts = {"Pass": 0, "Fail": 0}
    for r in rows:
        counts[str(r["label"])] += 1
    return counts


def generate_for_system(
    system: SystemSpec,
    out_root: Path,
    n_samples: int,
    sampling_seed: int,
    legacy_seed: int,
    evolved_seed: int,
    dt: float,
    total_time: float,
    kp_pos: float,
    kp_theta: float,
    cbf_alpha: float,
    control_limits: jnp.ndarray,
    paired_index_datasets: bool,
    failure_focused: bool,
    failure_target: str,
    failure_focus_max_attempts: int,
    random_sample_percentage: float,
    print_every: int,
    checkpoint_every: int,
    cbf_controller_kind: str,
    disturbance_norm_bound: float,
    disturbance_norm: str,
    seed_offset: int = 0,
    batch_index: Optional[int] = None,
    batch_seed_stride: int = 0,
) -> Dict[str, object]:
    system_start = time.perf_counter()
    print(
        f"Starting generation for {system.name}: n_samples={n_samples}, "
        f"mode={'failure-focused' if failure_focused else 'standard'}"
    )
    focus_stats = {
        "attempts": n_samples,
        "logical_rejections": 0,
        "non_failure_rejections": 0,
    }
    sys_dir = out_root / system.name
    sys_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = sys_dir / "metadata.json"
    legacy_step_log_path = sys_dir / "D_legacy_step_logs.csv"
    evolved_step_log_path = sys_dir / "D_evolved_step_logs.csv"
    if legacy_step_log_path.exists():
        legacy_step_log_path.unlink()
    if evolved_step_log_path.exists():
        evolved_step_log_path.unlink()
    checkpoint_counter = 0

    def checkpoint_write(
        legacy_rows: Sequence[Dict[str, object]],
        evolved_all_rows: Sequence[Dict[str, object]],
        *,
        force: bool = False,
    ) -> None:
        nonlocal checkpoint_counter
        if not force and (len(legacy_rows) == 0 or len(legacy_rows) % checkpoint_every != 0):
            return

        checkpoint_counter += 1
        d_evolved_partial = [row for row in evolved_all_rows if row["label"] == "Pass"]
        legacy_csv_path = sys_dir / "D_legacy.csv"
        evolved_csv_path = sys_dir / "D_evolved.csv"
        paired_csv_path = sys_dir / "D_paired_comparison.csv"
        write_dataset_csv(legacy_csv_path, system, legacy_rows)
        write_dataset_csv(evolved_csv_path, system, d_evolved_partial)
        if paired_index_datasets and len(legacy_rows) == len(evolved_all_rows):
            write_paired_comparison_csv(
                paired_csv_path,
                system,
                legacy_rows,
                evolved_all_rows,
            )
            paired_status = f"wrote paired={paired_csv_path}"
        else:
            paired_status = (
                "skipped paired "
                f"(paired_index_datasets={paired_index_datasets}, "
                f"legacy_rows={len(legacy_rows)}, evolved_all_rows={len(evolved_all_rows)})"
            )
            print(
                f"[{system.name}] checkpoint {checkpoint_counter}: "
                f"WARNING: skipped paired comparison (row counts: legacy={len(legacy_rows)}, evolved_all={len(evolved_all_rows)}), paired_index_datasets={paired_index_datasets}"
            )
        print(
            f"[{system.name}] checkpoint {checkpoint_counter}: "
            f"saved {len(legacy_rows)} legacy, {len(evolved_all_rows)} evolved-all cases; "
            f"legacy_csv={legacy_csv_path}, evolved_csv={evolved_csv_path} "
            f"(evolved_pass_rows={len(d_evolved_partial)}), {paired_status}"
        )

    def write_metadata_snapshot(
        *,
        generation_status: str,
        d_legacy_rows: Optional[Sequence[Dict[str, object]]] = None,
        d_evolved_all_rows: Optional[Sequence[Dict[str, object]]] = None,
        d_evolved_rows: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Dict[str, object]:
        legacy_rows = list(d_legacy_rows) if d_legacy_rows is not None else []
        evolved_all_rows = list(d_evolved_all_rows) if d_evolved_all_rows is not None else []
        evolved_rows = list(d_evolved_rows) if d_evolved_rows is not None else []

        metadata_obj = {
            "system": system.name,
            "generation_status": generation_status,
            "input_features": [
                {
                    "name": f.name,
                    "meaning": f.meaning,
                    "units": f.units,
                    "range": [f.bounds[0], f.bounds[1]],
                }
                for f in system.features
            ],
            "sampling_strategy": system.sampling_strategy,
            "generation_mode": {
                "failure_focused": failure_focused,
                "failure_target": failure_target if failure_focused else "none",
                "random_sample_percentage": random_sample_percentage if failure_focused else 100.0,
                "required_initial_clearance_m": INITIAL_OBSTACLE_CLEARANCE_M,
                "focus_attempt_stats": focus_stats,
            },
            "controller_parameters": {
                "legacy": {
                    "type": "proportional_controller",
                    "Kp_pos": kp_pos,
                    "Kp_theta": kp_theta,
                    "goal_state": [4.0, 0.0, 0.0, 0.0],
                    "control_limits": control_limits.tolist(),
                },
                "evolved": {
                    "type": (
                        "robust_cbf_qp_controller"
                        if cbf_controller_kind == "robust"
                        else "vanilla_cbf_qp_controller"
                    ),
                    "cbf_controller_kind": cbf_controller_kind,
                    "Kp_pos": kp_pos,
                    "Kp_theta": kp_theta,
                    "cbf_alpha_linear_class_k": cbf_alpha,
                    "cbf_velocity_weight": CBF_VELOCITY_WEIGHT,
                    "disturbance_norm_bound": disturbance_norm_bound,
                    "disturbance_norm": disturbance_norm,
                    "barrier": (
                        "h_cbf(x,t)=||p-p_obs(t)||^2-r^2 + k_v*v*<p-p_obs(t),[cos(theta),sin(theta)]>; "
                        "safety label uses h_safe=||p-p_obs(t)||^2-r^2"
                    ),
                    "control_limits": control_limits.tolist(),
                },
            },
            "simulation": {
                "time_step_s": dt,
                "total_duration_s": total_time,
                "num_steps": int(total_time / dt),
            },
            "seeds": {
                "sampling_seed": sampling_seed,
                "legacy_base_seed": legacy_seed,
                "evolved_base_seed": evolved_seed,
                "seed_offset": seed_offset,
                "batch_index": batch_index,
                "batch_seed_stride": batch_seed_stride,
            },
            "dataset_sizes": {
                "D_legacy": len(legacy_rows),
                "D_evolved": len(evolved_rows),
                "D_paired_comparison": len(evolved_all_rows) if paired_index_datasets else 0,
            },
            "label_counts": {
                "D_legacy": summarize_labels(legacy_rows),
                "D_evolved_before_filter": summarize_labels(evolved_all_rows),
                "D_evolved_after_filter": summarize_labels(evolved_rows),
                "D_paired_legacy": summarize_labels(legacy_rows) if paired_index_datasets else {},
                "D_paired_evolved": summarize_labels(evolved_all_rows) if paired_index_datasets else {},
            },
            "index_alignment": {
                "case_id_column": True,
                "paired_index_datasets_enabled": paired_index_datasets,
                "paired_note": (
                    "Use D_paired_comparison.csv for one-to-one test cases with both controller labels."
                    if paired_index_datasets
                    else "Enable --paired-index-datasets to export one-to-one comparison rows."
                ),
            },
        }

        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata_obj, f, indent=2)
        return metadata_obj

    write_metadata_snapshot(generation_status="in_progress")
    print(f"[{system.name}] wrote initial metadata: {metadata_path}")

    if failure_focused:
        inputs, d_legacy, d_evolved_all, focus_stats = generate_failure_focused_cases(
            system=system,
            n_samples=n_samples,
            sampling_seed=sampling_seed,
            legacy_seed=legacy_seed,
            evolved_seed=evolved_seed,
            dt=dt,
            total_time=total_time,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            failure_target=failure_target,
            max_attempts=failure_focus_max_attempts,
            random_sample_percentage=random_sample_percentage,
            print_every=print_every,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
            on_case_accepted=lambda leg_case, evo_case, leg_rows, evo_rows: (
                append_step_logs_csv(legacy_step_log_path, leg_case),
                append_step_logs_csv(evolved_step_log_path, evo_case),
                checkpoint_write(leg_rows, evo_rows, force=True),
            ),
        )
    else:
        inputs = sample_inputs(system, n_samples=n_samples, seed=sampling_seed)
        print(f"[{system.name}] sampled {len(inputs)} logical initial conditions")

        d_legacy = run_batch_simulation(
            system=system,
            input_vectors=inputs,
            use_cbf=False,
            dt=dt,
            total_time=total_time,
            base_seed=legacy_seed,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            print_every=print_every,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
            on_case_complete=lambda _idx, result, rows: (
                append_step_logs_csv(legacy_step_log_path, result),
                checkpoint_write(rows, [], force=True),
            ),
        )

        d_evolved_all = run_batch_simulation(
            system=system,
            input_vectors=inputs,
            use_cbf=True,
            dt=dt,
            total_time=total_time,
            base_seed=evolved_seed,
            kp_pos=kp_pos,
            kp_theta=kp_theta,
            cbf_alpha=cbf_alpha,
            control_limits=control_limits,
            print_every=print_every,
            cbf_controller_kind=cbf_controller_kind,
            disturbance_norm_bound=disturbance_norm_bound,
            disturbance_norm=disturbance_norm,
            on_case_complete=lambda _idx, result, rows: (
                append_step_logs_csv(evolved_step_log_path, result),
                checkpoint_write(d_legacy, rows, force=True),
            ),
        )
    d_evolved = [row for row in d_evolved_all if row["label"] == "Pass"]

    checkpoint_write(d_legacy, d_evolved_all, force=True)
    write_dataset_csv(sys_dir / "D_legacy.csv", system, d_legacy)
    write_dataset_csv(sys_dir / "D_evolved.csv", system, d_evolved)
    if paired_index_datasets:
        stale_path = sys_dir / "D_evolved_paired.csv"
        if stale_path.exists():
            stale_path.unlink()
        write_paired_comparison_csv(
            sys_dir / "D_paired_comparison.csv",
            system,
            d_legacy,
            d_evolved_all,
        )

    metadata = write_metadata_snapshot(
        generation_status="complete",
        d_legacy_rows=d_legacy,
        d_evolved_all_rows=d_evolved_all,
        d_evolved_rows=d_evolved,
    )

    print(
        f"Completed {system.name}: D_legacy={len(d_legacy)}, "
        f"D_evolved={len(d_evolved)}, output_dir={sys_dir}, "
        f"elapsed={time.perf_counter() - system_start:.2f}s"
    )

    return metadata


def main() -> None:
    total_start = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument(
        "--safe-runtime",
        action="store_true",
        help=(
            "Relaunch with conservative CPU settings (single-threaded BLAS/XLA, CPU-only JAX, "
            "JIT disabled) to reduce intermittent native runtime crashes."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="examples/unicycle/start_to_goal/results/operational_rule_datasets",
    )
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=15.0)
    parser.add_argument("--sampling-seed", type=int, default=20260212)
    parser.add_argument("--legacy-seed", type=int, default=30100)
    parser.add_argument("--evolved-seed", type=int, default=40100)
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help=(
            "Offset added to sampling, legacy, and evolved seeds. Use this when running "
            "multiple independent dataset batches."
        ),
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=None,
        help=(
            "Optional batch index used to derive a seed offset. Effective offset is "
            "--seed-offset + --batch-index * --batch-seed-stride."
        ),
    )
    parser.add_argument(
        "--batch-seed-stride",
        type=int,
        default=100000,
        help="Seed stride used with --batch-index to keep batch sampling streams distinct.",
    )
    parser.add_argument("--kp-pos", type=float, default=1.2)
    parser.add_argument("--kp-theta", type=float, default=1.0)
    parser.add_argument("--cbf-alpha", type=float, default=2.0)
    parser.add_argument("--a-limit", type=float, default=3.0)
    parser.add_argument("--omega-limit", type=float, default=2.5)
    parser.add_argument(
        "--paired-index-datasets",
        action="store_true",
        help="Also write D_paired_comparison.csv with one row per case and both controller labels.",
    )
    parser.add_argument(
        "--failure-focused",
        action="store_true",
        help=(
            "Focus sampling on failure-inducing cases while enforcing logical starts "
            "(outside obstacle with safety margin)."
        ),
    )
    parser.add_argument(
        "--failure-target",
        type=str,
        choices=["legacy", "evolved", "either", "both"],
        default="legacy",
        help=(
            "When --failure-focused is enabled, define which controller failure pattern "
            "must hold for a case to be accepted."
        ),
    )
    parser.add_argument(
        "--failure-focus-max-attempts",
        type=int,
        default=20000,
        help="Maximum candidate attempts per system for --failure-focused mode.",
    )
    parser.add_argument(
        "--random-sample-percentage",
        type=float,
        default=0.0,
        help=(
            "In --failure-focused mode, percentage of accepted cases reserved for random logical "
            "samples (not required to satisfy --failure-target)."
        ),
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=25,
        help="Print terminal progress every N cases/attempts.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Persist partial CSV outputs every N accepted/simulated cases.",
    )
    parser.add_argument(
        "--cbf-controller",
        type=str,
        choices=["vanilla", "robust"],
        default="vanilla",
        help="CBF-QP controller used for evolved runs.",
    )
    parser.add_argument(
        "--disturbance-norm-bound",
        type=float,
        default=0.1,
        help="Required for --cbf-controller robust: upper bound on disturbance norm.",
    )
    parser.add_argument(
        "--disturbance-norm",
        type=str,
        choices=["2", "inf"],
        default="2",
        help="Norm type used by robust controller disturbance term.",
    )
    args = parser.parse_args()
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")
    if args.random_sample_percentage < 0.0 or args.random_sample_percentage > 100.0:
        raise ValueError("--random-sample-percentage must be in [0, 100]")
    if args.cbf_controller == "robust" and args.disturbance_norm_bound <= 0.0:
        raise ValueError("--disturbance-norm-bound must be > 0 when --cbf-controller robust")
    if args.batch_seed_stride <= 0:
        raise ValueError("--batch-seed-stride must be > 0")
    if args.batch_index is not None and args.batch_index < 0:
        raise ValueError("--batch-index must be >= 0")

    out_root = Path(args.out_dir)
    control_limits = jnp.array([args.a_limit, args.omega_limit])
    batch_seed_offset = 0
    if args.batch_index is not None:
        batch_seed_offset = args.batch_index * args.batch_seed_stride
    effective_seed_offset = args.seed_offset + batch_seed_offset
    effective_sampling_seed = args.sampling_seed + effective_seed_offset
    effective_legacy_seed = args.legacy_seed + effective_seed_offset
    effective_evolved_seed = args.evolved_seed + effective_seed_offset
    if min(effective_sampling_seed, effective_legacy_seed, effective_evolved_seed) < 0:
        raise ValueError(
            "Effective seeds must be >= 0. Adjust --seed-offset, --batch-index, or base seeds."
        )

    print(
        "Seed configuration: "
        f"base_sampling={args.sampling_seed}, base_legacy={args.legacy_seed}, "
        f"base_evolved={args.evolved_seed}, seed_offset={args.seed_offset}, "
        f"batch_index={args.batch_index}, batch_seed_stride={args.batch_seed_stride}, "
        f"effective_offset={effective_seed_offset}"
    )

    summaries = {}
    for i, system in enumerate([DYNAMIC_SYSTEM, STATIC_SYSTEM]):  ## STATIC_SYSTEM,
        summaries[system.name] = generate_for_system(
            system=system,
            out_root=out_root,
            n_samples=args.n_samples,
            sampling_seed=effective_sampling_seed + 1000 * i,
            legacy_seed=effective_legacy_seed + 1000 * i,
            evolved_seed=effective_evolved_seed + 1000 * i,
            dt=args.dt,
            total_time=args.tf,
            kp_pos=args.kp_pos,
            kp_theta=args.kp_theta,
            cbf_alpha=args.cbf_alpha,
            control_limits=control_limits,
            paired_index_datasets=args.paired_index_datasets,
            failure_focused=args.failure_focused,
            failure_target=args.failure_target,
            failure_focus_max_attempts=args.failure_focus_max_attempts,
            random_sample_percentage=args.random_sample_percentage,
            print_every=args.print_every,
            checkpoint_every=args.checkpoint_every,
            cbf_controller_kind=args.cbf_controller,
            disturbance_norm_bound=args.disturbance_norm_bound,
            disturbance_norm=args.disturbance_norm,
            seed_offset=effective_seed_offset,
            batch_index=args.batch_index,
            batch_seed_stride=args.batch_seed_stride,
        )

    summary_path = out_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Wrote datasets to: {out_root}")
    for system_name, md in summaries.items():
        legacy_counts = md["label_counts"]["D_legacy"]
        evolved_counts = md["label_counts"]["D_evolved_after_filter"]
        print(
            f"{system_name}: D_legacy Pass={legacy_counts['Pass']} Fail={legacy_counts['Fail']} | "
            f"D_evolved Pass={evolved_counts['Pass']} Fail={evolved_counts['Fail']}"
        )
    print(f"Total generation time: {time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()

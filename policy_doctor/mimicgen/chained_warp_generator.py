"""Constraint-aware MimicGen DataGenerator with iterative IC resampling.

The standard `mimicgen.datagen.data_generator.DataGenerator` runs the entire
trajectory in one pass — sampling a fresh initial condition (object pose),
then warping each subtask of the seed demo to align with the resulting state.
There's no notion of "force the object to land within a window of pose T
after subtask N."

This module adds that. The mechanism is intentionally simple:

  1. We subclass DataGenerator and override generate() to insert a constraint
     check between subtasks.
  2. After subtask N completes (where N == ``constraint.subtask_idx``), we
     read the current object pose. If it's within the configured slack box
     of ``constraint.target_pose``, generation continues — and MimicGen's
     own per-subtask warp absorbs the (constrained) achieved state as the
     reference for subtasks N+1..M-1. That's the "chained warp" effect.
  3. If the achieved pose is outside the slack box, we mark the trial as
     a constraint-failure and return early. The outer trial-budget loop
     (in MimicGen's ``generate_dataset`` or in our own script wrapper) then
     resamples a fresh IC and retries.

This is strictly cheaper than the previous reject-after-full-trajectory
implementation, and the slack box is meant to be sized from data
(e.g. ``slack = α × stddev`` over the failure-trajectory cluster), giving
us a meaningful tolerance instead of one we have to guess.

See `docs/experiment_log_failure_targeting_may11.md` for the design context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constraint specification
# ---------------------------------------------------------------------------

@dataclass
class IntermediateConstraint:
    """A "the object must be near pose T after subtask N completes" target.

    Attributes:
        subtask_idx:   0-based subtask index. The check happens after the
                       subtask at this index *completes*, i.e. at the start
                       of subtask ``subtask_idx + 1``.
        target_pose:   Per-object target ``{obj_name: {"x": float, "y": float,
                       "z_rot": float}}``. World-frame absolute.
        slack:         Per-object slack box ``{obj_name: {"x": float,
                       "y": float, "z_rot": float}}``. ± offsets from target.
        slack_widen_factor: Multiplier applied to slack when the outer caller
                       retries after enough rejections (see
                       ``ChainedWarpDataGenerator.last_outcome``).
        objects:       Subset of objects to actually constrain. None ⇒ all.

    Both target_pose and slack are in world coordinates. The check is:
        |achieved.x - target.x| ≤ slack.x   (and same for y / z_rot)

    z_rot is wrapped to [-pi, pi] before comparison so that 3.13 and -3.13
    are treated as ~0.02 apart, not ~6.26.
    """

    subtask_idx: int
    target_pose: dict[str, dict[str, float]]
    slack: dict[str, dict[str, float]]
    slack_widen_factor: float = 2.0
    objects: Optional[list[str]] = None

    def widen(self, factor: float) -> "IntermediateConstraint":
        """Return a copy with slack multiplied by ``factor``."""
        widened: dict[str, dict[str, float]] = {
            obj: {axis: v * factor for axis, v in axes.items()}
            for obj, axes in self.slack.items()
        }
        return IntermediateConstraint(
            subtask_idx=self.subtask_idx,
            target_pose=self.target_pose,
            slack=widened,
            slack_widen_factor=self.slack_widen_factor,
            objects=self.objects,
        )

    def is_satisfied(self, achieved_pose: dict[str, dict[str, float]]) -> tuple[bool, dict[str, float]]:
        """Check whether ``achieved_pose`` lies within this constraint's slack box.

        Args:
            achieved_pose: ``{obj_name: {"x", "y", "z_rot"}}`` from the sim
                after the constrained subtask completed.

        Returns:
            ``(satisfied, distances)``. ``distances`` is the worst-case axis
            distance per object, useful for telemetry / debugging.
        """
        check_objs = self.objects or list(self.target_pose.keys())
        worst: dict[str, float] = {}
        for obj in check_objs:
            if obj not in self.target_pose or obj not in self.slack:
                continue
            target = self.target_pose[obj]
            slack = self.slack[obj]
            achieved = achieved_pose.get(obj)
            if achieved is None:
                # Object not in observation — be conservative and fail.
                worst[obj] = float("inf")
                return False, worst
            dx = abs(achieved["x"] - target["x"])
            dy = abs(achieved["y"] - target["y"])
            dth = _wrap_angle(achieved["z_rot"] - target["z_rot"])
            # Normalise each axis by its slack; the worst normalised axis
            # tells us how close to the boundary we are.
            sx = slack.get("x", 0.0)
            sy = slack.get("y", 0.0)
            sz = slack.get("z_rot", 0.0)
            ratios = []
            if sx > 0: ratios.append(dx / sx)
            if sy > 0: ratios.append(dy / sy)
            if sz > 0: ratios.append(abs(dth) / sz)
            if not ratios:
                continue
            worst[obj] = float(max(ratios))
            if worst[obj] > 1.0:
                return False, worst
        return True, worst


def _wrap_angle(theta: float) -> float:
    """Wrap to [-pi, pi]. Pure float math so the dataclass stays dependency-thin."""
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


# ---------------------------------------------------------------------------
# Outcome record (for the outer retry loop)
# ---------------------------------------------------------------------------

@dataclass
class GenerationOutcome:
    """Reason MimicGen's last ``generate()`` returned what it did."""

    task_success: bool                # MimicGen's own success criterion
    constraint_met: Optional[bool]    # our intermediate constraint check
    failure_reason: Optional[str]     # short string for telemetry
    worst_axis_ratio: float = 0.0     # max |achieved - target| / slack across axes
    subtasks_executed: int = 0
    distances: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constraint-aware generator
# ---------------------------------------------------------------------------

# Imported lazily so the module can be loaded outside the mimicgen env (e.g.
# for unit tests of IntermediateConstraint).
def _get_data_generator_base():
    from mimicgen.datagen.data_generator import DataGenerator
    return DataGenerator


def make_chained_warp_generator_class():
    """Build the ChainedWarpDataGenerator subclass at call time.

    Done lazily because MimicGen and robosuite imports are expensive and not
    available in the policy_doctor env. Test code that doesn't need a real
    sim should import IntermediateConstraint / GenerationOutcome directly.
    """

    DataGenerator = _get_data_generator_base()

    class ChainedWarpDataGenerator(DataGenerator):
        """DataGenerator that early-aborts when the post-subtask-N pose
        misses the configured intermediate target.

        See module docstring for the design rationale. Usage:

            constraint = IntermediateConstraint(
                subtask_idx=0,
                target_pose={"nut": {"x": 0.10, "y": 0.20, "z_rot": 0.0}},
                slack={"nut": {"x": 0.03, "y": 0.03, "z_rot": 0.5}},
            )
            gen = ChainedWarpDataGenerator(..., constraint=constraint)
            result = gen.generate(env, env_interface)
            if not gen.last_outcome.constraint_met:
                # outer code resamples IC₀ and retries
                ...

        When ``constraint`` is None, this class behaves exactly like the
        standard DataGenerator.
        """

        def __init__(self, *args, constraint: Optional[IntermediateConstraint] = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.constraint = constraint
            self.last_outcome: Optional[GenerationOutcome] = None

        # The body of generate() below is a verbatim copy of the upstream
        # implementation in mimicgen/datagen/data_generator.py, with two
        # additions: (a) a constraint check after subtask `subtask_idx`
        # completes, and (b) recording of `self.last_outcome` for the outer
        # retry loop. Anything else marked "# CW:" is from us.
        def generate(
            self,
            env,
            env_interface,
            select_src_per_subtask: bool = False,
            transform_first_robot_pose: bool = False,
            interpolate_from_last_target_pose: bool = True,
            render: bool = False,
            video_writer=None,
            video_skip: int = 5,
            camera_names=None,
            pause_subtask: bool = False,
        ):
            from mimicgen.utils import pose_utils as PoseUtils
            from mimicgen.datagen.waypoint import WaypointSequence, WaypointTrajectory

            self.last_outcome = None  # CW: reset
            constraint = self.constraint

            env.reset()
            new_initial_state = env.get_state()

            all_subtask_inds = self.randomize_subtask_boundaries()

            selected_src_demo_ind = None
            prev_executed_traj = None

            generated_states = []
            generated_obs = []
            generated_datagen_infos = []
            generated_actions = []
            generated_success = False
            generated_src_demo_inds = []
            generated_src_demo_labels = []

            # CW: track which subtask we just finished so we can check the constraint
            # in the *next* iteration's get_datagen_info() call (which gives us the
            # post-execution object poses).
            constraint_met_so_far: Optional[bool] = None
            constraint_worst_ratio: float = 0.0
            constraint_distances: dict[str, float] = {}

            for subtask_ind in range(len(self.task_spec)):
                is_first_subtask = (subtask_ind == 0)

                cur_datagen_info = env_interface.get_datagen_info()

                # CW: BEFORE we set up the next subtask, check if the previous one
                # was the constraint subtask. If so, evaluate the constraint here.
                if constraint is not None and subtask_ind == constraint.subtask_idx + 1:
                    achieved = _datagen_info_to_xy_yaw(cur_datagen_info)
                    satisfied, worst = constraint.is_satisfied(achieved)
                    constraint_met_so_far = satisfied
                    constraint_distances = worst
                    constraint_worst_ratio = max(worst.values()) if worst else 0.0
                    # Optional per-trial debug print (set CW_DEBUG=1).
                    import os as _os
                    if _os.environ.get("CW_DEBUG"):
                        for _obj, _t in constraint.target_pose.items():
                            _a = achieved.get(_obj, {})
                            print(
                                f"[CW] subtask {subtask_ind - 1} done | obj={_obj} | "
                                f"achieved=({_a.get('x', 0):.4f}, {_a.get('y', 0):.4f}, "
                                f"{_a.get('z_rot', 0):+.3f}) | "
                                f"target=({_t['x']:.4f}, {_t['y']:.4f}, {_t['z_rot']:+.3f}) | "
                                f"worst_ratio={constraint_worst_ratio:.3f} | "
                                f"satisfied={satisfied}",
                                flush=True,
                            )
                    if not satisfied:
                        # Early-abort: skip executing subtasks N+1..M-1 entirely.
                        self.last_outcome = GenerationOutcome(
                            task_success=False,
                            constraint_met=False,
                            failure_reason="intermediate_constraint_violated",
                            worst_axis_ratio=constraint_worst_ratio,
                            subtasks_executed=subtask_ind,
                            distances=constraint_distances,
                        )
                        return _empty_result(new_initial_state)

                subtask_object_name = self.task_spec[subtask_ind]["object_ref"]
                cur_object_pose = (
                    cur_datagen_info.object_poses[subtask_object_name]
                    if (subtask_object_name is not None) else None
                )

                need_source_demo_selection = (is_first_subtask or select_src_per_subtask)
                if need_source_demo_selection:
                    selected_src_demo_ind = self.select_source_demo(
                        eef_pose=cur_datagen_info.eef_pose,
                        object_pose=cur_object_pose,
                        subtask_ind=subtask_ind,
                        src_subtask_inds=all_subtask_inds[:, subtask_ind],
                        subtask_object_name=subtask_object_name,
                        selection_strategy_name=self.task_spec[subtask_ind]["selection_strategy"],
                        selection_strategy_kwargs=self.task_spec[subtask_ind]["selection_strategy_kwargs"],
                    )
                assert selected_src_demo_ind is not None
                selected_src_subtask_inds = all_subtask_inds[selected_src_demo_ind, subtask_ind]

                src_ep_datagen_info = self.src_dataset_infos[selected_src_demo_ind]
                src_subtask_eef_poses = src_ep_datagen_info.eef_pose[
                    selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
                ]
                src_subtask_target_poses = src_ep_datagen_info.target_pose[
                    selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
                ]
                src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[
                    selected_src_subtask_inds[0] : selected_src_subtask_inds[1]
                ]
                src_subtask_object_pose = (
                    src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_inds[0]]
                    if (subtask_object_name is not None) else None
                )

                if is_first_subtask or transform_first_robot_pose:
                    src_eef_poses = np.concatenate(
                        [src_subtask_eef_poses[0:1], src_subtask_target_poses], axis=0
                    )
                else:
                    src_eef_poses = np.array(src_subtask_target_poses)

                src_subtask_gripper_actions = np.concatenate(
                    [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], axis=0
                )

                if subtask_object_name is not None:
                    transformed_eef_poses = PoseUtils.transform_source_data_segment_using_object_pose(
                        obj_pose=cur_object_pose,
                        src_eef_poses=src_eef_poses,
                        src_obj_pose=src_subtask_object_pose,
                    )
                else:
                    transformed_eef_poses = src_eef_poses

                traj_to_execute = WaypointTrajectory()

                if interpolate_from_last_target_pose and (not is_first_subtask):
                    assert prev_executed_traj is not None
                    last_waypoint = prev_executed_traj.last_waypoint
                    init_sequence = WaypointSequence(sequence=[last_waypoint])
                else:
                    init_sequence = WaypointSequence.from_poses(
                        poses=cur_datagen_info.eef_pose[None],
                        gripper_actions=src_subtask_gripper_actions[0:1],
                        action_noise=self.task_spec[subtask_ind]["action_noise"],
                    )
                traj_to_execute.add_waypoint_sequence(init_sequence)

                transformed_seq = WaypointSequence.from_poses(
                    poses=transformed_eef_poses,
                    gripper_actions=src_subtask_gripper_actions,
                    action_noise=self.task_spec[subtask_ind]["action_noise"],
                )
                transformed_traj = WaypointTrajectory()
                transformed_traj.add_waypoint_sequence(transformed_seq)

                traj_to_execute.merge(
                    transformed_traj,
                    num_steps_interp=self.task_spec[subtask_ind]["num_interpolation_steps"],
                    num_steps_fixed=self.task_spec[subtask_ind]["num_fixed_steps"],
                    action_noise=(
                        float(self.task_spec[subtask_ind]["apply_noise_during_interpolation"])
                        * self.task_spec[subtask_ind]["action_noise"]
                    ),
                )

                traj_to_execute.pop_first()

                exec_results = traj_to_execute.execute(
                    env=env,
                    env_interface=env_interface,
                    render=render,
                    video_writer=video_writer,
                    video_skip=video_skip,
                    camera_names=camera_names,
                )

                if len(exec_results["states"]) > 0:
                    generated_states += exec_results["states"]
                    generated_obs += exec_results["observations"]
                    generated_datagen_infos += exec_results["datagen_infos"]
                    generated_actions.append(exec_results["actions"])
                    generated_success = generated_success or exec_results["success"]
                    generated_src_demo_inds.append(selected_src_demo_ind)
                    generated_src_demo_labels.append(
                        selected_src_demo_ind * np.ones(
                            (exec_results["actions"].shape[0], 1), dtype=int
                        )
                    )

                prev_executed_traj = traj_to_execute

                if pause_subtask:
                    input(
                        "Pausing after subtask {} execution. Press any key to continue...".format(
                            subtask_ind
                        )
                    )

            # CW: after the final subtask, re-evaluate the constraint if it was
            # the LAST subtask (so we never entered the next iteration's check).
            if (
                constraint is not None
                and constraint_met_so_far is None
                and constraint.subtask_idx == len(self.task_spec) - 1
            ):
                final_info = env_interface.get_datagen_info()
                achieved = _datagen_info_to_xy_yaw(final_info)
                constraint_met_so_far, worst = constraint.is_satisfied(achieved)
                constraint_distances = worst
                constraint_worst_ratio = max(worst.values()) if worst else 0.0

            if len(generated_actions) > 0:
                generated_actions = np.concatenate(generated_actions, axis=0)
                generated_src_demo_labels = np.concatenate(generated_src_demo_labels, axis=0)

            self.last_outcome = GenerationOutcome(
                task_success=generated_success,
                constraint_met=constraint_met_so_far,
                failure_reason=(
                    None if generated_success else "task_failed"
                ),
                worst_axis_ratio=constraint_worst_ratio,
                subtasks_executed=len(self.task_spec),
                distances=constraint_distances,
            )

            return dict(
                initial_state=new_initial_state,
                states=generated_states,
                observations=generated_obs,
                datagen_infos=generated_datagen_infos,
                actions=generated_actions,
                success=generated_success,
                src_demo_inds=generated_src_demo_inds,
                src_demo_labels=generated_src_demo_labels,
            )

    return ChainedWarpDataGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _datagen_info_to_xy_yaw(datagen_info) -> dict[str, dict[str, float]]:
    """Pull ``{obj: {x, y, z_rot}}`` from a MimicGen datagen_info.

    object_poses is a dict mapping ``obj_name`` → 4x4 pose matrix in world frame.
    z_rot is extracted from the rotation part as yaw (atan2(R[1,0], R[0,0])).
    """
    out: dict[str, dict[str, float]] = {}
    obj_poses = getattr(datagen_info, "object_poses", {}) or {}
    items = obj_poses.items() if hasattr(obj_poses, "items") else []
    for obj_name, pose in items:
        arr = np.asarray(pose)
        if arr.ndim == 2 and arr.shape == (4, 4):
            x = float(arr[0, 3])
            y = float(arr[1, 3])
            z_rot = float(np.arctan2(arr[1, 0], arr[0, 0]))
        elif arr.ndim == 1 and arr.shape[0] >= 3:
            # Fallback if object_poses is already (x, y, z_rot[, ...]).
            x = float(arr[0])
            y = float(arr[1])
            z_rot = float(arr[2]) if arr.shape[0] >= 3 else 0.0
        else:
            continue
        out[obj_name] = {"x": x, "y": y, "z_rot": z_rot}
    return out


def _empty_result(initial_state) -> dict[str, Any]:
    """Return the same shape as DataGenerator.generate() but with no trajectory."""
    return dict(
        initial_state=initial_state,
        states=[],
        observations=[],
        datagen_infos=[],
        actions=np.zeros((0,), dtype=np.float32),
        success=False,
        src_demo_inds=[],
        src_demo_labels=np.zeros((0, 1), dtype=int),
    )


def cluster_to_chained_warp_constraint(
    center_feature: "list[float] | np.ndarray",
    stddev_feature: "list[float] | np.ndarray",
    state_schema: dict[str, dict[str, int]],
    subtask_idx: int,
    *,
    slack_alpha: float = 1.5,
    slack_widen_factor: float = 2.0,
    min_slack_xy: float = 0.003,
    max_slack_xy: float = 0.03,
    min_slack_z_rot: float = 0.05,
    max_slack_z_rot: float = 0.5,
    objects: "list[str] | None" = None,
) -> dict:
    """Build a ``chained_warp_constraint`` dict from a node-cluster's center + stddev.

    The output matches the schema consumed by
    ``scripts/run_mimicgen_generate.py --chained_warp_constraint=...``.

    Args:
        center_feature:    Cluster centre in (x, y, sinθ, cosθ) feature space.
                           Shape ``(4 * n_objects,)``.
        stddev_feature:    Per-dim stddev for the cluster.
        state_schema:      Object → qpos-index mapping (only used for the
                           object name list; values are ignored here).
        subtask_idx:       MimicGen subtask boundary to constrain on (the
                           constraint fires after this subtask completes).
        slack_alpha:       Multiplier on stddev when deriving slack box.
        slack_widen_factor: Carried into the constraint dict for the generator
                           to use during retries.
        min/max slack:     Clamp to a manipulation-realistic range.
        objects:           Restrict checking to a subset of object names. None
                           ⇒ all objects in ``state_schema``.

    Returns:
        ``{"subtask_idx": int, "target_pose": {obj: {x, y, z_rot}},
            "slack": {obj: {x, y, z_rot}}, "slack_widen_factor": float,
            "objects": [...] | None}``
    """
    from policy_doctor.mimicgen.failure_targeting import (
        cluster_center_to_object_poses,
    )
    target_pose = cluster_center_to_object_poses(
        np.asarray(center_feature, dtype=np.float64),
        state_schema=state_schema,
    )
    slack = derive_slack_from_stddev(
        stddev_feature,
        state_schema=state_schema,
        alpha=slack_alpha,
        min_slack_xy=min_slack_xy,
        max_slack_xy=max_slack_xy,
        min_slack_z_rot=min_slack_z_rot,
        max_slack_z_rot=max_slack_z_rot,
    )
    out = {
        "subtask_idx": int(subtask_idx),
        "target_pose": target_pose,
        "slack": slack,
        "slack_widen_factor": float(slack_widen_factor),
    }
    if objects is not None:
        out["objects"] = list(objects)
    return out


def derive_slack_from_stddev(
    stddev_feature: list[float] | np.ndarray,
    state_schema: dict[str, dict[str, int]],
    alpha: float = 1.5,
    min_slack_xy: float = 0.003,    # 3 mm — below this the warp's own noise dominates
    max_slack_xy: float = 0.03,     # 3 cm — bigger than this isn't a meaningful target
    min_slack_z_rot: float = 0.05,  # ~3°
    max_slack_z_rot: float = 0.5,   # ~29° — keeps the cluster centered, not "anywhere"
) -> dict[str, dict[str, float]]:
    """Per-object slack box from a node-cluster's per-dim stddev.

    The cluster's center/stddev are in our (x, y, sin(z_rot), cos(z_rot))
    feature space. We convert back to angular slack via the
    Pythagorean-on-sin/cos heuristic — adequate for the order-of-magnitude
    we need (the user explicitly allowed loose matching).

    Args:
        stddev_feature: 4 * n_objects-dim per-dim stddev for the cluster.
        state_schema:   Used only for object names / ordering. Matches the
                        encoding used by ``_state_to_cluster_features``.
        alpha:          Slack multiplier on stddev. 1.5 ≈ a 1.5-sigma window.
        min_slack_*:    Lower clamp — prevents pathological tightness when
                        a cluster collapses to a single point.
        max_slack_*:    Upper clamp — prevents pathological looseness when
                        the cluster is essentially uniform across the workspace.

    Returns:
        ``{obj_name: {"x": slack_x, "y": slack_y, "z_rot": slack_z_rot}}``,
        same shape as ``IntermediateConstraint.slack``.
    """
    arr = np.asarray(stddev_feature, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for i, obj_name in enumerate(sorted(state_schema)):
        std_x = float(arr[4 * i + 0])
        std_y = float(arr[4 * i + 1])
        std_sin = float(arr[4 * i + 2])
        std_cos = float(arr[4 * i + 3])
        # Convert (sin, cos) stddev pair → angular stddev (rough Pythagorean proxy).
        std_z_rot = float(np.sqrt(std_sin ** 2 + std_cos ** 2))
        slack_x = float(np.clip(alpha * std_x, min_slack_xy, max_slack_xy))
        slack_y = float(np.clip(alpha * std_y, min_slack_xy, max_slack_xy))
        slack_z_rot = float(np.clip(alpha * std_z_rot, min_slack_z_rot, max_slack_z_rot))
        out[obj_name] = {"x": slack_x, "y": slack_y, "z_rot": slack_z_rot}
    return out

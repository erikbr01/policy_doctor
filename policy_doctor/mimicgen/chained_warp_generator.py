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

    Full SE(3): translation in xyz and orientation as a quaternion. Slack
    is per-translation-axis plus a single ``rotation`` scalar (max angular
    distance in radians).

    Attributes:
        subtask_idx:   0-based subtask index. The check happens after the
                       subtask at this index completes, i.e. at the start
                       of subtask ``subtask_idx + 1``.
        target_pose:   Per-object SE(3) target ``{obj_name: {"x", "y", "z",
                       "qw", "qx", "qy", "qz"}}``. World frame, quaternion
                       in wxyz convention.
        slack:         Per-object slack ``{obj_name: {"x", "y", "z",
                       "rotation"}}``. ± offsets in metres on x/y/z; ``rotation``
                       is a max angular distance in radians. Any axis whose
                       slack value is ``None`` or absent is *not* checked
                       (treated as "don't care").
        slack_widen_factor: Multiplier applied to slack when the outer caller
                       retries after enough rejections.
        objects:       Subset of objects to actually constrain. None ⇒ all
                       in ``target_pose``.

    Distance metrics:
        translation:  |achieved.x - target.x|  per axis (same for y, z)
        rotation:     min-angle between achieved and target quaternions,
                      i.e. 2 * arccos(|achieved · target|)
    """

    subtask_idx: int
    target_pose: dict[str, dict[str, float]]
    slack: dict[str, dict[str, "float | None"]]
    slack_widen_factor: float = 2.0
    objects: Optional[list[str]] = None

    def widen(self, factor: float) -> "IntermediateConstraint":
        """Return a copy with slack multiplied by ``factor`` (None values preserved)."""
        widened: dict[str, dict[str, "float | None"]] = {}
        for obj, axes in self.slack.items():
            widened[obj] = {
                axis: (None if v is None else float(v) * factor)
                for axis, v in axes.items()
            }
        return IntermediateConstraint(
            subtask_idx=self.subtask_idx,
            target_pose=self.target_pose,
            slack=widened,
            slack_widen_factor=self.slack_widen_factor,
            objects=self.objects,
        )

    def is_satisfied(self, achieved_pose: dict[str, dict[str, float]]) -> tuple[bool, dict[str, float]]:
        """Check whether ``achieved_pose`` lies within this constraint's slack.

        Args:
            achieved_pose: ``{obj_name: {"x", "y", "z", "qw", "qx", "qy", "qz"}}``
                from the sim after the constrained subtask completed. Each
                object's pose must include xyz and quaternion (wxyz).

        Returns:
            ``(satisfied, worst_ratios)`` where ``worst_ratios[obj]`` is the
            max (axis-distance / axis-slack) across all checked axes for that
            object — a value ≤1 means satisfied, ``inf`` means missing data.
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
                worst[obj] = float("inf")
                return False, worst

            ratios: list[float] = []
            # Translation axes (xyz).
            for axis in ("x", "y", "z"):
                s = slack.get(axis)
                if s is None or s <= 0:
                    continue
                if axis not in achieved or axis not in target:
                    continue
                d = abs(float(achieved[axis]) - float(target[axis]))
                ratios.append(d / float(s))

            # Orientation: single scalar angular slack.
            sr = slack.get("rotation")
            if sr is not None and sr > 0:
                ang = _quaternion_angle(achieved, target)
                if ang is not None:
                    ratios.append(ang / float(sr))

            if not ratios:
                continue
            worst[obj] = float(max(ratios))
            if worst[obj] > 1.0:
                return False, worst
        return True, worst


def _wrap_angle(theta: float) -> float:
    """Wrap to [-pi, pi]."""
    return float(np.arctan2(np.sin(theta), np.cos(theta)))


def _quaternion_angle(
    achieved: dict[str, float],
    target: dict[str, float],
) -> "float | None":
    """Min-angle between two unit quaternions, in radians.

    Both inputs must carry ``qw``/``qx``/``qy``/``qz`` keys (wxyz convention).
    Returns ``None`` if either side is missing the orientation entries — the
    caller decides what to do (typically: skip the rotation check).

    Uses ``2 * acos(|dot|)``, clamped to [0, π] so the constraint can't blow
    up on floating-point dot products that drift slightly outside [-1, 1].
    """
    keys = ("qw", "qx", "qy", "qz")
    if not all(k in achieved and k in target for k in keys):
        return None
    dot = sum(float(achieved[k]) * float(target[k]) for k in keys)
    # |dot| handles the q vs -q ambiguity (q and -q represent the same rotation).
    dot = max(-1.0, min(1.0, abs(dot)))
    return float(2.0 * np.arccos(dot))


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

def _rot_to_quat_wxyz(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation matrix → unit quaternion (wxyz), canonical hemisphere (qw ≥ 0).

    Uses the standard Shepperd-style branch on the maximum diagonal/trace to
    avoid numerical issues near 180° rotations.
    """
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0)) * 2.0
        qw = 0.25 * s
        qx = float(R[2, 1] - R[1, 2]) / s
        qy = float(R[0, 2] - R[2, 0]) / s
        qz = float(R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 2.0
        qw = float(R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = float(R[0, 1] + R[1, 0]) / s
        qz = float(R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = float(np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])) * 2.0
        qw = float(R[0, 2] - R[2, 0]) / s
        qx = float(R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = float(R[1, 2] + R[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])) * 2.0
        qw = float(R[1, 0] - R[0, 1]) / s
        qx = float(R[0, 2] + R[2, 0]) / s
        qy = float(R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    return qw, qx, qy, qz


def _datagen_info_to_pose7(datagen_info) -> dict[str, dict[str, float]]:
    """Pull ``{obj: {x, y, z, qw, qx, qy, qz}}`` from a MimicGen datagen_info.

    object_poses is a dict mapping ``obj_name`` → 4x4 pose matrix in world frame.
    The quaternion is in canonical (qw ≥ 0) hemisphere.
    """
    out: dict[str, dict[str, float]] = {}
    obj_poses = getattr(datagen_info, "object_poses", {}) or {}
    items = obj_poses.items() if hasattr(obj_poses, "items") else []
    for obj_name, pose in items:
        arr = np.asarray(pose)
        if arr.ndim != 2 or arr.shape != (4, 4):
            continue
        x = float(arr[0, 3])
        y = float(arr[1, 3])
        z = float(arr[2, 3])
        qw, qx, qy, qz = _rot_to_quat_wxyz(arr[:3, :3])
        out[obj_name] = {
            "x": x, "y": y, "z": z,
            "qw": qw, "qx": qx, "qy": qy, "qz": qz,
        }
    return out


# Back-compat alias for any external callers (now returns the full SE(3) dict).
_datagen_info_to_xy_yaw = _datagen_info_to_pose7


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
    min_slack_xyz: float = 0.003,
    max_slack_xyz: float = 0.03,
    min_slack_rotation: float = 0.05,
    max_slack_rotation: float = 0.5,
    objects: "list[str] | None" = None,
) -> dict:
    """Build a ``chained_warp_constraint`` dict from a node-cluster's center + stddev.

    The output matches the schema consumed by
    ``scripts/run_mimicgen_generate.py --chained_warp_constraint=...``.

    Args:
        center_feature:    Cluster centre in 7-dim per-object feature space
                           ``[x, y, z, qw, qx, qy, qz]`` (see
                           :func:`policy_doctor.mimicgen.failure_targeting._state_to_cluster_features`).
                           Shape ``(7 * n_objects,)``.
        stddev_feature:    Per-dim stddev for the cluster (same shape).
        state_schema:      Object → qpos-index mapping. Only the object names
                           and ordering matter here; indices aren't used.
        subtask_idx:       MimicGen subtask boundary to constrain on (the
                           constraint fires after this subtask completes).
        slack_alpha:       Multiplier on stddev when deriving slack box.
        slack_widen_factor: Carried into the constraint dict for the generator
                           to use during retries.
        min/max slack:     Clamp to a manipulation-realistic range (xyz in
                           metres, rotation in radians).
        objects:           Restrict checking to a subset of object names. None
                           ⇒ all objects in ``state_schema``.

    Returns:
        ``{"subtask_idx": int,
            "target_pose": {obj: {x, y, z, qw, qx, qy, qz}},
            "slack":       {obj: {x, y, z, rotation}},
            "slack_widen_factor": float, "objects": [...] | None}``
    """
    from policy_doctor.mimicgen.failure_targeting import (
        cluster_center_to_object_poses,
    )
    target_pose_full = cluster_center_to_object_poses(
        np.asarray(center_feature, dtype=np.float64),
        state_schema=state_schema,
    )
    # Strip the auxiliary "z_rot" key — it's only there for legacy IC-range
    # consumers; the SE(3) constraint compares quaternions directly.
    target_pose = {
        obj: {k: v for k, v in pose.items() if k != "z_rot"}
        for obj, pose in target_pose_full.items()
    }
    slack = derive_slack_from_stddev(
        stddev_feature,
        state_schema=state_schema,
        alpha=slack_alpha,
        min_slack_xyz=min_slack_xyz,
        max_slack_xyz=max_slack_xyz,
        min_slack_rotation=min_slack_rotation,
        max_slack_rotation=max_slack_rotation,
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
    min_slack_xyz: float = 0.003,        # 3 mm — below this the warp's own noise dominates
    max_slack_xyz: float = 0.03,         # 3 cm — bigger than this isn't a meaningful target
    min_slack_rotation: float = 0.05,    # ~3°
    max_slack_rotation: float = 0.5,     # ~29° — keeps the cluster centered, not "anywhere"
) -> dict[str, dict[str, float]]:
    """Per-object slack from a node-cluster's per-dim stddev (full SE(3)).

    The cluster's center/stddev are in the 7-dim feature space
    ``[x, y, z, qw, qx, qy, qz]`` per object (see :func:`_state_to_cluster_features`).
    Translation slack is the per-axis 1-D stddev scaled by ``alpha``;
    rotation slack is derived from the L2 norm of the quaternion-component
    stddev: for small dispersions, ``||Δq||_2 ≈ θ/2``, so we use
    ``angular_stddev ≈ 2 × ||q_stddev||_2``.

    Args:
        stddev_feature: 7 * n_objects per-dim stddev for the cluster.
        state_schema:   Used only for object names / ordering.
        alpha:          Slack multiplier on stddev. 1.5 ≈ a 1.5-sigma window.
        min_slack_xyz / max_slack_xyz:           Clamp on translation slack (metres).
        min_slack_rotation / max_slack_rotation: Clamp on rotation slack (radians).

    Returns:
        ``{obj_name: {"x", "y", "z", "rotation"}}`` — matches
        ``IntermediateConstraint.slack``.
    """
    arr = np.asarray(stddev_feature, dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    for i, obj_name in enumerate(sorted(state_schema)):
        base = 7 * i
        std_x = float(arr[base + 0])
        std_y = float(arr[base + 1])
        std_z = float(arr[base + 2])
        std_qw = float(arr[base + 3])
        std_qx = float(arr[base + 4])
        std_qy = float(arr[base + 5])
        std_qz = float(arr[base + 6])
        q_norm = float(np.sqrt(std_qw ** 2 + std_qx ** 2 + std_qy ** 2 + std_qz ** 2))
        std_rotation = 2.0 * q_norm   # see docstring
        slack_x = float(np.clip(alpha * std_x, min_slack_xyz, max_slack_xyz))
        slack_y = float(np.clip(alpha * std_y, min_slack_xyz, max_slack_xyz))
        slack_z = float(np.clip(alpha * std_z, min_slack_xyz, max_slack_xyz))
        slack_rot = float(np.clip(alpha * std_rotation, min_slack_rotation, max_slack_rotation))
        out[obj_name] = {"x": slack_x, "y": slack_y, "z": slack_z, "rotation": slack_rot}
    return out

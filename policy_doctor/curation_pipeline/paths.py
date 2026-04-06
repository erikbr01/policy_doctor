"""Path resolution for train/eval/attribution outputs.

Matches the naming used in scripts/train/train_policies.sh and
scripts/eval/eval_save_episodes.sh.
"""

from typing import List, Union


def get_train_name(
    train_date: str,
    task: str,
    policy: str,
    seed: Union[int, str],
) -> str:
    """Baseline training run name: ${train_date}_train_${policy}_${task}_${seed}."""
    return f"{train_date}_train_{policy}_{task}_{seed}"


def get_train_dir(
    train_output_dir: str,
    train_date: str,
    task: str,
    policy: str,
    seed: Union[int, str],
) -> str:
    """Full path to the training run directory."""
    name = get_train_name(train_date, task, policy, seed)
    return f"{train_output_dir.rstrip('/')}/{train_date}/{name}"


def get_eval_dir(
    eval_output_dir: str,
    eval_date: str,
    task: str,
    policy: str,
    seed: Union[int, str],
    train_ckpt: str = "latest",
    eval_as_train_seed: bool = True,
) -> str:
    """Full path to the evaluation output directory (rollouts for one checkpoint).

    If eval_as_train_seed is True, the eval run name uses the same seed as training.
    """
    seed_str = str(seed)
    if eval_as_train_seed:
        name = get_train_name(eval_date, task, policy, seed_str)
    else:
        name = f"{eval_date}_train_{policy}_{task}_{seed_str}"
    return f"{eval_output_dir.rstrip('/')}/{eval_date}/{name}/{train_ckpt}"


def get_eval_output_dir_for_ckpt(
    eval_output_dir: str,
    eval_date: str,
    task: str,
    policy: str,
    seed: Union[int, str],
    train_ckpt: str,
    eval_as_train_seed: bool = True,
) -> str:
    """Same as get_eval_dir; alias for clarity when building eval_save_episodes --output_dir."""
    return get_eval_dir(
        eval_output_dir,
        eval_date,
        task,
        policy,
        seed,
        train_ckpt=train_ckpt,
        eval_as_train_seed=eval_as_train_seed,
    )


def expand_seeds(seeds: List[Union[int, str]]) -> List[str]:
    """Normalize seeds to list of strings."""
    if isinstance(seeds, (int, str)):
        return [str(seeds)]
    return [str(s) for s in seeds]

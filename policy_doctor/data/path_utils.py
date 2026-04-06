"""Path helpers for seed-specific eval/train dirs. No dependency on influence_visualizer."""


def get_eval_dir_for_seed(eval_dir: str, seed: str, reference_seed: str) -> str:
    """Derive eval_dir path for a given seed from the reference eval_dir.

    Assumes the path has a segment ending with _reference_seed (e.g. square_mh_0).
    Replaces that segment so it ends with _seed (e.g. square_mh_1).

    Args:
        eval_dir: Path for the reference seed (e.g. .../square_mh_0/latest).
        seed: Target seed string (e.g. "1").
        reference_seed: Seed string that eval_dir is for (e.g. "0").

    Returns:
        Path with the seed segment replaced for the target seed.
    """
    if seed == reference_seed:
        return eval_dir
    path = eval_dir.rstrip("/")
    parts = path.split("/")
    suffix = "_" + reference_seed
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(suffix):
            parts[i] = parts[i][: -len(suffix)] + "_" + seed
            return "/".join(parts)
    return eval_dir


def get_train_dir_for_seed(train_dir: str, seed: str, reference_seed: str) -> str:
    """Derive train_dir path for a given seed from the reference train_dir.

    Same logic as get_eval_dir_for_seed but for train_dir (e.g. .../square_mh_0 -> .../square_mh_1).
    """
    if seed == reference_seed:
        return train_dir
    path = train_dir.rstrip("/")
    parts = path.split("/")
    suffix = "_" + reference_seed
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].endswith(suffix):
            parts[i] = parts[i][: -len(suffix)] + "_" + seed
            return "/".join(parts)
    return train_dir

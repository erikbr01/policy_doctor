"""Load precomputed InfEmbed embeddings from npz."""

from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np


def load_infembed_embeddings(
    path: Path,
    level: str = "rollout",
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load InfEmbed npz with keys rollout_embeddings (N_rollout, D), demo_embeddings (N_demo, D).

    level: 'rollout' or 'demo'. Returns (embeddings, metadata list).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"InfEmbed file not found: {path}")
    with np.load(path, allow_pickle=False) as f:
        rollout_emb = np.asarray(f["rollout_embeddings"])
        demo_emb = np.asarray(f["demo_embeddings"])

    if level == "rollout":
        n = rollout_emb.shape[0]
        metadata = [{"rollout_idx": i, "timestep": i} for i in range(n)]
        return rollout_emb, metadata
    else:
        n = demo_emb.shape[0]
        metadata = [{"demo_idx": i, "timestep": i} for i in range(n)]
        return demo_emb, metadata

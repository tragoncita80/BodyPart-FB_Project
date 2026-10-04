import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from humenv.misc.motionlib import canonicalize, load_episode_based_h5


class PairedTrajectoryBuffer:
    """
    Buffer that stores (full, upper, lower) trajectory triplets indexed by motion_id.
    Each triplet shares the same base motion but differs in which body parts are active.

    File naming convention (from dataset_list.txt):
        <motion_id>.hdf5         -> full body motion
        <motion_id>_upper.hdf5   -> upper-body motion, lower body neutral
        <motion_id>_lower.hdf5   -> lower-body motion, upper body neutral
    """

    def __init__(self, motions_file: str, motions_root: str, seq_length: int, device: str):
        self.seq_length = seq_length
        self.device = device

        # Group files by motion_id
        with open(motions_file, "r") as f:
            h5files = [el.strip().replace(" ", "") for el in f.readlines() if el.strip()]

        groups = defaultdict(dict)
        for h5 in h5files:
            stem = Path(h5).stem  # filename without .hdf5
            if stem.endswith("_upper"):
                motion_id, kind = stem[:-6], "upper"
            elif stem.endswith("_lower"):
                motion_id, kind = stem[:-6], "lower"
            else:
                motion_id, kind = stem, "full"
            groups[motion_id][kind] = h5

        # Keep only complete triplets
        self.triplets = []   # list of dicts: {"full": np.ndarray, "upper": np.ndarray, "lower": np.ndarray}
        for motion_id, files in groups.items():
            if not all(k in files for k in ("full", "upper", "lower")):
                continue
            episode = {}
            ok = True
            for kind in ("full", "upper", "lower"):
                path = canonicalize(files[kind], base_path=motions_root)
                eps = load_episode_based_h5(path, keys=None)
                if len(eps) == 0:
                    ok = False
                    break
                obs = eps[0]["observation"].astype(np.float32)
                if len(obs) < seq_length:
                    ok = False
                    break
                episode[kind] = obs
            if ok:
                # Trim to common length so the three sequences align frame-by-frame
                T = min(episode["full"].shape[0], episode["upper"].shape[0], episode["lower"].shape[0])
                episode = {k: v[:T] for k, v in episode.items()}
                episode["motion_id"] = motion_id
                self.triplets.append(episode)

        print(f"[PairedTrajectoryBuffer] Loaded {len(self.triplets)} complete triplets.")
        if len(self.triplets) == 0:
            raise RuntimeError("No complete (full, upper, lower) triplets found. Check filenames.")

    def __len__(self):
        return len(self.triplets)

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict:
        """
        Sample batch_size triplets, each as a sequence of length seq_length
        starting from the SAME random offset within the motion (so frames are aligned).

        Returns a dict with tensors of shape (batch_size, seq_length, obs_dim).
        """
        idxs = np.random.randint(0, len(self.triplets), size=batch_size)
        full_seqs, upper_seqs, lower_seqs = [], [], []
        for i in idxs:
            ep = self.triplets[i]
            T = ep["full"].shape[0]
            start = np.random.randint(0, T - self.seq_length + 1)
            full_seqs.append(ep["full"][start:start + self.seq_length])
            upper_seqs.append(ep["upper"][start:start + self.seq_length])
            lower_seqs.append(ep["lower"][start:start + self.seq_length])
        return {
            "full":  torch.from_numpy(np.stack(full_seqs)).to(self.device),
            "upper": torch.from_numpy(np.stack(upper_seqs)).to(self.device),
            "lower": torch.from_numpy(np.stack(lower_seqs)).to(self.device),
        }


def load_neutral_z_states(neutral_path: str, motions_root: str, seq_length: int, device: str) -> torch.Tensor:
    """Load a single neutral-pose trajectory of length seq_length, as (1, seq_length, obs_dim)."""
    path = canonicalize(neutral_path, base_path=motions_root)
    eps = load_episode_based_h5(path, keys=None)
    obs = eps[0]["observation"].astype(np.float32)
    if obs.shape[0] < seq_length:
        # Pad by repeating the last frame
        pad = np.tile(obs[-1:], (seq_length - obs.shape[0], 1))
        obs = np.concatenate([obs, pad], axis=0)
    obs = obs[:seq_length]
    return torch.from_numpy(obs).unsqueeze(0).to(device)  # (1, seq_length, obs_dim)
"""Make a release copy of a fairseq HuBERT checkpoint.

Removes the optimizer state (only needed to resume training), which shrinks
the Dog-HuBERT checkpoint from 1.1 GB to about 360 MB. Model weights, config
and task state (the label dictionary) are kept, so the result still loads with
fairseq.checkpoint_utils.load_model_ensemble_and_task.

Absolute paths in the saved config that point at our servers are replaced
with placeholders; they are not needed to load the model.

Usage: python trim_checkpoint.py checkpoint_145_60000.pt dog_hubert.pt
"""
import sys

import torch
from omegaconf import OmegaConf

DROP_KEYS = ["last_optimizer_state"]
PLACEHOLDER = "/path/to/"


def scrub_paths(node):
    """Replace absolute server paths in a config dict with placeholders."""
    if isinstance(node, dict):
        return {k: scrub_paths(v) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub_paths(v) for v in node]
    if isinstance(node, str) and node.startswith(("/home", "/data", "/bigdrive")):
        return PLACEHOLDER + node.rstrip("/").split("/")[-1]
    return node


def main(src, dst):
    ckpt = torch.load(src, map_location="cpu")
    for key in DROP_KEYS:
        ckpt.pop(key, None)
    ckpt["args"] = None
    cfg = ckpt["cfg"]
    if OmegaConf.is_config(cfg):
        ckpt["cfg"] = OmegaConf.create(scrub_paths(OmegaConf.to_container(cfg, resolve=False)))
    else:  # fairseq stores the config as a plain dict
        ckpt["cfg"] = scrub_paths(cfg)
    torch.save(ckpt, dst)
    print(f"saved {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

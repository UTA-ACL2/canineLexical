"""Download model weights and data for canineLexical from the Hugging Face Hub.

    pip install huggingface_hub
    python download.py                 # everything (about 820 MB)
    python download.py --models        # model weights only
    python download.py --data          # phone/word/activity data only
    python download.py --models --only dog_hubert.pt kmeans_layer11_k140.pkl

Models go to ./checkpoints and data to ./data (both are git-ignored).
"""
import argparse
import os

from huggingface_hub import hf_hub_download, snapshot_download

MODEL_REPO = "ArlingtonCL2/canine-lexical-models"
DATA_REPO = "ArlingtonCL2/canine-lexical-data"

HERE = os.path.dirname(os.path.abspath(__file__))

MODEL_FILES = {
    "dog_hubert.pt": "Dog-HuBERT, stage 2 after 60k updates (step 4)",
    "kmeans_layer11_k140.pkl": "k-means over layer-11 features, 140 phones (step 4)",
    "panns_dog_sed_finetuned.pth": "fine-tuned PANNs dog detector (step 2)",
    "tsn_dog_activity.pth": "fine-tuned TSN activity classifier (step 6)",
}


def download_models(only=None):
    target = os.path.join(HERE, "checkpoints")
    for name, desc in MODEL_FILES.items():
        if only and name not in only:
            continue
        print(f"[models] {name}: {desc}")
        hf_hub_download(MODEL_REPO, name, local_dir=target)
    return target


def download_data():
    target = os.path.join(HERE, "data")
    print(f"[data] {DATA_REPO}")
    snapshot_download(DATA_REPO, repo_type="dataset", local_dir=target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", action="store_true", help="download model weights")
    parser.add_argument("--data", action="store_true", help="download phone/word/activity data")
    parser.add_argument("--only", nargs="+", choices=sorted(MODEL_FILES), help="with --models: only these files")
    args = parser.parse_args()

    both = not (args.models or args.data)
    if args.models or both:
        print("saved to", download_models(args.only))
    if args.data or both:
        print("saved to", download_data())


if __name__ == "__main__":
    main()

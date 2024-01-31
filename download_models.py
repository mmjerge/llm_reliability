#!/usr/bin/env python

"""This module downloads opensource models from Hugging Face.
   Adapted from NeurIPS 2023 TDC challenge download_data.py script"""

import os
import argparse
from huggingface_hub import snapshot_download

HUGGINGFACE_MODELS = {
    "meta": {
        "base": "meta-llama/Llama-2-7b-hf",
        "large": "meta-llama/Llama-2-70b-hf"
    },
    "mistral": {
        "base": "mistralai/Mistral-7B-v0.1"
    },
    "microsoft": {
        "base": "microsoft/phi-2"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Add optional huggingface_hub cache directory.")

    parser.add_argument("--model_cache_path", 
                        type=str, 
                        nargs='?',
                        default=None,
                        help="Add optional path for new huggingface_hub model cache directory."
                              " Default is ~/.cache/huggingface/hub")
    
    parser.add_argument("--dataset_cache_path", 
                        type=str, 
                        nargs='?',
                        default=None,
                        help="Add optional path for new huggingface_hub dataset cache directory."
                              " Default is ~/.cache/huggingface/hub")
    
    args = parser.parse_args()

    for model_name, models in HUGGINGFACE_MODELS.items():
        for model_size, model_type in models.items():
            model_download_path = f"./models/{model_name}/{model_size}/{model_type}"
            if not os.path.exists(model_download_path):
                print(f"Downloading {model_name} {model_size}.")
                snapshot_download(repo_id=model_type, 
                                  cache_dir=args.model_cache_path,
                                  local_dir=model_download_path)
                print("Done.")
            else:
                print(f"Model {model_name} {model_size} found; skipping.")

    dataset_download_path = f"./data/truthful_qa"

    if not os.path.exists(dataset_download_path):
        print(f"Downloading dataset.")
        snapshot_download(repo_id="truthful_qa",
                          repo_type="dataset",
                          local_dir=dataset_download_path,
                          cache_dir=args.dataset_cache_path)
        print("Dataset downloaded.")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    main()
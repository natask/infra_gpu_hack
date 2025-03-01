#!/usr/bin/env python3
import argparse
import os
import json
from datasets import load_dataset


def download_dataset(dataset_name, split='train', save_dir='./datasets'):
    """
    Downloads a dataset from Hugging Face and saves each entry under a directory named after the dataset name.

    Args:
        dataset_name (str): The name of the dataset to download.
        split (str): The dataset split to download (e.g., 'train', 'test').
        save_dir (str): The directory to save the downloaded dataset.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset(dataset_name, split=split)

    # Create a directory for the dataset
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Save each entry as a separate file
    for idx, entry in enumerate(dataset):
        entry_path = os.path.join(dataset_dir, f"entry_{idx}.json")
        with open(entry_path, "w") as f:
            json.dump(entry, f)

    print(f"Dataset '{dataset_name}' ({split} split) downloaded and saved to '{dataset_dir}'")


def main():
    parser = argparse.ArgumentParser(description="Download a dataset from Hugging Face")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to download")
    parser.add_argument('--split', type=str, default='train', help="Dataset split to download (default: 'train')")
    parser.add_argument('--save_dir', type=str, default='./datasets', help="Directory to save the dataset")
    args = parser.parse_args()

    download_dataset(args.dataset_name, args.split, args.save_dir)


if __name__ == '__main__':
    main()


"""
Generate CLIP embeddings for fashion products.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import sys

# Allow importing models from the project root.
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from models.image_processor import ImageProcessor


DATA_DIR = BASE_DIR / "data"
PRODUCTS_PATH = DATA_DIR / "products.csv"
OUTPUT_PATH = DATA_DIR / "embeddings.pkl"


class FashionDataset(Dataset):

    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(str(row["image_path"]))

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            return image, index

        except Exception as e:
            print(f"Skipping image {image_path}: {e}")
            return None, index


def collate_images(batch):
    valid = [
        (img, idx)
        for img, idx in batch
        if img is not None
    ]

    if not valid:
        return None

    images, indices = zip(*valid)

    return list(images), list(indices)


def build_embeddings(batch_size=8):

    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {PRODUCTS_PATH}"
        )

    df = pd.read_csv(PRODUCTS_PATH)

    if df.empty:
        raise ValueError("Product dataset is empty.")

    print(f"Loaded {len(df)} products.")

    # Initialize CLIP once
    processor = ImageProcessor()

    dataset = FashionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_images
    )

    embeddings = []
    valid_indices = []

    for batch in tqdm(loader, desc="Generating CLIP embeddings"):

        if batch is None:
            continue

        images, indices = batch

        # Process all images in the batch
        vectors = processor.get_image_embeddings(images)

        embeddings.extend(vectors)
        valid_indices.extend(indices)

    if not embeddings:
        raise RuntimeError("No images were processed.")

    # Keep product metadata aligned with vectors
    result = df.iloc[valid_indices].copy()
    result.reset_index(drop=True, inplace=True)

    result["Embedding"] = [
        np.asarray(vector, dtype=np.float32)
        for vector in embeddings
    ]

    # Save new CLIP embeddings
    result.to_pickle(OUTPUT_PATH)

    print("\nCompleted!")
    print(f"Products processed: {len(result)}")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    args = parser.parse_args()

    build_embeddings(batch_size=args.batch_size)

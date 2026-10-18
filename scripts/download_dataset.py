

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


# Project directories
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
PRODUCTS_PATH = DATA_DIR / "products.csv"


def download_dataset(limit=2000):
    """
    Download product images and metadata.

    Args:
        limit (int): Maximum number of products to download.
    """

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading fashion dataset from Hugging Face...")

    dataset = load_dataset(
        "benitomartin/fashion-product-images-small-900x1200",
        split="train",
        streaming=True
    )

    products = []

    print(f"Downloading up to {limit} products...")

    for index, item in enumerate(
        tqdm(dataset.take(limit), total=limit)
    ):
        try:
            # Extract product information
            product_id = str(item["id"])

            name = item.get(
                "productDisplayName", "Unknown product"
            )

            category = item.get(
                "articleType", "Unknown"
            )

            color = item.get(
                "baseColour", "Unknown"
            )

            # Extract product image
            image = item["image"].convert("RGB")

            # Save product image
            image_path = IMAGE_DIR / f"{product_id}.jpg"

            image.save(
                image_path,
                format="JPEG",
                quality=90
            )

            # Save product metadata
            products.append({
                "id": product_id,
                "name": name,
                "category": category,
                "color": color,
                "image_path": str(image_path)
            })

        except Exception as e:
            print(f"Skipping product {index}: {e}")

    if not products:
        raise RuntimeError(
            "No products were downloaded. "
            "Check your internet connection and dataset format."
        )

    # Convert metadata into a DataFrame
    df = pd.DataFrame(products)

    # Save metadata
    df.to_csv(
        PRODUCTS_PATH,
        index=False,
        encoding="utf-8"
    )

    print("\nDataset download completed!")
    print(f"Products downloaded: {len(df)}")
    print(f"Images saved in: {IMAGE_DIR}")
    print(f"Metadata saved to: {PRODUCTS_PATH}")

    print("\nDataset preview:")
    print(df.head())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Download fashion dataset"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="Maximum number of products to download"
    )

    args = parser.parse_args()

    download_dataset(
        limit=args.limit
    )

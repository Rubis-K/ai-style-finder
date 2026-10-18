

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch

from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "openai/clip-vit-base-patch32"


class ImageProcessor:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading CLIP on {self.device}...")

        self.model = CLIPModel.from_pretrained(
            MODEL_NAME
        ).to(self.device)

        self.model.eval()

        self.preprocess = CLIPProcessor.from_pretrained(
            MODEL_NAME
        )

    
    
    def get_image_embeddings(self, images):
        """
        Generate normalized 512-dimensional CLIP embeddings.
        Compatible with Transformers 4.x and 5.x.
        """

        inputs = self.preprocess(
            images=images,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.inference_mode():
            outputs = self.model.get_image_features(
                pixel_values=pixel_values
            )

            # Transformers 5.x returns a structured object.
            # Transformers 4.x generally returns a tensor.
            if isinstance(outputs, torch.Tensor):
                features = outputs
            else:
                features = outputs.pooler_output

            # Ensure we have a tensor before normalization.
            assert isinstance(features, torch.Tensor), (
                f"Expected tensor, got {type(features)}"
            )

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=-1
            )

        return features.cpu().numpy().astype(np.float32)



    def encode_image(self, image_input, is_url=False):
        """
        Encode a PIL image, local image, or URL.

        Returns:
            {
                "base64": str,
                "vector": numpy.ndarray
            }
        """

        try:
            if isinstance(image_input, Image.Image):
                image = image_input.convert("RGB")

            elif is_url:
                response = requests.get(
                    str(image_input),
                    timeout=15
                )
                response.raise_for_status()

                with Image.open(
                    BytesIO(response.content)
                ) as img:
                    image = img.convert("RGB")

            else:
                with Image.open(Path(image_input)) as img:
                    image = img.convert("RGB")

            # Base64 image for OpenAI vision analysis
            buffer = BytesIO()
            image.save(buffer, format="JPEG")

            encoded_image = base64.b64encode(
                buffer.getvalue()
            ).decode("utf-8")

            # CLIP embedding
            vector = self.get_image_embeddings(
                [image]
            )[0]

            return {
                "base64": encoded_image,
                "vector": vector
            }

        except Exception as e:
            print(f"Error encoding image: {e}")

            return {
                "base64": None,
                "vector": None
            }

   
    def find_similar_products(
        self,
        user_vector,
        dataset,
        top_k=5,
        category="All",
        min_similarity=0.25
    ):
        """
        Find similar products using CLIP embeddings,
        category filtering, and a similarity threshold.
        """

        if user_vector is None or dataset.empty:
            return []

        # Exclude missing embeddings
        valid_rows = dataset[
            dataset["Embedding"].notna()
        ].copy()

        # Filter products by category
        if category and category != "All":
            valid_rows = valid_rows[
                valid_rows["category"]
                .fillna("")
                .astype(str)
                .str.casefold() == category.casefold()
            ]

        if valid_rows.empty:
            return []

        # Prepare vectors
        dataset_vectors = np.vstack(
            valid_rows["Embedding"].values
        ).astype(np.float32)

        query_vector = np.asarray(
            user_vector,
            dtype=np.float32
        ).reshape(1, -1)

        # Ensure matching embedding dimensions
        if query_vector.shape[1] != dataset_vectors.shape[1]:
            raise ValueError(
                "Embedding dimensions do not match. "
                "Regenerate embeddings using CLIP."
            )

        # Calculate cosine similarities
        similarities = cosine_similarity(
            query_vector,
            dataset_vectors
        )[0]

        # Sort highest to lowest
        top_indices = np.argsort(similarities)[::-1]

        results = []

        for index in top_indices:
            score = float(similarities[index])

            if score < min_similarity:
                continue

            row = valid_rows.iloc[index]

            results.append({
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "Unknown")),
                "category": str(row.get("category", "")),
                "color": str(row.get("color", "")),
                "image_path": str(row.get("image_path", "")),
                "similarity": score
            })

            if len(results) >= top_k:
                break

        return results


    def find_closest_match(self, user_vector, dataset):
        """
        Compatibility method for older code.
        """

        matches = self.find_similar_products(
            user_vector,
            dataset,
            top_k=1
        )

        if not matches:
            return None, None

        return matches[0], matches[0]["similarity"]

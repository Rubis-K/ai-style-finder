
"""
Style Finder - AI Fashion Recommendation Application
"""

import gradio as gr
import pandas as pd
from pathlib import Path
from PIL import Image

from models.llm_service import analyze_outfit
from models.image_processor import ImageProcessor


# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "embeddings.pkl"


class StyleFinderApp:

    def __init__(self):
        # Load image processor
        self.image_processor = ImageProcessor()

        # Load fashion dataset
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {DATASET_PATH}\n"
                "Run scripts/download_dataset.py and "
                "scripts/build_embeddings.py first."
            )

        self.data = pd.read_pickle(DATASET_PATH)

        print(
            f"Loaded {len(self.data)} fashion products."
        )

   
    def process_image(self, image, category, threshold):
        """
        Analyze an image and recommend similar products.
        """

        if image is None:
            return "Please upload an image.", []

        # Generate CLIP embedding
        result = self.image_processor.encode_image(image)
        user_vector = result["vector"]

        if user_vector is None:
            return "Unable to process the image.", []

        # Find similar products with filters
        similar_products = (
            self.image_processor.find_similar_products(
                user_vector=user_vector,
                dataset=self.data,
                top_k=5,
                category=category,
                min_similarity=threshold
            )
        )

        if not similar_products:
            return (
                "No products met the selected similarity "
                "threshold. Try a lower threshold or "
                "choose another category.",
                []
            )

        # OpenAI explains the retrieved products
        analysis = analyze_outfit(
            image=image,
            products=similar_products
        )

        # Prepare product gallery
        gallery = []

        for product in similar_products:
            image_path = product["image_path"]

            if Path(image_path).exists():
                caption = (
                    f"{product['name']} | "
                    f"{product['color']} | "
                    f"Similarity: {product['similarity']:.3f}"
                )

                gallery.append((image_path, caption))

        return analysis, gallery



# Initialize application
app = StyleFinderApp()


# Gradio Interface
with gr.Blocks(
    title="AI Style Finder"
) as demo:

    gr.Markdown(
        """
        # AI Style Finder

        Upload a fashion image to discover
        visually similar products and receive
        AI-powered fashion recommendations.
        """
    )

    with gr.Row():

        with gr.Column():

            image_input = gr.Image(
                type="pil",
                label="Upload Fashion Image"
            )

            analyze_button = gr.Button(
                "Analyze Style",
                variant="primary"
            )

            
            # Get available categories from the dataset
            categories = sorted(
                app.data["category"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            # Category filter
            category_input = gr.Dropdown(
                choices=["All"] + categories,
                value="All",
                label="Product Category"
            )

            # Minimum similarity
            similarity_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.25,
                step=0.01,
                label="Minimum Similarity"
            )


        with gr.Column():

            analysis_output = gr.Markdown(
                label="AI Fashion Analysis"
            )

    gr.Markdown("## Similar Fashion Products")

    product_gallery = gr.Gallery(
        label="Similar Fashion Products",
        columns=3,
        height=600,
        object_fit="contain",
        preview=True
    )

    analyze_button.click(
        fn=app.process_image,
        inputs=[
            image_input,
            category_input,
            similarity_input
        ],
        outputs=[
            analysis_output,
            product_gallery
        ]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=5000,
        share=False
    )

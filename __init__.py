
import os
from .image_processor import image_to_data_url


def analyze_outfit(image, products: list[dict]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "*Optional AI description is off. Add `OPENAI_API_KEY` to `.env` to enable it.*"

    try:
        from openai import OpenAI
        from config import OPENAI_MODEL

        catalog = "\n".join(
            f"- ID {p['id']}: {p['name']} | {p['category']} | {p['color']}"
            for p in products
        )
        response = OpenAI().responses.create(
            model=OPENAI_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "Describe the visible clothing and overall style in 3-5 sentences. "
                        "The product search results below are only visual neighbors, NOT verified matches. "
                        "Explain useful similarities and differences, and never invent prices, "
                        "availability, brands, or purchase links.\n\nCandidates:\n" + catalog
                    )},
                    {"type": "input_image", "image_url": image_to_data_url(image)},
                ],
            }],
            max_output_tokens=350,
        )
        return response.output_text or "No AI analysis was returned."
    except Exception as exc:
        return f"*AI analysis unavailable ({type(exc).__name__}: {exc}). Image search still works.*"

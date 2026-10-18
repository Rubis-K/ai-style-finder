import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-6-luna"
)
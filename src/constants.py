# ../src/constants.py

import os
from dotenv import load_dotenv
load_dotenv()


class Constant:
    # api keys
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")

    # file paths
    DATA_DIR = os.path.join("data")
    RUN_DIR = os.path.join(DATA_DIR, "runs")

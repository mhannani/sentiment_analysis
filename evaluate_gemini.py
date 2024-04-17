from dotenv import load_dotenv
from pathlib import Path
import os

from src.utils.parsers import parse_toml
from src.models.gemini import Gemini
from src.prompts.gemini import GeminiPrompt


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # toml path
    toml_path: str = Path(f"./configs/config.toml")

    # parse configuration
    config = parse_toml(toml_path)

    # openai_api_key
    google_api_key: str = os.environ.get("GOOGLE_API_KEY")

    # gemini prompt
    gemini_prompt = GeminiPrompt(config)

    # gemini model
    gemini = Gemini(config, google_api_key, gemini_prompt)
    
    # make prediction
    output = gemini.predict("انا بعدا مقابلة البحر لا يرحل ")
    
    print(output)

from dotenv import load_dotenv
from pathlib import Path
import os

from src.utils.parsers import parse_toml
from src.models.chatgpt import GPT
from src.prompts.chatgpt import GPTPrompt


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # toml path
    toml_path: str = Path(f"./configs/config.toml")

    # parse configuration
    config = parse_toml(toml_path)

    # openai_api_key
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")

    # create prompt
    gpt_prompt = GPTPrompt(config)

    # Instantiate GPT
    gpt = GPT(config, openai_api_key, gpt_prompt)

    # make prediction
    output = gpt.predict("اذا كانت مقابل لاشيء تعثر فما هو السقوط")
    
    print(output)

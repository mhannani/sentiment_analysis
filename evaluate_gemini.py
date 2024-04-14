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
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY")

    import google.generativeai as genai
    genai.configure(api_key="AIzaSyB6n7l4pUce16tFUzsIPzKZ3as2Pwaa-DU")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Write a story about a magic backpack.")

    print(response)

    # # create prompt
    # gpt_prompt = GPTPrompt(config)

    # # Instantiate GPT
    # gpt = GPT(config, openai_api_key, gpt_prompt)

    # output = gpt.predict("انا بعدا مقابلة البحر لا يرحل ")

    # print(output)




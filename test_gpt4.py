from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import os
import csv
import argparse


from src.utils.parsers import parse_toml
from src.models.chatgpt import GPT
from src.prompts.chatgpt import GPTPrompt
from src.utils.readers import read_json


if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate on preprocessed_corpus_json file.')
    
    # Add argument for config file
    parser.add_argument("config_file", type=str, help="configuration filename")
    
    # Add argument for input text
    parser.add_argument('text', type=str, help='Text to be classified')
    
    # parse args
    args = parser.parse_args()

    # toml path
    toml_path: str = Path(f"./configs/{args.config_file}.toml")

    # parse configuration
    config = parse_toml(toml_path)

    # openai_api_key
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")

    # create prompt
    gpt_prompt = GPTPrompt(config)

    # Instantiate GPT
    gpt = GPT(config, openai_api_key, gpt_prompt)

    # make prediction
    output = gpt.predict(args.text)
    
    print(output)

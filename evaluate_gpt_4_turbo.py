from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import os
import csv

from src.utils.parsers import parse_toml
from src.models.chatgpt import GPT
from src.prompts.chatgpt import GPTPrompt
from src.utils.readers import read_json

if __name__ == "__main__":

    # Load environment variables from .env file
    load_dotenv()

    # toml path
    toml_path: str = Path(f"./configs/config.toml")

    # parse configuration
    config = parse_toml(toml_path)

    # useful variables
    data_root = Path(config['data']['root'])
    
    # output root
    output_root = Path(config['output']['root'])
    
    # preprocessed mac json
    preprocessed_mac_json_path = data_root / config['data']['processed'] / config['data']['preprocessed_mac_json']

    # openai_api_key
    openai_api_key: str = os.environ.get("OPENAI_API_KEY")

    # create prompt
    gpt_prompt = GPTPrompt(config)

    # Instantiate GPT
    gpt = GPT(config, openai_api_key, gpt_prompt)

    # read json data
    json_data = read_json(preprocessed_mac_json_path)

    # output json
    output_csv = Path(output_root) / config['output']['predictions']

    # Create the root directory of the file if it does not exist
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file is empty
    if output_csv.exists():
        if os.stat(output_csv.as_posix()).st_size == 0:
            # Open output file for results with UTF-8 encoding
            with open(output_csv.as_posix(), 'a+', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter='\t')
                # Write the header row
                csv_writer.writerow(['key', 'tweets', 'gt_type', 'pred_type', 'class_name'])
    else:
        # Open output file for results with UTF-8 encoding
        with open(output_csv.as_posix(), 'w+', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter='\t')
            # Write the header row
            csv_writer.writerow(['key', 'tweets', 'gt_type', 'pred_type', 'class_name'])

    with open(output_csv.as_posix(), 'a+', newline='', encoding='utf-8') as csv_file:

        # Write the header row
        csv_writer = csv.writer(csv_file, delimiter='\t')

        # loop through the samples
        for sample in tqdm(json_data[5152:], desc="Processing samples", unit="sample", ncols=100):
            
            # key
            key = sample['key']

            # tweet
            tweet = sample['tweet']

            # type
            gt_type = sample['type']

            # make prediction
            output = gpt.predict(tweet)
            
            # class name
            class_name = sample['class_name']

            # extract the predicted class
            pred_type = output['predicted_class']

            # write current row
            csv_writer.writerow([key, tweet, gt_type, pred_type, class_name])

            # Ensure the data is written to the file after each iteration
            csv_file.flush()

from pathlib import Path

from src.preprocessor.preprocessor import Preprocessor
from src.utils.parsers import parse_toml


if __name__ == "__main__":

    # configration filepath
    CONFIG_FILE = Path("configs/config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    exernal_data = config['data']['external']
    raw_data = config['data']['raw']
    interim_data = config['data']['interim']
    processed_data = config['data']['processed']
    
    mac_csv_filename = config['data']['mac_csv_filename']
    preprocessed_mac_json = config['data']['preprocessed_mac_json']

    # constructing the filepath of the corpus
    csv_filepath_raw_data = data_root / raw_data / mac_csv_filename

    # json output path
    output_json_path = data_root / processed_data / preprocessed_mac_json

    # preprocessor instantiation
    preprocessor = Preprocessor(config, csv_filepath_raw_data, output_json_path)
    
    # preprocess data and export it
    df = preprocessor.preprocess_and_export()

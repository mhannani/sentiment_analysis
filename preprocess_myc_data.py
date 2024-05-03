from pathlib import Path

from src.preprocessor.myc_preprocessor import MYCPreprocessor
from src.utils.parsers import parse_toml


if __name__ == "__main__":

    # configration filepath
    CONFIG_FILE = Path("configs/myc_config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    exernal_data = config['data']['external']
    raw_data = config['data']['raw']
    interim_data = config['data']['interim']
    processed_data = config['data']['processed']
    
    
    corpus_csv_filename = config['data']['corpus_csv_filename']
    preprocessed_corpus_json = config['data']['preprocessed_corpus_json']
    preprocessed_corpus_csv = config['data']['preprocessed_corpus_csv']

    # constructing the filepath of the corpus
    csv_filepath_raw_data = data_root / raw_data / corpus_csv_filename

    # json output path
    output_json_path = data_root / processed_data / preprocessed_corpus_json

    # preprocessor instantiation
    preprocessor = MYCPreprocessor(config, csv_filepath_raw_data, output_json_path)
    
    # preprocess data and export it
    df = preprocessor.preprocess_and_export()

""" Split data into train and test datasels and save them"""
from pathlib import Path


from src.data.split import DataSplitter
from src.utils.parsers import parse_toml
from src.utils.readers import read_df


if __name__ == "__main__":

    # configration filepath
    CONFIG_FILE = Path("configs/config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # data root
    data_root = Path(config['data']['root'])
    
    # processed root
    processed_root = config['data']['processed']
    
    # preprocessed mac csv fielname
    preprocessed_mac_csv_filename = config['data']['preprocessed_mac_csv']

    # preprocessed mac csv abs path
    preprocessed_mac_csv_path = data_root / processed_root / preprocessed_mac_csv_filename
    
    # read dataframe
    df = read_df(preprocessed_mac_csv_path)
    
    # data splitter
    data_splitter = DataSplitter(config, df)
    
    # split and save data directly
    data_splitter.split_and_save()

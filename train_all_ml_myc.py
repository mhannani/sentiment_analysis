from pycaret.classification import ClassificationExperiment, compare_models, get_logs

from pathlib import Path
from src.utils.parsers import parse_toml
from src.data.split import DataSplitter

    
if __name__ == "__main__":
    # configuration filepath
    CONFIG_FILE = Path("configs/myc_config.toml")

    # read configuration object
    config = parse_toml(CONFIG_FILE)

    # useful variables
    data_root = Path(config['data']['root'])
    external_data = config['data']['external']
    raw_data = config['data']['raw']
    interim_data = config['data']['interim']
    processed_data = config['data']['processed']
    preprocessed_corpus_csv_filename = config['data']['preprocessed_corpus_csv']

    # constructing the filepath of the corpus
    csv_filepath_processed_data = data_root / processed_data / preprocessed_corpus_csv_filename

    # data splitter
    data_splitter = DataSplitter(config, csv_filepath_processed_data)
    
    # split data into train and test sets
    train_df, test_df = data_splitter.split()
    
    # create classification experiment
    exp = ClassificationExperiment()
    
    # init setup for experiment
    exp.setup(data = train_df, test_data = test_df, target = 'type', session_id = 42, log_experiment = True, use_gpu = True)
    
    print("Training started!")
    # train and compare models
    best = exp.compare_models()
    
    # get results logs
    exp_logs = exp.get_logs()
    
    print(exp_logs)


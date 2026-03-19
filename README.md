# Sentiment Analysis of Moroccan Arabic (Darija)

> A comparative study of sentiment analysis approaches for Moroccan Arabic (Darija), benchmarking fine-tuned BERT models, FastText embeddings, traditional ML classifiers, and large language models (GPT-4 Turbo, Google Gemini) across two Darija datasets.

## Table of Contents

- [Context and Motivation](#context-and-motivation)
- [Research Questions](#research-questions)
- [Architecture Overview](#architecture-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Models](#models)
  - [BERT-Based Models](#bert-based-models)
  - [FastText-Based Model](#fasttext-based-model)
  - [GPT-4 Turbo](#gpt-4-turbo)
  - [Google Gemini](#google-gemini)
  - [Traditional ML Models (PyCaret)](#traditional-ml-models-pycaret)
- [Datasets](#datasets)
  - [MAC Dataset](#mac-dataset)
  - [MYC Dataset](#myc-dataset)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiment Tracking](#experiment-tracking)
- [Configuration](#configuration)
- [Installation and Setup](#installation-and-setup)
- [Usage Examples](#usage-examples)
- [Notebooks](#notebooks)
- [License](#license)

---

## Context and Motivation

Sentiment analysis for Moroccan Arabic (Darija) poses unique challenges that standard Arabic NLP tools do not adequately address:

- **Darija is a low-resource dialect.** Unlike Modern Standard Arabic (MSA), Darija lacks standardized orthography. Speakers write the same word in multiple ways, mix Arabic script with Latin characters (Arabizi), and borrow heavily from French and Spanish.
- **Existing Arabic NLP models are trained primarily on MSA.** BERT models pretrained on formal Arabic text may not generalize well to the informal, dialectal nature of Darija social media content.
- **Code-switching is pervasive.** Moroccan social media users frequently switch between Darija, MSA, French, and Arabizi within a single sentence, making tokenization and classification significantly harder.
- **Large language models remain unproven for Darija.** While GPT-4 and Gemini demonstrate strong multilingual capabilities, their effectiveness on Darija sentiment classification has not been systematically evaluated.

This project investigates whether fine-tuned encoder models (BERT variants) outperform zero-shot large language models on Darija sentiment classification, and whether models pretrained specifically on Darija text offer meaningful advantages over general-purpose Arabic or multilingual models.

## Research Questions

1. **BERT fine-tuning vs. LLM prompting**: Does fine-tuning smaller BERT models on Darija sentiment data outperform zero-shot prompting of GPT-4 Turbo and Google Gemini?
2. **Darija-specific vs. standard Arabic models**: Do BERT models pretrained on Darija (DarijaBERT, DarijaBERT-Arabizi) achieve better sentiment classification accuracy than models pretrained on MSA (AraBERT) or multilingual corpora (mBERT)?
3. **Ternary vs. binary classification**: How do model performance characteristics differ between three-class sentiment (positive, neutral, negative) on the MAC dataset and two-class sentiment (positive, negative) on the MYC dataset?
4. **Embedding-based approaches**: How does a FastText Arabic embedding approach with a custom classifier head compare to transformer-based methods?
5. **Cross-dataset generalization**: How well do models trained on one Darija dataset (MAC) transfer to another (MYC)?

## Architecture Overview

The project follows an end-to-end pipeline from raw data to evaluated predictions:

```
+-------------------+     +----------------------+     +------------------+
|   Raw Datasets    |     |   Preprocessing      |     |   Data Splitting |
|                   | --> |                      | --> |                  |
|  - MAC corpus     |     |  - Text cleaning     |     |  - Stratified    |
|  - MYC corpus     |     |  - Normalization     |     |    train/test    |
+-------------------+     |  - Label encoding    |     +------------------+
                          +----------------------+              |
                                                                v
                    +-------------------------------------------+
                    |                                           |
                    v                                           v
      +-------------------------+              +-----------------------------+
      |   Fine-Tuning Pipeline  |              |   Zero-Shot LLM Pipeline   |
      |                         |              |                            |
      |  - BERT (5 variants)    |              |  - GPT-4 Turbo (LangChain) |
      |  - FastText + MLP       |              |  - Gemini Pro (LangChain)  |
      |  - PyCaret ML models    |              |  - Prompt engineering      |
      +-------------------------+              +-----------------------------+
                    |                                           |
                    v                                           v
              +---------------------------------------------------+
              |              Evaluation Module                    |
              |                                                   |
              |  Accuracy | Precision | Recall | F1 (macro)       |
              |  Confusion Matrix | Classification Report         |
              |  Per-class and cross-dialect evaluation            |
              +---------------------------------------------------+
                                    |
                                    v
                          +-------------------+
                          |  Experiment Logs  |
                          |  (CometML)        |
                          +-------------------+
```

## Features

- **Five BERT model variants** fine-tuned for Darija sentiment classification, including Darija-specific and general Arabic pretrained models.
- **FastText Arabic embeddings** combined with a custom PyTorch Lightning classifier head for a lightweight alternative to transformers.
- **GPT-4 Turbo evaluation** via LangChain with structured output parsing using Pydantic tool-calling and retry logic (tenacity).
- **Google Gemini evaluation** via LangChain for zero-shot sentiment classification.
- **Traditional ML model comparison** using PyCaret's automated model selection and comparison on the MYC dataset.
- **Comprehensive text preprocessing** tailored for Arabic and Darija text: URL removal, HTML stripping, emoji conversion, Arabic letter normalization, duplicate character reduction.
- **Stratified data splitting** to preserve class distributions across train and test sets.
- **Cross-dialect evaluation** that measures model performance separately on standard Arabic and dialectal (Darija) portions of the MAC dataset.
- **TOML-based configuration** for reproducible experiments across datasets.
- **CometML integration** for experiment tracking, metric logging, and confusion matrix visualization.

## Tech Stack

| Technology | Role |
|---|---|
| **PyTorch** | Deep learning framework for model training and inference |
| **Hugging Face Transformers** | Pretrained BERT models, tokenizers, `Trainer` API for fine-tuning |
| **PyTorch Lightning** | Training framework for the FastText classifier head |
| **FastText** | Arabic word embeddings (`facebook/fasttext-ar-vectors`) |
| **LangChain** | Orchestration framework for GPT-4 Turbo and Gemini prompting |
| **OpenAI API** | GPT-4 Turbo / GPT-3.5 Turbo inference |
| **Google Generative AI** | Gemini Pro inference |
| **PyCaret** | Automated ML model training and comparison |
| **scikit-learn** | Evaluation metrics, stratified splitting, classification reports |
| **CometML** | Experiment tracking, metric logging, confusion matrix visualization |
| **Optuna** | Hyperparameter optimization (experimental) |
| **pandas** | Data manipulation and preprocessing |
| **seaborn / matplotlib** | Data visualization |
| **tomli** | TOML configuration parsing |
| **tenacity** | Retry logic for API calls |
| **python-dotenv** | Environment variable management for API keys |

## Project Structure

```
sentiment_analysis/
|
|-- configs/                          # Experiment configuration files
|   |-- config.toml                   # MAC dataset configuration
|   |-- myc_config.toml               # MYC dataset configuration
|
|-- data/
|   |-- external/                     # Data from third-party sources
|   |-- raw/
|   |   |-- MAC/                      # Raw MAC corpus and lexicon
|   |   |-- MYC/                      # Raw MYC corpus
|   |-- interim/                      # Intermediate transformed data
|   |-- processed/
|       |-- MAC/                      # Cleaned MAC data (CSV, JSON, train/test splits)
|       |-- MYC/                      # Cleaned MYC data (CSV, JSON)
|
|-- docs/                             # Sphinx documentation source
|   |-- conf.py
|   |-- index.rst
|   |-- getting-started.rst
|   |-- commands.rst
|
|-- models/                           # Trained and serialized model artifacts
|
|-- notebooks/                        # Jupyter notebooks for analysis and presentation
|   |-- data_exploration.ipynb
|   |-- fine_tuned_bert_on_mac_evaluted_on_mac.ipynb
|   |-- TO_BE_PRESENTED_fasttext.ipynb
|   |-- TO_BE_PRESENTED_fine_tuned_bert_on_mac_evaluated_on_myc.ipynb
|   |-- TO_BE_PRESENTED_fine_tuned_bert_on_myc_evaluated_on_myc.ipynb
|   |-- TO_BE_PRESENTED_gpt_sentiment_analysis_on_myc.ipynb
|   |-- gpt_sentiment_analysis_on_mac.ipynb
|   |-- ml_models_myc.ipynb
|   |-- freezed_backbone_bert copy.ipynb
|   |-- plans.ipynb
|   |-- test_train.ipynb
|
|-- output/                           # Prediction results and evaluation outputs
|   |-- MAC/                          # GPT predictions on MAC data
|   |-- MYC/                          # GPT predictions on MYC data
|
|-- references/                       # Reference materials and papers
|-- reports/                          # Generated analysis reports
|
|-- scripts/                          # Shell scripts for environment and data setup
|   |-- download_mac_dataset.sh       # Downloads MAC corpus from GitHub
|   |-- activate_conda_env.sh
|   |-- setup_container.sh
|   |-- run_container.sh
|   |-- export_conda_env.sh
|   |-- image_requirements.sh
|
|-- src/                              # Source package
|   |-- data/
|   |   |-- sentiment_data.py         # PyTorch Dataset for BERT tokenized inputs
|   |   |-- split.py                  # Stratified train/test splitting with DataSplitter
|   |
|   |-- models/
|   |   |-- classifier.py             # ClassifierHead (Lightning) and SentimentClassifier (BERT wrapper)
|   |   |-- chatgpt.py                # GPT model wrapper with LangChain and tool-calling
|   |   |-- gemini.py                 # Gemini model wrapper with LangChain
|   |   |-- embeddings.py             # Embedding utilities
|   |
|   |-- preprocessor/
|   |   |-- preprocessor.py           # MAC dataset preprocessor (clean, encode, export)
|   |   |-- myc_preprocessor.py       # MYC dataset preprocessor (handles encoding issues, extra commas)
|   |
|   |-- prompts/
|   |   |-- chatgpt.py                # GPT prompt construction using LangChain templates
|   |   |-- gemini.py                 # Gemini prompt construction using LangChain messages
|   |
|   |-- trainer/
|   |   |-- trainers.py               # Custom BertTrainer extending HF Trainer
|   |
|   |-- types/
|   |   |-- review_class.py           # Pydantic model for structured GPT output (ReviewClass)
|   |
|   |-- utils/
|       |-- cleaners.py               # Text cleaning: URL/HTML removal, normalization, emoji handling
|       |-- encoders.py               # Label encoding and FastText text-to-embedding conversion
|       |-- evaluator.py              # Evaluator classes for ternary (MAC) and binary (MYC) classification
|       |-- get.py                    # Model/tokenizer loader with pretrained model ID mapping
|       |-- callbacks.py              # Custom training callbacks
|       |-- converters.py             # DataFrame-to-list converters
|       |-- counters.py               # Counting utilities
|       |-- find.py                   # Search utilities
|       |-- model.py                  # Model parameter counting utilities
|       |-- parsers.py                # TOML configuration parser
|       |-- readers.py                # CSV/JSON file readers
|       |-- save.py                   # CSV/JSON file writers
|       |-- visualizers.py            # Plotting utilities
|
|-- train_bert.py                     # Train all 5 BERT variants on a dataset
|-- train_with_fasttext.py            # Train FastText + classifier head
|-- train_all_ml_myc.py               # Train and compare ML models on MYC with PyCaret
|-- evaluate_myc_bert.py              # Evaluate fine-tuned BERT models on MYC test set
|-- evaluate_fasttext.py              # Evaluate FastText classifier from checkpoint
|-- evaluate_gpt_4_turbo.py           # Evaluate GPT-4 Turbo on preprocessed corpus
|-- evaluate_gpt_4_turbo_myc_data.py  # Evaluate GPT-4 Turbo on MYC data
|-- evaluate_gemini.py                # Evaluate Google Gemini on sample input
|-- preprocess_mac_data.py            # Run MAC preprocessing pipeline
|-- preprocess_myc_data.py            # Run MYC preprocessing pipeline
|-- split_dataset.py                  # Split preprocessed data into train/test and save
|-- setup.py                          # Package setup
|-- requirements.txt                  # Python dependencies
|-- Makefile                          # Build automation
|-- LICENSE                           # MIT License
|-- tox.ini                           # Tox testing configuration
```

## Models

### BERT-Based Models

The project fine-tunes five pretrained BERT variants for sequence classification. Each model is loaded via `AutoModelForSequenceClassification` from Hugging Face and fine-tuned end-to-end using the Hugging Face `Trainer` API.

| Model ID | Hugging Face Identifier | Pretraining Data | Notes |
|---|---|---|---|
| `bert-base-multilingual-cased` | `google-bert/bert-base-multilingual-cased` | 104 languages including Arabic | General multilingual baseline |
| `bert-base-arabic` | `asafaya/bert-base-arabic` | Large-scale Arabic text | Standard Arabic BERT |
| `bert-base-arabertv2` | `aubmindlab/bert-base-arabertv2` | 77GB of Arabic text (news, Wikipedia, OSCAR) | Strong MSA representation |
| `DarijaBERT` | `SI2M-Lab/DarijaBERT` | Moroccan Darija text | Darija-specific BERT (Arabic script) |
| `darijabert-arabizi` | `SI2M-Lab/DarijaBERT-arabizi` | Moroccan Darija in Arabizi script | Darija-specific BERT (Latin script) |

**Fine-tuning approach:**

- The `SentimentClassifier` class wraps a `BertForSequenceClassification` model. It supports two modes:
  - **Full fine-tuning**: All BERT layers and the classification head are updated during training.
  - **Frozen backbone**: The BERT encoder layers are frozen and only the classification head is retrained. The existing classifier head is replaced with a new `nn.Linear` layer mapping from `hidden_size` to `num_classes`.
- Training uses the Hugging Face `Trainer` with per-epoch evaluation and checkpointing.
- Metrics (accuracy, precision, recall, F1 with macro averaging) are computed at each evaluation step.
- Confusion matrices are logged to CometML at every epoch.
- Input text is tokenized with `padding='max_length'`, `truncation=True`, and `max_length=512`.

### FastText-Based Model

This approach uses pretrained Arabic FastText word embeddings (`facebook/fasttext-ar-vectors`) as a feature extractor, followed by a custom classification head.

**Pipeline:**

1. Load the pretrained FastText Arabic model from Hugging Face Hub.
2. For each sentence, split into tokens, compute per-token FastText embeddings, and average them into a single sentence embedding.
3. Feed the sentence embeddings into a `ClassifierHead` (PyTorch Lightning module):
   - `Linear(input_dim, 128)` followed by `ReLU`, `Dropout`, `Linear(128, num_classes)`, and `Softmax`.
4. Train with `CrossEntropyLoss` and Adam optimizer (`lr=0.001`) for 100 epochs.
5. Evaluate using accuracy, precision, recall, and F1 score.

### GPT-4 Turbo

GPT-4 Turbo is evaluated in a zero-shot prompting setup through the OpenAI API, orchestrated with LangChain.

**Approach:**

- A system message instructs the model to classify Arabic reviews as positive, neutral, or negative (ternary for MAC) or positive/negative (binary for MYC).
- The model is invoked using LangChain's `ChatOpenAI` with `temperature=0` for deterministic outputs.
- **Structured output** is enforced via LangChain tool-calling: the model's response is bound to a `ReviewClass` Pydantic schema that returns the predicted class as an integer.
- A `JsonOutputToolsParser` extracts the prediction from the tool call response.
- **Retry logic** uses `tenacity` with random exponential backoff (min 1s, max 10s, up to 6 attempts) to handle rate limits and transient API errors.
- Predictions are written incrementally to a CSV file with columns: `key`, `tweets`, `gt_type`, `pred_type`, `class_name`.

**System prompts:**

- MAC (ternary): *"Predict the class of this Arabic review (e.g ternary classification), whether it's positive (return 2), neutral (return 1) or negative (return 0) review. Please do not return anything other than that."*
- MYC (binary): *"Predict the class of this Arabic review (e.g binary classification), whether it's positive (return 2) or negative (return 0) review. Please do not return anything other than that."*

### Google Gemini

Google Gemini Pro is evaluated similarly to GPT-4 Turbo, using LangChain's `ChatGoogleGenerativeAI` integration.

**Approach:**

- A system message asks: *"What is the sentiment of the following tweets? Answer with positive, negative, or neutral."*
- The human message contains the tweet text.
- The model is invoked through LangChain's message-based interface.
- Unlike the GPT pipeline, Gemini does not currently use structured tool-calling output parsing.

### Traditional ML Models (PyCaret)

For the MYC dataset, the project also runs an automated ML model comparison using PyCaret:

- `ClassificationExperiment` is initialized with the train/test split.
- `compare_models()` trains and evaluates multiple classical ML algorithms (logistic regression, SVM, random forest, gradient boosting, and others).
- GPU acceleration is enabled.
- Experiment logs are recorded automatically.

## Datasets

### MAC Dataset

The **Moroccan Arabic Corpus (MAC)** is a sentiment analysis dataset of user-generated Arabic content (reviews, comments, tweets).

| Property | Value |
|---|---|
| **Source** | [LeMGarouani/MAC on GitHub](https://github.com/LeMGarouani/MAC) |
| **Task** | Ternary sentiment classification |
| **Classes** | Positive (2), Neutral (1), Negative (0) |
| **Text column** | `tweets` |
| **Sentiment column** | `type` (positive / neutral / negative / mixed) |
| **Dialect column** | `class` (dialectal / standard) |
| **Preprocessing** | Mixed-type samples are removed; `class` is renamed to `class_name` |
| **Train/test split** | 90% / 10% (stratified) |
| **Format** | CSV (raw), CSV + JSON (processed) |

The MAC dataset includes a `class` column that distinguishes between standard Arabic and dialectal (Darija) text, enabling cross-dialect evaluation.

### MYC Dataset

The **MYC** dataset is a binary sentiment corpus of Moroccan Arabic text.

| Property | Value |
|---|---|
| **Task** | Binary sentiment classification |
| **Classes** | Positive (1), Negative (0) -- mapped from original labels 1 and -1 |
| **Text column** | `sentence` (renamed to `tweets` during preprocessing) |
| **Sentiment column** | `polarity` (renamed to `type` during preprocessing) |
| **Encoding** | UTF-16 (requires special handling for parsing) |
| **Known issues** | Extra commas in CSV rows, malformed labels (`1-` instead of `-1`) |
| **Train/test split** | 80% / 20% (stratified) |
| **Format** | CSV (raw, UTF-16 encoded), CSV + JSON (processed) |

The MYC preprocessor (`MYCPreprocessor`) includes special logic to handle the dataset's formatting issues: it reads the file with UTF-16 encoding, handles rows with extra commas by joining all elements except the last (which is the label), and corrects malformed labels.

## Text Preprocessing Pipeline

Both datasets go through a shared text cleaning pipeline defined in `src/utils/cleaners.py`. The steps are applied in order:

1. **HTML tag removal** -- Strip `<br>` tags, anchor tags (`<a>...</a>`), and all remaining HTML tags.
2. **URL removal** -- Remove all `http://` and `https://` URLs.
3. **Special character removal** -- Remove all characters that are not word characters or whitespace.
4. **Single character removal** -- Remove isolated single Latin characters (e.g., stray letters from code-switching).
5. **Comma removal** -- Strip commas.
6. **Lowercasing** -- Convert all text to lowercase.
7. **Whitespace normalization** -- Collapse multiple spaces into a single space.
8. **Emoji conversion** -- Convert emoji characters to their English text descriptions using the `emoji` library (e.g., a heart emoji becomes `:red_heart:`).
9. **Trimming** -- Remove leading and trailing whitespace.
10. **Duplicate letter reduction** -- Collapse repeated characters (e.g., "هههههه" becomes "ه"), repeated punctuation, and repeated spaces.
11. **Arabic letter normalization** -- Normalize variant forms of Arabic letters:
    - Aleph with Hamza (above/below) and Madda to plain Aleph
    - Aleph Maksura to Ya
    - Ta Marbuta at word boundaries to Ha

After cleaning, labels are encoded numerically:
- **MAC**: `negative=0`, `neutral=1`, `positive=2`; dialect class: `standard=0`, `dialectal=1`
- **MYC**: `negative (-1) -> 0`, `positive (1) -> 1`

## Training

### Train BERT Models

Train all five BERT variants on a dataset. The script iterates through each model, initializes a CometML experiment, and runs the Hugging Face `Trainer`.

```bash
python train_bert.py <config_file> <experiment_name> <finetune_mode>
```

**Arguments:**

| Argument | Description | Example |
|---|---|---|
| `config_file` | Name of the TOML config file (without path or extension) | `config` or `myc_config` |
| `exp_name` | Experiment name for CometML and output directory | `mac-ternary-exp1` |
| `finetune` | Whether to fine-tune all layers (`True`) or freeze the backbone (`False`) | `True` |

**Example:**

```bash
# Fine-tune all BERT models on MAC dataset
python train_bert.py config mac-experiment-1 True

# Fine-tune all BERT models on MYC dataset
python train_bert.py myc_config myc-experiment-1 True
```

### Train FastText Classifier

Train the FastText embedding + classifier head pipeline.

```bash
python train_with_fasttext.py <config_file> <experiment_name>
```

**Example:**

```bash
# Train FastText classifier on MAC dataset
python train_with_fasttext.py config fasttext-mac-exp1

# Train FastText classifier on MYC dataset
python train_with_fasttext.py myc_config fasttext-myc-exp1
```

### Train All ML Models (MYC)

Run PyCaret's automated model comparison on the MYC dataset.

```bash
python train_all_ml_myc.py
```

This script uses the `myc_config.toml` configuration and trains multiple classical ML models, comparing them automatically.

## Evaluation

### Metrics

All models are evaluated using the following metrics with **macro averaging** (to account for class imbalance):

| Metric | Description |
|---|---|
| **Accuracy** | Proportion of correctly classified samples |
| **Precision (macro)** | Average precision across all classes |
| **Recall (macro)** | Average recall across all classes |
| **F1 Score (macro)** | Harmonic mean of precision and recall, averaged across classes |
| **Confusion Matrix** | Per-class prediction breakdown |
| **Classification Report** | Per-class precision, recall, F1, and support |

The `Evaluator` class in `src/utils/evaluator.py` supports:
- **Overall evaluation** on the full test set.
- **Cross-dialect evaluation** (MAC only): splits the test set by `class_name` (standard vs. dialectal) and evaluates each portion separately.

The `MYCEvaluator` subclass adjusts the label set to `[0, 1]` for binary classification.

### Evaluate BERT Models on MYC

```bash
python evaluate_myc_bert.py
```

This loads each fine-tuned BERT checkpoint from disk and evaluates it on the MYC test set.

### Evaluate FastText Classifier

```bash
python evaluate_fasttext.py <config_file> <experiment_name>
```

**Example:**

```bash
python evaluate_fasttext.py config fasttext-mac-exp1
```

### Evaluate GPT-4 Turbo

```bash
python evaluate_gpt_4_turbo.py <preprocessed_corpus_json> <config_file>
```

**Prerequisites:** Set your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

**Example:**

```bash
python evaluate_gpt_4_turbo.py test config
```

### Evaluate Google Gemini

```bash
python evaluate_gemini.py
```

**Prerequisites:** Set your Google API key in a `.env` file:

```
GOOGLE_API_KEY=your-google-api-key
```

## Experiment Tracking

The project uses **CometML** for experiment tracking during BERT training. For each model and experiment:

- A new CometML experiment is created with the project name `{experiment_name}-id-{model_id}`.
- Training and evaluation metrics (loss, accuracy, precision, recall, F1) are logged at each epoch.
- Confusion matrices are logged as JSON artifacts at each evaluation step, named `confusion-matrix-epoch-{N}.json`.
- The number of trainable parameters is recorded in the output directory name.

To use CometML, set your API key as an environment variable or log in via the CometML CLI before training:

```bash
export COMET_API_KEY=your-comet-api-key
```

## Configuration

The project uses two TOML configuration files, one per dataset. Both follow the same structure.

### `configs/config.toml` (MAC Dataset)

```toml
[data]
root = "data"
external = "external/MAC"
interim = "interim/MAC"
processed = "processed/MAC"
raw = "raw/MAC"

corpus_csv_filename = "MAC_corpus.csv"
preprocessed_corpus_json = "MAC_corpus.json"
preprocessed_corpus_csv = "MAC_corpus.csv"

train_csv_filename = "train.csv"
test_csv_filename = "test.csv"

[params]
train_test_or_val_size = 0.1        # 10% test set
batch_size = 32
hidden_size = 512
dropout_prob = 0.1
num_classes = 3                      # positive, neutral, negative

[output]
root = "output/MAC"
tsv_corpus_predictions = "predictions.mac.gpt.tsv"

[prompting]
system_message = "Predict the class of this Arabic review ..."
gemini_system_message = "What is the sentiment of the following tweets? ..."

[pretrained]
bert_models = ['asafaya/bert-base-arabic', 'SI2M-Lab/DarijaBERT-arabizi']
```

### `configs/myc_config.toml` (MYC Dataset)

```toml
[data]
root = "data"
external = "external/MYC"
interim = "interim/MYC"
processed = "processed/MYC"
raw = "raw/MYC"

corpus_csv_filename = "MYC_corpus.csv"
preprocessed_corpus_json = "MYC_corpus.json"
preprocessed_corpus_csv = "MYC_corpus.csv"

train_csv_filename = "train.csv"
test_csv_filename = "test.csv"

[params]
train_test_or_val_size = 0.2        # 20% test set
batch_size = 64
num_epoch = 36
hidden_size = 512
dropout_prob = 0.1
num_classes = 2                      # positive, negative

[output]
root = "output/MYC"
tsv_corpus_predictions = "results_of_MYC_corpus.csv"

[prompting]
system_message = "Predict the class of this Arabic review ..."
gemini_system_message = "What is the sentiment of the following tweets? ..."
```

**Key configuration differences between MAC and MYC:**

| Parameter | MAC | MYC |
|---|---|---|
| `num_classes` | 3 (ternary) | 2 (binary) |
| `train_test_or_val_size` | 0.1 (10% test) | 0.2 (20% test) |
| `batch_size` | 32 | 64 |
| `num_epoch` | (set in training args) | 36 |

## Installation and Setup

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended for BERT training)
- conda (recommended) or pip

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd sentiment_analysis
```

### Step 2: Create a Conda Environment

```bash
conda create -n sentiment python=3.9
conda activate sentiment
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install the Source Package

```bash
pip install -e .
```

### Step 5: Download the MAC Dataset

```bash
bash scripts/download_mac_dataset.sh
```

This downloads `MAC_corpus.csv` and `MAC_lexicon.xlsx` into `data/raw/MAC/`.

For the MYC dataset, place the raw CSV file at `data/raw/MYC/MYC_corpus.csv`.

### Step 6: Set Up API Keys

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=sk-your-openai-key
GOOGLE_API_KEY=your-google-api-key
COMET_API_KEY=your-comet-api-key
```

### Step 7: GPU Setup (for BERT Training)

Ensure PyTorch is installed with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage Examples

### Full Pipeline: MAC Dataset

```bash
# 1. Preprocess the MAC dataset
python preprocess_mac_data.py

# 2. Split into train/test sets
python split_dataset.py

# 3. Train all BERT models
python train_bert.py config mac-exp-1 True

# 4. Train FastText classifier
python train_with_fasttext.py config fasttext-mac-1

# 5. Evaluate GPT-4 Turbo
python evaluate_gpt_4_turbo.py test config

# 6. Evaluate Gemini
python evaluate_gemini.py
```

### Full Pipeline: MYC Dataset

```bash
# 1. Preprocess the MYC dataset
python preprocess_myc_data.py

# 2. Train all BERT models on MYC
python train_bert.py myc_config myc-exp-1 True

# 3. Evaluate BERT models on MYC test set
python evaluate_myc_bert.py

# 4. Train and compare classical ML models
python train_all_ml_myc.py

# 5. Train FastText classifier on MYC
python train_with_fasttext.py myc_config fasttext-myc-1
```

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploration, analysis, and presentation of results.

| Notebook | Description |
|---|---|
| `data_exploration.ipynb` | Exploratory data analysis of the MAC and MYC datasets: class distributions, text length statistics, sample inspection |
| `fine_tuned_bert_on_mac_evaluted_on_mac.ipynb` | Evaluation results of BERT models fine-tuned on MAC and tested on MAC |
| `TO_BE_PRESENTED_fine_tuned_bert_on_mac_evaluated_on_myc.ipynb` | Cross-dataset evaluation: BERT models trained on MAC, evaluated on MYC |
| `TO_BE_PRESENTED_fine_tuned_bert_on_myc_evaluated_on_myc.ipynb` | Evaluation results of BERT models fine-tuned on MYC and tested on MYC |
| `TO_BE_PRESENTED_fasttext.ipynb` | FastText classifier training and evaluation results |
| `TO_BE_PRESENTED_gpt_sentiment_analysis_on_myc.ipynb` | GPT-4 Turbo zero-shot evaluation on MYC dataset |
| `gpt_sentiment_analysis_on_mac.ipynb` | GPT sentiment analysis results on MAC dataset |
| `ml_models_myc.ipynb` | Classical ML model comparison on MYC using PyCaret |
| `freezed_backbone_bert copy.ipynb` | Experiments with frozen BERT backbone (only classifier head trained) |
| `plans.ipynb` | Project planning and experiment roadmap |
| `test_train.ipynb` | Training experimentation and debugging notebook |

## License

MIT License. Copyright (c) 2024, mhannani. See [LICENSE](LICENSE) for details.

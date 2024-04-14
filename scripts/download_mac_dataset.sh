#!/bin/bash

echo "Downloading has been started"

# Define the MAC URL
MAC_URL="https://raw.githubusercontent.com/LeMGarouani/MAC/main/MAC%20corpus.csv"

# Define the MAC's Lexicon
MAC_LEXICON=https://raw.githubusercontent.com/LeMGarouani/MAC/main/MAC%20lexicon.xlsx

# Define the destination directory
DEST_DIR="$(pwd)/data/raw/MAC"

# Create the directory if it doesn't exist
mkdir -p "$DEST_DIR"

echo "Downloading MAC dataset"

# Download the MAC dataset
curl -s -o "$DEST_DIR/MAC_corpus.csv" "$MAC_URL"

echo "Downloading MAC's Lexicon"

# Download the MAC dataset
curl -s -o "$DEST_DIR/MAC_lexicon.xlsx" "$MAC_LEXICON"

# Check if download was successful
if [ $? -eq 0 ]; then
    echo "Done!"
else
    echo "Download failed! Check the wheel :)"
fi

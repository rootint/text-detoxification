#!/bin/bash

# The URL to download the zip file from
url="https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"

# The output folder
output_folder="data/raw"

# Use wget to download the file
wget "$url" -P "$output_folder"

# Unzip the file in the output folder
unzip -o "$output_folder/filtered_paranmt.zip" -d "$output_folder"
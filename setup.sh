#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python3 -m spacy download en_core_web_sm
# LLM-Reliability

LLM Reliability Study.

This repository is still under development.

# Reliability Study of Large Language Models Using Inference-Time Methods

## Introduction

This repository is dedicated to the study of the reliability of large language models (LLMs) using inferenc-time methods. The aim is to systematize and evaluate different inference-time methods on diverse models and datasets.

## Project Description

- **Objective:** To assess and enhance the reliability of LLMs using various ensemble strategies
- **Methods:** Implementation and comparison of ensemble techniques 
- **Applications:** Focus on natural language processing tasks such as sentiment analysis, text classification, and question-answering systems

## Directory Description



## Repository Structure

- `/models` - Language model implementations and modifications
- `/data` - Datasets
- `/experiments` - Experiment scripts and setup instructions
- `/results` - Results, graphs, and analytical reports from experiments
- `/docs` - Additional documentation and project reports
- `/utils` - Generalized preprocessing scripts
- `/src` - Main source code
- `/tests` - Function and class tests using pytest
- `/notebooks` - Example notebooks

## Getting Started

To participate in this project:

Clone the repository (assumes SSH cloning capability)
   ```bash
   git clone git@github.com:mmjerge/llm_reliability.git
   ```
Create environment
   ```bash
   conda env create -f environment.yaml
   ```
Download models and data
   ```bash
   python3 src/main.py
   ```
Optionally, you can specify a different cache directory for the HuggingFace models and data by using the --cache_path argument. This can be useful for managing disk memory allocation. For example:
   ```bash
   python3 src/main.py --cache_path /path/to/your/cache
   ```
The default cache directory is ~/.cache/huggingface/hub if no --cache_path argument is provided.

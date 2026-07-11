# Next Word Prediction System using Transformer

A deep learning based **Next Word Prediction System** built using a decoder-only Transformer architecture in **PyTorch**. The model learns language patterns from the WikiText-2 dataset and generates probable next words based on the given input sequence.

This project focuses on understanding and implementing the core concepts behind modern language models, including **self-attention, positional encoding, token embeddings, and autoregressive text generation**.

---

## Project Overview

Next word prediction is an important Natural Language Processing (NLP) task used in applications such as:

* Smart keyboards
* Search autocomplete
* Email assistants
* Text generation systems
* Large Language Models (LLMs)

The objective of this project is to build a lightweight Transformer-based language model capable of predicting the next token given previous context.

---

# Key Features

### Transformer Decoder Architecture

Implemented a GPT-style decoder model with:

* Token Embeddings
* Positional Encoding
* Multi-Head Self Attention
* Feed Forward Neural Networks
* Layer Normalization
* Residual Connections
* Causal Attention Masking

### Complete NLP Pipeline

Developed an end-to-end pipeline including:

* Text preprocessing
* Custom tokenizer implementation
* Vocabulary generation
* Token encoding and decoding
* Dataset preparation
* Model training
* Text generation inference

### Deployment Ready

The trained model is integrated with:

* **FastAPI backend** for model inference
* **Streamlit interface** for interactive text generation

---

# Model Architecture

The system follows the Transformer decoder architecture:

```
Input Text
    |
    v
Tokenizer
    |
    v
Token Embedding
    |
    v
Positional Encoding
    |
    v
Transformer Decoder Blocks
    |
    ├── Multi Head Self Attention
    |
    ├── Feed Forward Network
    |
    v
Linear Projection Layer
    |
    v
Next Token Prediction
```

The model predicts the probability distribution over the vocabulary and selects the most probable next token.

---

# Dataset

## WikiText-2 Dataset

The model is trained on the WikiText-2 language modeling dataset.

Dataset characteristics:

* Source: Wikipedia articles
* Task: Next token prediction
* Vocabulary size: ~66K tokens
* Context length: 64 tokens

---

# Technology Stack

## Machine Learning

* Python
* PyTorch
* Transformer Architecture
* Deep Learning
* Natural Language Processing

## Backend

* FastAPI

## Frontend

* Streamlit

## Data Processing

* HuggingFace Datasets
* NumPy

## Development Tools

* Git
* PyCharm

---

# Project Structure

```
next-word-prediction-transformer/

│
├── app/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # Text generation logic
│   ├── model_loader.py      # Model loading
│   └── schemas.py           # API schemas
│
├── frontend/
│   └── streamlit_app.py     # User interface
│
├── models/
│   ├── transformer.py       # Transformer implementation
│   └── embedding.py         # Embedding layers
│
├── training/
│   ├── train.py             # Training pipeline
│   └── predict.py           # Prediction script
│
├── utils/
│   ├── tokenizer.py         # Custom tokenizer
│   ├── dataset.py           # Dataset creation
│   ├── preprocess.py        # Text preprocessing
│   └── padding.py           # Sequence padding
│
├── requirements.txt
├── README.md
└── test.py
```

---

# Training Pipeline

The model training process consists of:

### 1. Data Preparation

* Download WikiText-2 dataset
* Clean and preprocess text
* Build vocabulary
* Convert text into token sequences

### 2. Dataset Creation

Input-target pairs are generated:

Example:

```
Input:
The weather is

Target:
beautiful
```

The model learns to predict the target token from previous context.

### 3. Model Training

Training configuration:

| Parameter       | Value               |
| --------------- | ------------------- |
| Framework       | PyTorch             |
| Architecture    | Transformer Decoder |
| Loss Function   | Cross Entropy Loss  |
| Optimizer       | AdamW               |
| Context Length  | 64 tokens           |
| Vocabulary Size | ~66K                |

---

# Installation and Setup

## Clone Repository

```bash
git clone https://github.com/ruchi315/next-word-prediction-transformer.git

cd next-word-prediction-transformer
```

---

## Install Dependencies

Create environment:

```bash
python -m venv venv
```

Activate environment:

Mac/Linux:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Train Model

```bash
python -m training.train
```

---

## Generate Predictions

Run inference:

```bash
python -m training.predict
```

Example:

```
Input:
machine learning

Output:
machine learning is a field of artificial intelligence...
```

---

## Run FastAPI Backend

```bash
uvicorn app.main:app --reload
```

API:

```
http://127.0.0.1:8000
```

---

## Run Streamlit Application

```bash
streamlit run frontend/streamlit_app.py
```

---

# Sample Output

Example:

```
Prompt:
the united

Generated Text:

the united states is one of the largest economies...
```

---

# Engineering Highlights

* Implemented Transformer architecture from scratch using PyTorch
* Built custom tokenization and preprocessing pipeline
* Developed complete training and inference workflow
* Integrated trained deep learning model with API backend
* Created interactive UI for real-time predictions

---

# Future Improvements

* Train on larger datasets
* Implement Byte Pair Encoding (BPE) tokenizer
* Add beam search decoding
* Improve text generation quality using larger models
* Deploy model using cloud infrastructure
* Add model evaluation metrics

---

# Author

**Khushi Jainth**

Integrated Dual Degree Student
IIT (BHU), Varanasi

---

## License

This project is developed for educational and learning purposes.

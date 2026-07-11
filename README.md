# Next Word Prediction System using Transformer

A Transformer-based language model that predicts the next possible words given an input sequence.  
The project implements a GPT-style decoder architecture from scratch using **PyTorch**, trained on the **WikiText-2 dataset** to learn language patterns and generate context-aware text predictions.

The system includes a complete ML pipeline:
- Data preprocessing and custom tokenization
- Vocabulary generation
- Dataset creation for next-token prediction
- Transformer decoder model implementation
- Model training and checkpointing
- Text generation inference pipeline
- FastAPI backend
- Streamlit interactive user interface

---

# Project Overview

Next word prediction is a fundamental Natural Language Processing (NLP) task used in applications such as:

- Autocomplete systems
- Smart keyboards
- Search suggestions
- Email assistants
- Large Language Models (LLMs)

This project explores the core architecture behind modern language models by implementing a Transformer-based neural network that learns word relationships and predicts the most probable next token based on previous context.

---

# Features

## Transformer Architecture

Implemented a decoder-only Transformer architecture inspired by GPT models.

Components include:

- Token Embeddings
- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Neural Network
- Layer Normalization
- Residual Connections
- Causal Masking for autoregressive generation


## Custom NLP Pipeline

The complete data pipeline was implemented from scratch:

- Text preprocessing
- Custom tokenizer
- Vocabulary creation
- Token encoding and decoding
- Context-target sequence generation
- Padding and batching


## Model Training

The model is trained using:

- Framework: PyTorch
- Dataset: WikiText-2
- Objective: Next token prediction
- Loss Function: Cross Entropy Loss
- Optimizer: AdamW
- Hardware acceleration: Apple Silicon MPS support


## Inference System

The trained model can generate text predictions based on user prompts.

Example:

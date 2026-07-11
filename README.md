# Next Word Prediction System using Transformer

A deep learning based **Next Word Prediction System** built using a decoder-only Transformer architecture in **PyTorch**.

The model learns language patterns from the **WikiText-2 dataset** using a custom NLP pipeline and predicts the most probable next words based on the given input sequence.

This project implements the core concepts behind modern language models, including:

- Token embeddings
- Positional encoding
- Multi-head self-attention
- Transformer decoder blocks
- Autoregressive text generation

The project also includes a complete deployment pipeline with a **FastAPI backend** and **Streamlit interactive interface**.

---

# Project Overview

Next word prediction is a fundamental Natural Language Processing (NLP) task used in:

- Smart keyboard suggestions
- Search autocomplete
- Email assistants
- Text generation systems
- Large Language Models (LLMs)

The goal of this project is to build a lightweight Transformer-based language model that predicts the next token based on previous context.

---

# Key Features

## Transformer Decoder Architecture

Implemented a GPT-style decoder-only Transformer model with:

- Token Embeddings
- Positional Encoding
- Multi-Head Self Attention
- Feed Forward Neural Network
- Layer Normalization
- Residual Connections
- Causal Attention Masking

---

## Complete NLP Pipeline

Built an end-to-end pipeline including:

- Dataset loading
- Text preprocessing
- Custom tokenizer implementation
- Vocabulary generation
- Token encoding and decoding
- Sequence generation
- Model training
- Text generation inference

---

## Deployment

The trained model is integrated with:

- **FastAPI** for backend inference
- **Streamlit** for interactive user interface

---

# System Architecture

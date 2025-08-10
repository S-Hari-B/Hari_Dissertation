# Multi‑Agent EV Valuation — Dissertation Project

## Overview
This repository contains the runnable code for my dissertation project, a modular multi‑agent pipeline for short‑horizon valuation of electric‑vehicle companies. The system combines:
1) Fundamentals forecasting (Agent 1),
2) Expert‑news sentiment using FinBERT (Agent 2),
3) Public‑news sentiment using a GPT model (Agent 3),
4) A unified aggregation and ranking module (Agent 4).

The actual checkpoint for Agent 1 (AutoGluon ensemble) is large (~3 GB) and is stored outside the repository. Instructions to obtain it are below.

## Repository Layout
```
Hari_Dissertation/
│
├── agents/ # Individual agent training/building code (for reference/retraining)
│   ├── agent1.ipynb
│   ├── Agent2.ipynb
│   ├── agent3.ipynb
│   └── agent4.ipynb
│
├── unified_cli.py # Main CLI entry point for running the full system
├── unified_core_final.py # Core unified pipeline logic
├── README.md # Project documentation
└── .env.example # Environment variables example
```

## Prerequisites
- Python 3.10 or 3.11
- Alpha Vantage API key (for fundamentals and headline metadata)
- OpenAI API key (for GPT‑based public sentiment scoring)
- Optional CUDA GPU for faster FinBERT inference

## Setup

1. **Download the Multi-Agent Core Files**
   To get started, download the zipped core files containing the main scripts:  
   [**Download multi_agent_core_files.zip**](https://github.com/S-Hari-B/Hari_Dissertation/raw/main/multi_agent_core_files.zip)  
   Extract the contents into the repository root.

2. **Install dependencies**
   Install required libraries manually or via your preferred method (pip/conda).

3. **Configure environment variables**
   Create a `.env` file at the repository root based on `.env.example`:
   ```env
   AV_KEY=YOUR_ALPHA_VANTAGE_KEY
   OPENAI_API_KEY=YOUR_OPENAI_KEY
   ```

4. **Obtain the Agent 1 model checkpoint**
   Download the pre‑trained model from Google Drive:  
   [Download Agent 1 Model](https://drive.google.com/file/d/1ncPJ9qA7HqvRZmGf5oc1ogy60kRxvvjU/view?usp=sharing)  
   After downloading:
   - Unzip or place the **entire** predictor folder at the path `./agent1_model` so that `./agent1_model/learner.pkl` and related files exist.
   - Do not rename internal files. Only the top‑level directory should be `agent1_model`.

## Quick Start
Run the command‑line tool and choose tickers from the predefined universe.
```bash
python unified_cli.py
```
The script will:
- Build a panel by querying Alpha Vantage for fundamentals and headlines,
- Score expert sentiment with FinBERT locally,
- Score public sentiment with GPT,
- Return a ranked list with a short rationale per company.

## Notes
- Alpha Vantage free tier is restrictive. Use a paid tier or run with fewer tickers to avoid hitting the quota.
- The `.env` file is read via `python‑dotenv`.
- The OpenAI model used by default in the code is `gpt-4o-mini`. You can change `model=` in the relevant functions if needed.
- The first FinBERT run will download `yiyanghkust/finbert-tone` from Hugging Face to your cache.
- This project is for academic demonstration only. It does not provide financial advice.

## License
This repository contains academic research code. Third‑party models and APIs are used under their respective licenses.

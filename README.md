# Text Summarizer

Turn long conversations and messages into short summaries with a fine-tuned T5 model, served by a FastAPI backend and a clean web interface.

![Demo]()

## Overview

This project fine-tunes **T5-small** on the [SAMSum](https://huggingface.co/datasets/Samsung/samsum) dataset of chat dialogues, then wraps the model in a **FastAPI** service with a responsive single-page UI. Paste a conversation, click **Summarize**, and get a short summary in seconds.

## Features

- Fine-tuned T5-small for dialogue summarization
- REST API (`POST /summarize/`) with request validation
- Web UI with live word count, sample text, one-click copy, and light/dark mode
- Runs on CPU, Apple Silicon (MPS) or NVIDIA GPU, whichever is available
- Health check endpoint (`GET /health`)
- Full training notebook included

## Tech stack

Python, PyTorch, Hugging Face Transformers, FastAPI, Uvicorn, HTML/CSS/JavaScript

## Project structure

```
text-summarizer/
├── notebooks/
│   └── training.ipynb     # Fine-tuning and evaluation (trained on Google Colab)
├── assets/
│   └── demo.gif           # Demo of the web UI
├── app.py                 # FastAPI app and inference code
├── index.html             # Web interface
├── requirements.txt
└── README.md
```

## How it works

1. **Training:** T5-small is fine-tuned on SAMSum using Hugging Face Transformers (see `notebooks/training.ipynb`).
2. **Serving:** `app.py` loads the model once at startup and exposes it through FastAPI.
3. **Inference:** input text is cleaned, tokenized (max 512 tokens), and summarized with beam search (4 beams, up to 150 tokens).
4. **UI:** `index.html` sends the text to `/summarize/` and displays the result.


## Getting started

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/text-summarizer.git
cd text-summarizer
pip install -r requirements.txt
```

### 2. Get the model


**Option : Train it yourself**

Run `notebooks/training.ipynb` (a free Colab GPU works), then save the model into a folder named `saved_summary_model` in the project root. That is the default location `app.py` loads from.

### 3. Run the app

```bash
python -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000 in your browser. Interactive API docs are at http://127.0.0.1:8000/docs.

## API

**POST** `/summarize/`

```bash
curl -X POST http://127.0.0.1:8000/summarize/ \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Anna: Are we still meeting at 5? Ben: Yes, see you at the cafe. Anna: Great, I will bring the notes."}'
```

Response:

```json
{ "summary": "Anna and Ben will meet at the cafe at 5." }
```

**GET** `/health` returns the service status and the device in use.

## Limitations

- Trained on short, informal chat dialogues, so long articles and formal documents may summarize poorly
- Input is truncated to 512 tokens
- Summaries can occasionally miss details or misattribute speakers, which is common for small seq2seq models

## Future improvements

- Fine-tune a larger model or add a news/article dataset
- Let users control summary length
- Add Docker support and a hosted demo
- Add automated tests for the API

## Acknowledgements

- [SAMSum dataset](https://huggingface.co/datasets/Samsung/samsum) (check its license for usage terms)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [T5: Exploring the Limits of Transfer Learning](https://arxiv.org/abs/1910.10683)

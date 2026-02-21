# Medical Assistant Chatbot — Fine-tuned TinyLlama with LoRA

A domain-specific AI assistant for healthcare, built by fine-tuning TinyLlama-1.1B-Chat using parameter-efficient fine-tuning (LoRA). The assistant understands medical queries and provides accurate, relevant health information in a conversational interface.

>  **Disclaimer:** This assistant is for educational purposes only and does not replace professional medical advice. Always consult a qualified doctor.

---

---

## Project Overview

This project fine-tunes **TinyLlama-1.1B-Chat-v1.0** on a curated medical question-answer dataset using **LoRA (Low-Rank Adaptation)** via the Hugging Face `peft` library. The objective is to build a healthcare assistant that can accurately respond to medical questions, significantly outperforming the base pre-trained model.

The fine-tuned model is deployed as an interactive chatbot using **Gradio** on Hugging Face Spaces with **ZeroGPU** for fast, free inference.

---

## Repository Structure

```
├── model-training.ipynb     # Full training pipeline: data preprocessing, LoRA fine-tuning, evaluation
├── app.py                   # Gradio web interface for the chatbot
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Dataset

- **Source:** [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)
- **Domain:** Healthcare / Medical Q&A
- **Format:** Instruction-response pairs covering a wide range of medical topics including symptoms, treatments, prevention, and general health advice
- **Preprocessing:** Data was tokenized using the TinyLlama tokenizer, formatted into a `<|user|> ... <|assistant|>` chat template, and truncated to fit within the model's context window. Low-quality and duplicate entries were filtered out to ensure high-quality training examples.

---

## 🔧 Fine-Tuning Methodology

The model was fine-tuned using **LoRA (Low-Rank Adaptation)**, a parameter-efficient technique that injects trainable low-rank matrices into the transformer layers while keeping the original model weights frozen. This allows effective fine-tuning on limited GPU resources such as Google Colab's free T4 GPU.

**LoRA Configuration:**
- Rank (`r`): 16
- Alpha: 32
- Target modules: `q_proj`, `v_proj`
- Dropout: 0.05

**Training Framework:** Hugging Face `transformers` + `peft` + `trl`

---

## 🧪 Experiment Results

Three experiments were conducted with different learning rates and batch size configurations to identify the optimal hyperparameters. All experiments ran for 5 epochs on the same dataset.

| Experiment | Learning Rate | Batch Size | Grad Accum | Effective Batch | Epochs | Train Loss | Val Loss | Perplexity | BLEU Score | ROUGE-L Score | GPU Memory (GB) | Time (Min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (No Fine-tuning) | — | — | — | — | — | — | 11.8996 | 147,204.22 | 67.67 | 76.57 | — | — |
| Exp 1 — High LR | 1e-4 | 2 | 4 | 8 | 5 | 0.7976 | 0.3680 | 1.44 | 88.67 | 94.72 | 6.28 | 27.06 |
| Exp 2 — Medium LR | 5e-5 | 1 | 3 | 3 | 5 | 0.5717 | 0.3658 | 1.44 | 88.69 | 94.81 | 6.28 | 47.97 |
| Exp 3 — Low LR | 2e-5 | 2 | 2 | 4 | 5 | 0.8304 | 0.3893 | 1.48 | 80.38 | 85.27 | 6.28 | 27.29 |

**Best Model: Exp 2 (Medium LR, 5e-5)** — achieved the lowest validation loss (0.3658) and highest ROUGE-L score (94.81).

### Improvements Over Baseline

| Metric | Baseline | Best Model (Exp 2) | Improvement |
|---|---|---|---|
| Perplexity | 147,204.22 | 1.44 | **↓ 99.999%** |
| BLEU Score | 67.67 | 88.69 | **↑ 31.06%** |
| ROUGE-L Score | 76.57 | 94.81 | **↑ 23.83%** |

All three fine-tuned experiments dramatically outperformed the baseline, confirming the significant value of domain-specific fine-tuning.

---


## Running the App Locally

### Prerequisites
- Python 3.10 or higher
- pip
- At least 8GB RAM (16GB recommended)
- GPU is optional but speeds up inference significantly

### Step 1 — Clone the repository
```bash
git clone https://github.com/fadhuweb/medical_chatbot_assistant.git
cd medical_chatbot_assistant
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
python app.py
```

The app will automatically download the fine-tuned model from Hugging Face Hub on first run. 

Then open your browser and go to:
```
http://127.0.0.1:7860
```

---

## Running the Training Notebook on Google Colab

The training notebook `model-training.ipynb` is designed to run end-to-end on **Google Colab** with minimal setup.

1. Click the **Open in Colab** badge at the top of this README
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom
4. The notebook covers data loading, preprocessing, LoRA fine-tuning, evaluation, and saving the model

---

## Tech Stack

| Component | Tool |
|---|---|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning | LoRA via Hugging Face `peft` |
| Training Environment | Google Colab (T4 GPU) |
| Web Interface | Gradio |
| Deployment | Hugging Face Spaces (ZeroGPU) |
| Evaluation | BLEU, ROUGE-L, Perplexity |

---
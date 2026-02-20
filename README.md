# Medical Assistant Chatbot — Fine-tuned TinyLlama with LoRA

A domain-specific AI assistant for healthcare, built by fine-tuning TinyLlama using parameter-efficient fine-tuning (LoRA). The assistant answers medical questions and provides health-related information in a conversational interface.

> **Disclaimer:** This assistant is for educational purposes only and does not replace professional medical advice. Always consult a qualified doctor.

## Project Overview

This project fine-tunes **TinyLlama-1.1B-Chat** on a medical question-answer dataset using **LoRA (Low-Rank Adaptation)** via the `peft` library. The goal is to improve the model's ability to answer healthcare questions accurately compared to the base pre-trained model.

The final model is deployed as an interactive chatbot using **Gradio** on Hugging Face Spaces with ZeroGPU for fast inference.

---

## Repository Structure

```
├── model-training.ipynb     # Full training pipeline (data prep, fine-tuning, evaluation)
├── app.py                   # Gradio web interface
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

### Running the App Locally

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/fadhuweb/medical_chatbot_assistant.git
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the model
The app will automatically download the fine-tuned model from Hugging Face Hub on first run. Make sure you have an internet connection.

### Step 4 — Run the app
```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:7860
```

---

## Running the Training Notebook

The training notebook `model-training.ipynb` is designed to run on **Google Colab** with a free GPU.

1. Open the notebook in Colab using the badge above
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run all cells from top to bottom
4. The notebook will handle data loading, preprocessing, fine-tuning, and evaluation automatically

---

## Tech Stack

- **Model:** TinyLlama-1.1B-Chat-v1.0
- **Fine-tuning:** LoRA via Hugging Face `peft`
- **Training:** Google Colab (T4 GPU)
- **Interface:** Gradio
- **Deployment:** Hugging Face Spaces (ZeroGPU)

---

## Dependencies

```
torch==2.5.1
transformers==4.47.1
accelerate==1.2.1
peft==0.14.0
gradio
huggingface_hub
sentencepiece==0.2.0
protobuf==3.20.3
spaces
```

---


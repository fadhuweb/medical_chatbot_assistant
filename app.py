import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import os
import spaces
from huggingface_hub import snapshot_download

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_MODEL_REPO = "fadhuweb/medical-assistant-model"
MAX_TOKENS = 100

# ============================================================================
# MODEL LOADING
# ============================================================================
model = None
tokenizer = None

def load_model():
    global model, tokenizer

    # Auto-detect: use local if folder exists, otherwise download from Hub
    if os.path.exists("./best_model"):
        adapter_path = "./best_model"
    else:
        adapter_path = snapshot_download(repo_id=HF_MODEL_REPO)

    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_model_name = peft_config.base_model_name_or_path

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.float16)
    model = model.merge_and_unload()
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

# Load on startup
load_model()

# ============================================================================
# GENERATION
# ============================================================================
@spaces.GPU
def generate_response(message, history):
    """Generate response - compatible with Gradio ChatInterface"""
    global model, tokenizer

    if model is None or tokenizer is None:
        return "Model not loaded. Please refresh."

    prompt = f"<|user|>\n{message}\n<|assistant|>\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=80,
    )

    # Move to GPU if available
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response if response else "Unable to generate response."

# ============================================================================
# GRADIO UI
# ============================================================================
css = """
    .disclaimer {
        background: rgba(254, 243, 199, 0.3);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #d97706;
        font-size: 0.9rem;
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
    #title {
        text-align: center;
    }
"""

with gr.Blocks() as demo:

    gr.Markdown("# 🏥 Medical Assistant", elem_id="title")
    gr.Markdown("### Intelligent Healthcare Assistant", elem_id="title")

    with gr.Accordion("📖 How to use", open=False):
        gr.Markdown("""
            1. **Type your question** in the text box at the bottom.
            2. **Press Enter** or click the send button to submit.
            3. **Read the response** from our AI assistant.
            4. **Reference the FAQs** if you need starting points.
        """)

    gr.HTML("""
        <div class="disclaimer">
            <strong>⚠️ Important Notice:</strong> This AI assistant is for educational purposes only.
            It does not replace professional medical advice. Always consult a qualified doctor.
        </div>
    """)

    gr.Markdown("## ❓ Frequently Asked Questions")

    chatbot = gr.ChatInterface(
        fn=generate_response,
        chatbot=gr.Chatbot(height=450),
        textbox=gr.Textbox(
            placeholder="Ask your medical question...",
            container=False,
            scale=7
        ),
        examples=[
            "What are common symptoms of the flu?",
            "How can I prevent high blood pressure?",
            "What is the recommended daily water intake?",
            "Explain why sleep is important for health.",
        ],
        cache_examples=False,
    )

    gr.HTML('<p class="footer">Medical Assistant | Fine-tuned with LoRA | For Educational Use Only</p>')

# ============================================================================
# LAUNCH
# ============================================================================
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        css=css,
    )

"""
Medical Assistant Chatbot - Professional Interface
Fine-tuned with LoRA | Powered by TinyLlama
"""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
# Use relative path as recommended in implementation plan
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model")
MAX_TOKENS = 100
TEMPERATURE = 0.5

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Medical Assistant Pro",
    page_icon="🏥",
    layout="wide",
)

# ============================================================================
# PROFESSIONAL STYLING (Premium Look)
# ============================================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        max-width: 900px;
        padding-top: 3rem;
        /* Removed forced white background to fix dark mode text visibility */
    }
    
    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e40af, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #64748b; /* Explicit gray color */
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Disclaimer box */
    .disclaimer {
        background: rgba(254, 243, 199, 0.1); /* Lighter background */
        border-left: 5px solid #f59e0b;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .disclaimer-text {
        color: #d97706; /* Explicit amber color */
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }

    /* FAQ Buttons Styling */
    .stButton>button {
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
        margin-bottom: 0.5rem;
    }
    
    /* Footer Styling */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1rem 0;
    }

    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model(adapter_path):
    """Load model and tokenizer"""
    try:
        if not os.path.exists(adapter_path):
            st.error(f"Model path not found: {adapter_path}")
            return None, None, False
            
        # Load config
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model_name = peft_config.base_model_name_or_path
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        except:
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        return model, tokenizer, True
    
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, None, False

# ============================================================================
# RESPONSE GENERATION
# ============================================================================
def generate_response(question, model, tokenizer):
    """Generate medical response"""
    prompt = f"<|user|>\nAnswer this question truthfully\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                min_new_tokens=10,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
            )
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
    
    if question in response:
        response = response.replace(question, "").strip()
    
    if not response or len(response) < 10:
        response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    return response

# ============================================================================
# SIDEBAR (Instructions & Settings)
# ============================================================================
with st.sidebar:
    st.markdown("## 🏥 Medical Assistant")
    st.markdown("---")
    
    st.markdown("### 📋 Instructions")
    st.info("""
    1. **Ask a Question**: Use the chat input at the bottom of the screen.
    2. **Be Specific**: Detailed questions result in better answers.
    3. **Quick Start**: Click one of the suggested questions when starting.
    4. **Reset**: Use the 'Clear' button to reset the session.
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ System Info")
    st.write(f"**Model:** TinyLlama (Fine-tuned)")
    st.write(f"**Status:** {'🟢 Ready' if 'model_loaded' in st.session_state and st.session_state.model_loaded else '🔴 Not Loaded'}")
    
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚠️ Caution")
    st.warning("This AI is for educational use only. Always consult a doctor for medical concerns.")

# ============================================================================
# SESSION STATE
# ============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# ============================================================================
# MAIN UI
# ============================================================================
st.markdown('<h1 class="main-title">Medical Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Healthcare Assistant powered by AI</p>', unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="disclaimer">
    <p class="disclaimer-text"><strong>Important Notice:</strong> This medical assistant is an AI based on experimental data. It does not provide medical diagnoses or replacement for professional consultations.</p>
</div>
""", unsafe_allow_html=True)

# LOAD MODEL
if not st.session_state.model_loaded:
    with st.status("Initializing Medical Model...", expanded=True) as status:
        st.write("Loading weights...")
        model, tokenizer, success = load_model(MODEL_PATH)
        if success:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_loaded = True
            status.update(label="Model Ready!", state="complete", expanded=False)
        else:
            status.update(label="Initialization Failed", state="error")
            st.stop()

# ============================================================================
# CHAT INTERFACE
# ============================================================================
# Create a container for messages to ensure it stays separate from inputs
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.messages:
        avatar = "👨‍⚕️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# Quick Questions (Only show if no messages)
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    faq_cols = st.columns(2)
    questions = [
        "What are common symptoms of the flu?",
        "How can I prevent high blood pressure?",
        "What is the recommended daily intake of water?",
        "Explain the importance of sleep for health."
    ]
    
    for i, q in enumerate(questions):
        if faq_cols[i % 2].button(q, use_container_width=True, key=f"faq_{i}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# Chat Input (Pinned to bottom)
if prompt := st.chat_input("Ask your medical question here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Rerun immediately to show the user's message while the bot thinks
    st.rerun()

# Check if last message is from user and we need to generate response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_q = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant", avatar="👨‍⚕️"):
        with st.spinner("Analyzing and generating response..."):
            response = generate_response(
                user_q,
                st.session_state.model,
                st.session_state.tokenizer
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown(
    '<p class="footer-text">Medical Assistant Chatbot | Fine-tuned with LoRA | Version 2.1</p>',
    unsafe_allow_html=True
)
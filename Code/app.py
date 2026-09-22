import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os 
from dotenv import load_dotenv

st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")
HF_USERNAME = "Akay2026" 

MODELS = {
    "Foundation Model": "unsloth/Llama-3.2-1B-Instruct",
    "LoRA Model": f"{HF_USERNAME}/University-AI-LoRA-Merged",
    "QLoRA Model": f"{HF_USERNAME}/University-AI-LoRA-Merged", # Useing Lora Merged Model due to error in creating QlOra Merged Model. 
    "DPO Model": f"{HF_USERNAME}/University-AI-DPO-Merged",
}

SYSTEM_PROMPT = "You are a helpful University AI Assistant."

load_dotenv()

HF_TOKEN = os.getenv("HF_Read")

@st.cache_resource
def load_model(repo_name):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.2-1B-Instruct", # Tokenizer is same for all models, so we can use the foundation model tokenizer
        token=HF_TOKEN,
    )
    print("Tokenizer loaded.")
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        repo_name,
        token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
    )
    print("Model loaded.")
    model.eval()
    return model, tokenizer



def generate_response_from_loaded_model(
    model,
    tokenizer,
    messages,
    max_new_tokens=200,
    temperature=0.7,
):
    """
    Generate the next assistant response for a conversation.
    """
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()
    return response


st.title("🎓 University AI Assistant")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODELS.keys())
)

if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role":"assistant","content":"Hello! I am your University AI Assistant. How can I help you today?"}
    ]

if selected_model != st.session_state.current_model:
    st.session_state.current_model = selected_model
    st.session_state.messages = [
        {"role":"assistant","content":f"You are now chatting with the {selected_model}."}
    ]

model_path = MODELS[selected_model]

with st.spinner(f"Loading {selected_model}..."):
    model, tokenizer = load_model(model_path)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask your question...")

if question:
    st.session_state.messages.append({"role":"user","content":question})

    with st.chat_message("user"):
        st.markdown(question)

    conversation = [{"role":"system","content":SYSTEM_PROMPT}]
    conversation.extend(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            response = generate_response_from_loaded_model(
                model,
                tokenizer,
                conversation,
            )
            st.markdown(response)

    st.session_state.messages.append(
        {"role":"assistant","content":response}
    )

import streamlit as st
from huggingface_hub import InferenceClient

# 1. Setup - Use st.secrets for deployment, or a text input for testing
st.title("📖 AI Study Buddy")
hf_token = st.sidebar.text_input("Enter HF Token", type="password")

if hf_token:
    client = InferenceClient(api_key=hf_token)
    
    # Model Selection (Mistral and Llama are great for this)
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    tab1, tab2 = st.tabs(["Simple Explainer", "Quiz Generator"])

    with tab1:
        st.header("Explain it Like I'm 5")
        concept = st.text_area("Paste the complex topic here:")
        if st.button("Simplify"):
            prompt = f"Explain this concept to a 10-year-old student using simple analogies: {concept}"
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            st.info(response.choices[0].message.content)

    with tab2:
        st.header("Generate a Quick Quiz")
        notes = st.text_area("Paste your study notes:")
        if st.button("Create Quiz"):
            prompt = f"Based on these notes, generate 3 multiple-choice questions with answers: {notes}"
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800
            )
            st.success(response.choices[0].message.content)
else:
    st.warning("Please enter your Hugging Face Token in the sidebar to start.")
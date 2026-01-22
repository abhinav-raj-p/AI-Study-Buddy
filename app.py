import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2
import io

# --- Page Config ---
st.set_page_config(page_title="AI Study Buddy", page_icon="📖", layout="wide")
st.title("📖 AI-Powered Study Buddy")

# --- Sidebar: Secrets & Settings ---
# When deploying to HF Spaces, use st.secrets["HF_TOKEN"]
with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input("Enter Hugging Face Token", type="password")
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# --- Helper Functions ---
def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def call_ai(prompt, system_message="You are a helpful academic tutor."):
    if not hf_token:
        st.error("Please enter your Hugging Face Token in the sidebar!")
        return None
    
    client = InferenceClient(api_key=hf_token)
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat & Simplify", "📄 PDF Summarizer", "🃏 Flashcards"])

with tab1:
    st.header("Ask a Question")
    user_query = st.text_input("Paste a complex concept or ask a study question:")
    if st.button("Explain Simply"):
        with st.spinner("Simplifying..."):
            ans = call_ai(f"Explain this concept in very simple terms for a student: {user_query}")
            if ans: st.info(ans)

with tab2:
    st.header("PDF Notes Analysis")
    uploaded_file = st.file_uploader("Upload your lecture notes", type="pdf")
    
    if uploaded_file:
        raw_text = extract_text(uploaded_file)
        st.success("PDF Content Extracted!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Summarize Notes"):
                with st.spinner("Summarizing..."):
                    summary = call_ai(f"Summarize these notes into bullet points: {raw_text[:4000]}")
                    st.write(summary)
        with col2:
            question = st.text_input("Ask a question about this PDF:")
            if st.button("Search PDF"):
                with st.spinner("Searching..."):
                    ans = call_ai(f"Based on these notes: {raw_text[:4000]}, answer this: {question}")
                    st.write(ans)

with tab3:
    st.header("Flashcard Generator")
    if 'raw_text' in locals() or 'raw_text' in globals():
        if st.button("Generate Flashcards"):
            with st.spinner("Creating cards..."):
                cards = call_ai(f"Create 5 Flashcard style questions and answers from these notes: {raw_text[:4000]}")
                st.write(cards)
    else:
        st.warning("Please upload a PDF in the Summarizer tab first.")
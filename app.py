import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2
import io

# --- Page Config ---
st.set_page_config(page_title="AI Study Buddy", page_icon="📖", layout="wide")

# --- Token Management (The "Pro" Way) ---
# This looks for the secret in your Hugging Face Space Settings
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = None

# --- Helper Functions ---
def extract_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def call_ai(prompt, system_message="You are a helpful academic tutor who explains things clearly."):
    if not hf_token:
        st.error("HF_TOKEN missing! Please add it to your Space Secrets in Settings.")
        return None
    
    try:
        client = InferenceClient(api_key=hf_token)
        # Using Mistral 7B - perfect for instruction following
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"AI Error: {e}")
        return None

# --- Sidebar UI ---
with st.sidebar:
    st.title("📖 Study Settings")
    st.markdown("---")
    st.info("Your AI Study Buddy uses Mistral-7B to help you ace your exams.")
    if not hf_token:
        st.warning("⚠️ Secrets not configured.")
    else:
        st.success("✅ AI Engine Connected")

# --- Main App Title ---
st.title("📖 AI-Powered Study Buddy")
st.markdown("Upload notes, simplify concepts, and generate study materials instantly.")

# --- Main Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Chat & Simplify", "📄 PDF Summarizer", "🃏 Flashcards"])

with tab1:
    st.header("Ask a Question")
    user_query = st.text_input("Paste a complex concept (e.g., 'What is Cache Coherence?'):")
    if st.button("Explain Simply", key="simplify_btn"):
        if user_query:
            with st.spinner("Simplifying for you..."):
                ans = call_ai(f"Explain this concept in simple terms for a student with analogies: {user_query}")
                if ans:
                    st.markdown("### 💡 Explanation")
                    st.info(ans)
        else:
            st.warning("Please enter a question first!")

with tab2:
    st.header("PDF Notes Analysis")
    uploaded_file = st.file_uploader("Upload your lecture notes", type="pdf")
    
    if uploaded_file:
        # Cache the extraction so it doesn't re-run every click
        if 'pdf_text' not in st.session_state:
            with st.spinner("Extracting text..."):
                st.session_state.pdf_text = extract_text(uploaded_file)
        
        st.success("PDF Content Loaded!")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Summarize Notes"):
                with st.spinner("Summarizing into bullets..."):
                    # Use a slice to stay within model limits
                    context = st.session_state.pdf_text[:4000]
                    summary = call_ai(f"Summarize these notes into clear bullet points: {context}")
                    if summary:
                        st.markdown("### 📝 Summary")
                        st.write(summary)
        
        with col2:
            question = st.text_input("Ask a specific question about these notes:")
            if st.button("Search PDF"):
                if question:
                    with st.spinner("Searching..."):
                        context = st.session_state.pdf_text[:4000]
                        ans = call_ai(f"Based ONLY on these notes: {context}, answer this question: {question}")
                        if ans:
                            st.markdown("### 🔍 Answer")
                            st.write(ans)
                else:
                    st.warning("Enter a question about the PDF.")

with tab3:
    st.header("Flashcard Generator")
    if 'pdf_text' in st.session_state:
        if st.button("Generate 5 Flashcards"):
            with st.spinner("Creating study cards..."):
                context = st.session_state.pdf_text[:4000]
                cards = call_ai(f"Create 5 Flashcard questions and answers (Front/Back style) from these notes: {context}")
                if cards:
                    st.session_state.generated_cards = cards
                    st.markdown("### 🃏 Your Flashcards")
                    st.write(cards)
                    
                    # Add a download button for the student
                    st.download_button(
                        label="📥 Download Flashcards",
                        data=cards,
                        file_name="flashcards.txt",
                        mime="text/plain"
                    )
    else:
        st.warning("⚠️ Please upload a PDF in the 'PDF Summarizer' tab first.")
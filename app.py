import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2

# --- Page Config ---
st.set_page_config(
    page_title="AI Study Buddy", 
    page_icon="🎓", 
    layout="wide",
)

# --- Mobile-Responsive CSS ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #fcfcfc;
    }
    
    /* Responsive adjustment for mobile */
    @media (max-width: 640px) {
        .main .block-container {
            padding: 1rem 1rem;
        }
        h1 {
            font-size: 1.8rem !important;
        }
    }

    /* Buttons styling */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 0.5rem;
        background-color: #4A90E2;
        color: white;
        border: none;
        transition: 0.3s;
    }
    
    /* Card-like containers for mobile */
    .css-1r6slb0, .stVerticalBlock {
        gap: 1.5rem;
    }
    
    /* Hide Sidebar on mobile by default */
    [data-testid="sidebarNavView"] {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Token Management ---
try:
    hf_token = st.secrets["HF_TOKEN"]
except Exception:
    hf_token = None

# --- Helper Functions ---
def extract_text(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        st.error(f"Error: {e}")
        return ""

def call_ai(prompt, system_message="You are a brilliant academic tutor."):
    if not hf_token:
        st.error("Missing HF_TOKEN")
        return None
    try:
        client = InferenceClient(api_key=hf_token)
        model_id = "meta-llama/Llama-3.2-3B-Instruct" 
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

# --- Header Section ---
st.title("🎓 AI Study Buddy")
st.caption("Simplified Learning | PDF Analysis | Flashcards")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["💬 Explain", "📄 Analyze", "🃏 Flashcards"])

with tab1:
    st.markdown("### 💡 Concept Explainer")
    user_query = st.text_input("What is confusing you?", placeholder="e.g. Backpropagation")
    
    if st.button("🚀 Explain for Mobile"):
        if user_query:
            with st.spinner("🧠 Thinking..."):
                ans = call_ai(f"Explain simply with a daily life analogy: {user_query}")
                if ans:
                    st.success("Here is your explanation:")
                    st.write(ans)

with tab2:
    st.markdown("### 📄 Document Analysis")
    uploaded_file = st.file_uploader("Upload Notes", type="pdf")
    
    if uploaded_file:
        if 'pdf_text' not in st.session_state:
            st.session_state.pdf_text = extract_text(uploaded_file)
        
        st.success("✅ File Ready")

        # Layout shifts: Side-by-side on PC, Stacked on Mobile
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📝 Summarize"):
                with st.spinner("Summarizing..."):
                    summary = call_ai(f"Summary of: {st.session_state.pdf_text[:4000]}")
                    st.markdown(summary)
        
        with col2:
            q = st.text_input("Ask about PDF:")
            if st.button("🔍 Search"):
                with st.spinner("Finding..."):
                    ans = call_ai(f"Context: {st.session_state.pdf_text[:4000]} Query: {q}")
                    st.info(ans)

with tab3:
    st.markdown("### 🃏 Quiz Generator")
    if 'pdf_text' in st.session_state:
        if st.button("🎯 Generate Set"):
            with st.spinner("Generating..."):
                cards = call_ai(f"Create 5 flashcards from: {st.session_state.pdf_text[:4000]}")
                st.write(cards)
                st.download_button("📥 Save to Phone", cards, "flashcards.txt")
    else:
        st.warning("Upload a PDF first.")

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    st.write("Current Model: Llama 3.2")
    if hf_token:
        st.success("Connected")
    else:
        st.error("Token Missing")
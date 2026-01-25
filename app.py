import streamlit as st
from huggingface_hub import InferenceClient
import PyPDF2
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(page_title="AI Study Buddy", page_icon="🎓", layout="wide")

# --- Custom JS for Copy to Clipboard ---
def copy_to_clipboard(text):
    # This creates a hidden button that triggers the browser's copy command
    escaped_text = text.replace("`", "\\`").replace("'", "\\'").replace('"', '\\"')
    copy_js = f"""
    <script>
    function copyFunction() {{
      const text = `{escaped_text}`;
      navigator.clipboard.writeText(text).then(() => {{
        alert("Copied to clipboard!");
      }});
    }}
    </script>
    <button onclick="copyFunction()" style="
        background-color: #4A90E2; 
        color: white; 
        border: none; 
        padding: 10px 20px; 
        border-radius: 10px; 
        cursor: pointer;
        width: 100%;
        font-weight: bold;
        margin-top: 10px;">
        📋 Copy to Clipboard
    </button>
    """
    components.html(copy_js, height=70)

# --- Mobile Responsive CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #fcfcfc; }
    @media (max-width: 640px) {
        .main .block-container { padding: 1rem; }
        h1 { font-size: 1.8rem !important; }
    }
    .stButton>button { width: 100%; border-radius: 12px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- Token Management ---
hf_token = st.secrets.get("HF_TOKEN")

# --- AI & PDF Logic ---
def call_ai(prompt):
    if not hf_token: return "Token Error"
    client = InferenceClient(api_key=hf_token)
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[{"role": "system", "content": "You are a concise academic tutor."},
                      {"role": "user", "content": prompt}],
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e: return f"Error: {e}"

def extract_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "".join([p.extract_text() or "" for p in reader.pages])

# --- Main UI ---
st.title("🎓 AI Study Buddy")
tab1, tab2, tab3 = st.tabs(["💬 Explain", "📄 Analyze", "🃏 Flashcards"])

with tab1:
    st.subheader("💡 Simple Explainer")
    query = st.text_input("Enter concept:", placeholder="e.g. Virtual Memory")
    if st.button("🚀 Explain"):
        with st.spinner("Processing..."):
            ans = call_ai(f"Explain this concept with a simple analogy: {query}")
            st.info(ans)
            copy_to_clipboard(ans)

with tab2:
    st.subheader("📄 PDF Analyzer")
    file = st.file_uploader("Upload Notes", type="pdf")
    if file:
        text = extract_text(file)
        
        # Context Limit Indicator
        usage = min(len(text), 4500)
        st.write(f"Analyzing {usage} characters of your notes:")
        st.progress(usage / 4500)
        
        if st.button("📝 Summarize"):
            summary = call_ai(f"Summarize these notes: {text[:4500]}")
            st.write(summary)
            copy_to_clipboard(summary)

with tab3:
    st.subheader("🃏 Quick Quiz")
    if 'text' in locals():
        if st.button("🎯 Generate Flashcards"):
            cards = call_ai(f"Create 5 flashcards from: {text[:4500]}")
            st.write(cards)
            copy_to_clipboard(cards)
    else:
        st.warning("Upload a PDF first!")

# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    st.write("✅ Ready to Study" if hf_token else "❌ Token Missing")
    st.info("Tip: Use 'Copy to Clipboard' to save your AI summaries directly to your phone's notes app!")
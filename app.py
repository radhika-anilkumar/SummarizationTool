import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import tempfile
 
import streamlit as st
 
st.set_page_config(
    page_title="PDF Summarizer",  # Browser tab title
    page_icon=":rocket:",          # Optional favicon emoji or file path
)
 
 
# Load summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
 
summarizer = load_summarizer()
 
# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
 
# Function to chunk text (because Hugging Face models have a token limit)
def split_text(text, max_chunk=1000):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk)
            current_chunk = para + "\n"
    chunks.append(current_chunk)
    return chunks
 
# Streamlit UI
st.title("PDF Summarizer")
# st.write("If you're seeing this, Streamlit UI is working fine.")
 
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
 
if uploaded_file:
    st.info("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(uploaded_file)
   
    if raw_text.strip():
        st.success("Text extracted. Generating summary...")
 
        chunks = split_text(raw_text)
        summary = ""
        for chunk in chunks:
            summarized = summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            summary += summarized + " "
 
        st.subheader("📝 Summary")
        st.write(summary)
    else:
        st.warning("No text could be extracted from this PDF.")
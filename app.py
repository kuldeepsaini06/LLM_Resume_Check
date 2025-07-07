import streamlit as st
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer, util
import tempfile

# Load pre-trained model (e.g., BERT or all-MiniLM)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Function to extract text from PDF using pdfplumber
# Ensure you have pdfplumber installed: pip install pdfplumber
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return " ".join(page.extract_text() or "" for page in pdf.pages)

def get_similarity_score(resume_text, jd_text):
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    jd_embedding = model.encode(jd_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
    return round(score * 100, 2)

def extract_keywords(text):
    words = text.lower().split()
    return set([word.strip(",.:-") for word in words if len(word) > 3])

# Streamlit UI
st.title("ATS Resume Checker based on Job Description")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (TXT)", type=["txt", "docx"])

if resume_file and jd_file:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = jd_file.read().decode("utf-8")

    score = get_similarity_score(resume_text, jd_text)

    st.markdown(f"Match Score: **{score}%**")

    # Keyword analysis
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    missing_keywords = jd_keywords - resume_keywords

    st.markdown("Missing Keywords:")
    st.write(", ".join(sorted(missing_keywords)) if missing_keywords else "All relevant keywords are present.")


st.set_page_config(page_title="Resume Checker", layout="wide")  
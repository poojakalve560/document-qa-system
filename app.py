import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def create_embeddings(chunks):
    return model.encode(chunks)


def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def search(query, index, chunks):
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=3)
    return [chunks[i] for i in I[0]]


if uploaded_file is not None:
    text = extract_text(uploaded_file)

    if text.strip() == "":
        st.error("Could not extract text from PDF")
    else:
        st.success("PDF processed")

        chunks = chunk_text(text)
        embeddings = create_embeddings(chunks)
        index = store_embeddings(embeddings)

        query = st.text_input("Ask a question")

        if query:
            results = search(query, index, chunks)

            final_answer = " ".join(results).replace("\n", " ")

            # remove extra spaces
            final_answer = " ".join(final_answer.split())

            st.subheader("Answer")
            st.success(final_answer)

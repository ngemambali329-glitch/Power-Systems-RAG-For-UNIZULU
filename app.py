import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize model for embedding
@st.cache(allow_output_mutation=True)
def load_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

# Initialize text generator
@st.cache(allow_output_mutation=True)
def load_text_generator():
    generator = pipeline("text-generation", model="gpt2")
    return generator

embedding_model = load_embedding_model()
text_generator = load_text_generator()

st.title("Local RAG Learning Assistant (No API key required)")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        documents.append(text)
    # Embed documents
    st.session_state["documents"] = documents
    embeddings = embedding_model.encode(documents, convert_to_numpy=True)
    # Build FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    st.session_state["faiss_index"] = index
    st.success("Knowledge base built with uploaded PDFs!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Input question
question = st.text_input("Enter your question:")

# Display previous chats
if st.session_state["chat_history"]:
    st.subheader("Previous Conversations")
    for q, a in st.session_state["chat_history"]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

# Ask button
if st.button("Ask") and question:
    if "faiss_index" in st.session_state and "documents" in st.session_state:
        # Embed question
        question_embedding = embedding_model.encode([question], convert_to_numpy=True)
        # Search similar documents
        D, I = st.session_state["faiss_index"].search(question_embedding, k=3)
        retrieved_docs = [st.session_state["documents"][idx] for idx in I[0]]

        # Prepare input for generator
        context = "\n---\n".join(retrieved_docs)
        prompt = f"Context:\n{context}\nQuestion: {question}\nAnswer:"

        # Generate answer
        generated = text_generator(prompt, max_length=200, do_sample=True)[0]['generated_text']
        answer = generated.split("Answer:")[-1].strip()

        # Save chat history
        st.session_state["chat_history"].append((question, answer))
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please upload PDFs to build the knowledge base.")

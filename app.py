import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize session state
if "knowledge_base" not in st.session_state:
    st.session_state["knowledge_base"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

st.title("RAG-based Learning Assistant")

# Upload PDFs/Documents
uploaded_files = st.file_uploader("Upload PDFs or Text Documents", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    docs = []
    for uploaded_file in uploaded_files:
        # Load PDF
        loader = PyPDFLoader(uploaded_file)
        pages = loader.load()
        docs.extend(pages)

    # Create or update knowledge base
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(docs, embeddings)
    st.session_state["knowledge_base"] = knowledge_base
    st.success("Knowledge base updated with uploaded documents!")

# Chat input
question = st.text_input("Enter your question:")

# Display previous chats
if st.session_state["chat_history"]:
    st.subheader("Previous Conversations")
    for i, (q, a) in enumerate(st.session_state["chat_history"]):
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Assistant:** {a}")

# Generate response
if st.button("Ask") and question:
    if st.session_state["knowledge_base"]:
        # Set up retriever
        retriever = st.session_state["knowledge_base"].as_retriever(search_type="similarity", search_kwargs={"k": 3})
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=retriever,
        )
        answer = qa_chain.run({"question": question, "chat_history": st.session_state["chat_history"]})
        st.session_state["chat_history"].append((question, answer))
        st.markdown(f"**Answer:** {answer}")
    else:
        st.warning("Please upload documents to build the knowledge base.")

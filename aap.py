import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

st.set_page_config(page_title="RAG ChatGPT for arXiv")

st.title("🤖 Ask Your Paper!")
st.write("Upload a research paper PDF from arXiv and ask questions.")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading the paper..."):
        # Step 1: Load PDF
        loader = PyPDFLoader(uploaded_file)
        documents = loader.load()

        # Step 2: Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Step 3: Embed and store in memory
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Step 4: Add memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        st.success("Ready! Ask your questions below ⬇️")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("💬 Ask a question:")
        if query:
            result = qa_chain.run(query)
            st.session_state.chat_history.append((query, result))

        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")

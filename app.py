# ---------------- ENV SETUP ----------------
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import shutil

# ---------------- LANGCHAIN IMPORTS ----------------
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


# ---------------- SESSION STATE ----------------
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- UI ----------------
st.title("🔎 AI Document Assistant")


# ---------------- CLEAN OLD DATA ----------------
def clean_old_data():
    if os.path.exists("./doc_files/"):
        shutil.rmtree("./doc_files/")

    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")


# ---------------- DOCUMENT PROCESSING ----------------
def process_documents(path):

    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(docs)

    emb = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    # Create fresh FAISS DB
    vector_db = FAISS.from_documents(docs, emb)

    # Save index
    vector_db.save_local("faiss_index")

    # Load Groq LLM
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", streaming=True)
    except:
        llm = ChatGroq(model="llama-3.1-8b-instant", streaming=True)

    # ---------------- TOOL ----------------
    @tool
    def retrieve_context(query: str):
        """
        Retrieve relevant text chunks from uploaded PDF using similarity search.

        This tool is used to fetch context before answering any question.
        """

        docs = vector_db.similarity_search(query=query, k=6)

        if not docs:
            return "No relevant context found."

        context = ""
        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            context += f"[Page {page}] {doc.page_content}\n\n"

        return context


    # ---------------- SYSTEM PROMPT ----------------
    system_prompt = """
You are a document QA assistant.

You MUST ALWAYS use retrieve_context before answering.

Rules:
- Always call retrieve_context first
- Answer ONLY using retrieved context
- If context is empty or irrelevant, say "Not found in document"
- Always include page references like [Page X]
"""


    # Memory
    memory = InMemorySaver()

    # Agent (FIXED)
    agent = create_react_agent(
        model=llm,
        tools=[retrieve_context],
        prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True


# ---------------- FILE UPLOAD ----------------
if not st.session_state.document_uploaded:

    uploaded = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded:

        with st.spinner("Processing..."):

            # CLEAN OLD DATA FIRST (IMPORTANT FIX)
            clean_old_data()

            path = "./doc_files/"
            os.makedirs(path, exist_ok=True)

            for file in uploaded:
                file_path = os.path.join(path, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            process_documents(path)

            # Reset chat on new upload
            st.session_state.messages = []

            st.rerun()


# ---------------- CHAT UI ----------------
if st.session_state.document_uploaded and st.session_state.agent:

    if st.button("🔄 Reset Chat"):
        st.session_state.messages = []

    # show history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖") \
            .markdown(msg["content"])

    query = st.chat_input("Ask anything about your documents...")

    if query:

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        st.chat_message("user", avatar="🧑").markdown(query)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):

                response = st.session_state.agent.invoke(
                    {"messages": [("user", query)]},
                    {"configurable": {"thread_id": "rag_chat"}}
                )

                answer = response["messages"][-1].content
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
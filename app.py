import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openrouter import ChatOpenRouter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda

# Vectorless IMPORTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="PDF Chat (Vectorless RAG)", layout="wide")

# Sidebar
st.sidebar.title("🔧 Settings")
API_KEY = st.sidebar.text_input("🔑 OpenRouter API Key", type="password")
session_id = st.sidebar.text_input("💬 Session ID", value="default_session")

if 'store' not in st.session_state:
    st.session_state.store = {}

# Chat history viewer
with st.sidebar.expander("📜 Chat History", expanded=True):
    session_history = st.session_state.store.get(session_id, ChatMessageHistory())
    for msg in session_history.messages:
        if msg.type == "human":
            st.markdown(f"**🧍 You:** {msg.content}")
        elif msg.type == "ai":
            st.markdown(f"**🤖 Assistant:** {msg.content}")
    if st.button("🗑️ Clear This Session"):
        st.session_state.store[session_id] = ChatMessageHistory()
        st.rerun()

# Main UI
st.title("📚 Chat With Your PDFs (Vectorless RAG)")
uploaded_files = st.file_uploader("📤 Upload PDF(s)", type="pdf", accept_multiple_files=True)

# ---------------- VECTORLESS RETRIEVER ---------------- #

class VectorlessRetriever:
    def __init__(self, documents):
        self.texts = [doc.page_content for doc in documents]
        self.docs = documents

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self.texts)

    def get_relevant_documents(self, query, k=2):
        if not query.strip():
            return self.docs[:k]

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        if len(similarities) == 0 or np.sum(similarities) == 0:
            return self.docs[:k]

        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [self.docs[i] for i in top_k_idx]

# ----------------------------------------------------- #

if API_KEY:
    try:
        llm = ChatOpenRouter(
            model="meta-llama/llama-3-8b-instruct",
            api_key=API_KEY
        )

        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                temp_path = "./temp.pdf"
                with open(temp_path, "wb") as file:
                    file.write(uploaded_file.getvalue())

                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())

            # Chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)

            # VECTORLESS RETRIEVER
            retriever = VectorlessRetriever(splits)

            def retrieve_docs(x):
                if isinstance(x, dict):
                    query = x.get("input", "")
                else:
                    query = x

                docs = retriever.get_relevant_documents(query)

                MAX_CHARS = 3000
                total = 0
                limited_docs = []

                for doc in docs:
                    if total + len(doc.page_content) > MAX_CHARS:
                        break
                    limited_docs.append(doc)
                    total += len(doc.page_content)

                return limited_docs

            wrapped_retriever = RunnableLambda(retrieve_docs)

            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Rephrase the user query into a standalone question."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            history_aware_retriever = create_history_aware_retriever(
                llm, wrapped_retriever, contextualize_q_prompt
            )

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "Answer ONLY from the provided context.\n"
                 "If not found, say: 'I don't know the answer of this question. Ask me about your PDF file only.'\n\n"
                 "{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])

            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            def get_session_history(session: str) -> BaseChatMessageHistory:
                if session not in st.session_state.store:
                    st.session_state.store[session] = ChatMessageHistory()
                return st.session_state.store[session]

            conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            user_input = st.chat_input("💬 Ask something about your PDFs...")

            if user_input:
                st.chat_message("user").write(user_input)

                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": session_id}},
                )

                st.chat_message("assistant").write(response['answer'])

    except Exception as e:
        import traceback
        st.error(f"Error: {str(e)}")
        st.text(traceback.format_exc())
else:
    st.warning("Please enter your OpenRouter API Key.")
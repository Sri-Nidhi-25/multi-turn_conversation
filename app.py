import os
import streamlit as st
from dotenv import load_dotenv
from typing import List

# Import all the advanced components from your working demo
from demo_10_turns import (
    EnhancedMemory,
    detect_topic_switch,
    MultiSourceRetriever,
    parse_answer,
    # We will NOT import build_rag_chain from demo because it uses hardcoded TXT_PATHS.
    # Instead we'll define our own that accepts dynamic filenames.
)

# LangChain imports for building the chain
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please add it.")
    st.stop()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")


# ---------------------------------------------------------------------------
# Advanced build_rag_chain – uses per‑source retrievers and explicit allowed filenames
# ---------------------------------------------------------------------------
def build_rag_chain(splits, filenames, api_key):
    """
    Builds a retrieval chain where each source file has its own retriever,
    and the system prompt lists the exact allowed filenames.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name="dynamic_rag"
    )

    # Create one retriever per source file (with metadata filtering)
    source_retrievers = {}
    for fname in filenames:
        source_retrievers[fname] = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "filter": {"source": fname}}
        )

    default_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6}
    )

    retriever = MultiSourceRetriever(
        source_retrievers=source_retrievers,
        default_retriever=default_retriever,
    )

    llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")

    # Prompt to reformulate question with history
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, reformulate it as a standalone question. Do NOT answer."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Build list of allowed filenames for the prompt
    allowed_list = ", ".join(filenames)

    system_prompt = (
        "You are a document support assistant with access to multiple documents.\n\n"
        f"The ONLY valid source filenames are: {allowed_list}.\n"
        "RULES:\n"
        "1. Answer ONLY from the provided context.\n"
        "2. Cite every fact using the format: [Section: <topic/section name>, Source: <filename>].\n"
        "   Example: [Section: vacation policy, Source: policies.txt]\n"
        "   The <filename> MUST be one of the allowed filenames listed above.\n"
        "3. Place the citation immediately after the fact – inside the answer text.\n"
        "4. Do NOT repeat information already covered in 'Recent answers'.\n"
        "5. Be concise (3–5 sentences).\n"
        "6. If the answer is not in the documents, say so.\n\n"
        "Recent answers already given (do NOT repeat):\n{recent_summary}\n\n"
        "Context from documents:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Document Support Assistant", page_icon="📄")
    st.title("📄 Document-Aware Support Assistant")
    st.write("Upload **.txt** files, then chat. The assistant remembers context and cites sources with section names.")

    session_id = st.text_input("Session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}
    if "memory" not in st.session_state:
        st.session_state.memory = EnhancedMemory()
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []
    if "uploaded_filenames" not in st.session_state:
        st.session_state.uploaded_filenames = []

    # File upload
    uploaded_files = st.file_uploader(
        "Upload .txt file(s)", type="txt", accept_multiple_files=True
    )
    if uploaded_files and st.button("Index documents"):
        documents = []
        filenames = []
        for uf in uploaded_files:
            tmp = f"./temp_{uf.name}"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(uf.getvalue().decode("utf-8"))
            loader = TextLoader(tmp, encoding="utf-8")
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = uf.name
            documents.extend(docs)
            filenames.append(uf.name)
            os.remove(tmp)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        with st.spinner("Building vector index and retrieval chain…"):
            st.session_state.rag_chain = build_rag_chain(splits, filenames, GROQ_API_KEY)
            st.session_state.uploaded_filenames = filenames
        st.success(f"Indexed {len(splits)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.rag_chain is None:
        st.info("Please upload and index at least one .txt file to start chatting.")
        return

    def get_session_history(sid: str) -> BaseChatMessageHistory:
        if sid not in st.session_state.store:
            st.session_state.store[sid] = ChatMessageHistory()
        return st.session_state.store[sid]

    conversational_chain = RunnableWithMessageHistory(
        st.session_state.rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Display chat history
    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask something about your documents…")
    if user_input:
        memory = st.session_state.memory
        allowed_sources = set(st.session_state.uploaded_filenames)

        # Topic switch detection
        if detect_topic_switch(session_id, user_input, memory):
            hist = get_session_history(session_id)
            if len(hist.messages) > 4:
                hist.messages = hist.messages[-4:]
            st.toast("📌 Topic switch detected — context reset to recent messages.")

        # Repetition warning
        if memory.is_repeated_question(session_id, user_input):
            st.toast("ℹ️ You may have asked something similar before — I'll add only new info.")

        recent_summary = memory.get_recent_answers_summary(session_id)

        # Add user message to display
        st.session_state.chat_display.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = conversational_chain.invoke(
                    {"input": user_input, "recent_summary": recent_summary},
                    config={"configurable": {"session_id": session_id}},
                )
            raw_answer = response["answer"]
            context_docs = response.get("context", [])

            # Parse answer to remove citations and extract source->sections
            clean_answer, source_sections = parse_answer(raw_answer, context_docs)

            st.markdown(clean_answer)

            # Display sources
            if source_sections:
                st.markdown("**Sources:**")
                for src, sections in source_sections.items():
                    if sections:
                        sections_str = ", ".join(sorted(sections))
                        st.markdown(f"- {src} (sections: {sections_str})")
                    else:
                        st.markdown(f"- {src}")
            else:
                st.markdown("*No sources identified.*")

        # Store the clean answer (without citations) in chat history and memory
        st.session_state.chat_display.append({"role": "assistant", "content": clean_answer})
        memory.add_interaction(session_id, user_input, clean_answer)


if __name__ == "__main__":
    main()
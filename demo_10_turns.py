# import os
# import re
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory

# from app import EnhancedMemory, detect_topic_switch

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not set in .env")

# # ---------------------------------------------------------------------------
# # CONFIGURATION – List all your .txt files here
# # ---------------------------------------------------------------------------
# TXT_PATHS = [
#     "policies.txt",
#     "cat-facts.txt",
#     "dog-facts.txt",
#     # add as many as you need
# ]
# SESSION_ID = "demo_session"

# # ---------------------------------------------------------------------------
# # 1. Load & chunk ALL text files
# # ---------------------------------------------------------------------------
# print("Loading text files...")
# all_documents = []

# for file_path in TXT_PATHS:
#     if not os.path.exists(file_path):
#         print(f"  Warning: {file_path} not found – skipping")
#         continue
#     loader = TextLoader(file_path, encoding="utf-8")
#     docs = loader.load()
#     for d in docs:
#         d.metadata["source"] = os.path.basename(file_path)   # exact filename
#     all_documents.extend(docs)
#     print(f"  → Loaded {file_path} ({len(docs)} documents)")

# if not all_documents:
#     raise ValueError("No valid .txt files found in TXT_PATHS")

# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = splitter.split_documents(all_documents)
# print(f"  → Total chunks from all files: {len(splits)}")

# # ---------------------------------------------------------------------------
# # 2. Build vector store + retriever (with MMR for diversity across files)
# # ---------------------------------------------------------------------------
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectorstore = Chroma.from_documents(splits, embeddings, collection_name="multi_file_demo")

# # Use MMR retriever to get diverse chunks from different documents
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 6,              # retrieve 6 chunks total
#         "fetch_k": 20,      # fetch 20 initially to choose from
#         "lambda_mult": 0.5  # balance between similarity and diversity
#     }
# )

# # ---------------------------------------------------------------------------
# # 3. Build the RAG chain (explicit filename citation)
# # ---------------------------------------------------------------------------
# llm = ChatGroq(model="llama-3.1-8b-instant")

# contextualize_q_prompt = ChatPromptTemplate.from_messages([
#     ("system", "Given a chat history and the latest user question, reformulate it as a standalone question. Do NOT answer."),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])
# history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# system_prompt = (
#     "You are a document support assistant.\n\n"
#     "RULES:\n"
#     "1. Use ONLY the provided context (from multiple documents).\n"
#     "2. Cite every fact using the **exact filename** from the context, e.g., [Source: policies.txt] or [Source: cat-facts.txt]. Do NOT invent or capitalise filenames.\n"
#     "3. Recent answers are listed under 'Recent answers'. Do NOT repeat information already provided.\n"
#     "4. If answer not in documents, say so.\n"
#     "5. Be concise (3–5 sentences).\n\n"
#     "Recent answers already given:\n{recent_summary}\n\n"
#     "Context:\n{context}"
# )
# qa_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt),
#     MessagesPlaceholder("chat_history"),
#     ("human", "{input}"),
# ])

# question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
# rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# store = {}
# def get_session_history(sid: str):
#     if sid not in store:
#         store[sid] = ChatMessageHistory()
#     return store[sid]

# conversational_rag_chain = RunnableWithMessageHistory(
#     rag_chain,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history",
#     output_messages_key="answer",
# )

# # ---------------------------------------------------------------------------
# # 4. 10-turn question set (tests multi‑file retrieval)
# # ---------------------------------------------------------------------------
# questions = [
#     "What is the vacation policy?",
#     "How many days of paid leave are allowed per year?",
#     "Can I carry over unused vacation days to the next year?",
#     "How similar are dogs and cats?",
#     "Does the company provide equipment for remote work and its policy if any?",
#     "I already asked about vacation days — are there any blackout dates?",
#     "What are the differences between dogs and cats?",
#     "How long is the probation period for new hires?",
#     "You mentioned vacation days earlier — do they expire at year end?",
#     "How do I differentiate between dogs and cats?",
# ]

# # ---------------------------------------------------------------------------
# # 5. Run the demo with source extraction from answer text
# # ---------------------------------------------------------------------------
# memory = EnhancedMemory()
# print("\n" + "="*65)
# print("  10-Turn Conversational RAG Demo (multiple .txt files)")
# print("="*65)

# for i, q in enumerate(questions, 1):
#     print(f"\n{'─'*65}\nTurn {i:02d}\n{'─'*65}\nQ: {q}")

#     if detect_topic_switch(SESSION_ID, q, memory):
#         print("[⚡ Topic switch detected – trimming history]")
#         hist = get_session_history(SESSION_ID)
#         if len(hist.messages) > 4:
#             hist.messages = hist.messages[-4:]

#     if memory.is_repeated_question(SESSION_ID, q):
#         print("[🔁 Similar question detected – model will avoid repetition]")

#     recent_summary = memory.get_recent_answers_summary(SESSION_ID, n=3)
#     response = conversational_rag_chain.invoke(
#         {"input": q, "recent_summary": recent_summary},
#         config={"configurable": {"session_id": SESSION_ID}},
#     )
#     answer = response["answer"]
#     print(f"\nA: {answer}")

#     # --- Extract cited sources from the answer text (filenames) ---
#     # Pattern matches [Source: filename] – filename can contain dots, letters, hyphens, underscores
#     cited_sources = set()
#     matches = re.findall(r'\[Source:\s*([^\],\n]+)', answer)
#     for match in matches:
#         source = match.strip().rstrip(']').rstrip(',')
#         if source:
#             cited_sources.add(source)

#     if cited_sources:
#         print("\nSources:")
#         for src in sorted(cited_sources):
#             print(f"  - {src}")

#     memory.add_interaction(SESSION_ID, q, answer)

# print("\n" + "="*65)
# print("Demo complete.")

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import List


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in .env")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TXT_PATHS = [
    "policies.txt",
    "cat-facts.txt",
    "dog-facts.txt",
]
SESSION_ID = "demo_session"

ALLOWED_SOURCES = {os.path.basename(p) for p in TXT_PATHS} 


# ---------------------------------------------------------------------------
# EnhancedMemory – tracks Q&A history
# ---------------------------------------------------------------------------
class EnhancedMemory:
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self._store: dict[str, list[dict]] = {}

    def add_interaction(self, session_id: str, question: str, answer: str):
        if session_id not in self._store:
            self._store[session_id] = []
        entry = {
            "question": question,
            "answer": answer,
            "keywords": self._extract_keywords(question),
        }
        self._store[session_id].append(entry)
        if len(self._store[session_id]) > self.max_history:
            self._store[session_id] = self._store[session_id][-self.max_history:]

    def get_history(self, session_id: str) -> list[dict]:
        return self._store.get(session_id, [])

    def is_repeated_question(self, session_id: str, question: str, threshold: float = 0.55) -> bool:
        new_kw = self._extract_keywords(question)
        if not new_kw:
            return False
        for entry in self.get_history(session_id):
            old_kw = entry["keywords"]
            if not old_kw:
                continue
            overlap = len(new_kw & old_kw) / len(new_kw | old_kw)
            if overlap >= threshold:
                return True
        return False

    def get_recent_answers_summary(self, session_id: str, n: int = 3) -> str:
        history = self.get_history(session_id)
        if not history:
            return "No previous answers in this session."
        recent = history[-n:]
        lines = []
        for i, entry in enumerate(recent, 1):
            snippet = entry["answer"][:200].replace("\n", " ")
            if len(entry["answer"]) > 200:
                snippet += "…"
            lines.append(f"{i}. Q: {entry['question']}\n   A: {snippet}")
        return "\n".join(lines)

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        stopwords = {"a","an","the","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","could","should","may","might","shall","can","i","you","he","she","it","we","they","what","which","who","whom","how","when","where","why","that","this","and","or","but","if","in","on","at","to","for","of","with","about","already","asked","me","tell","mentioned","earlier"}
        words = text.lower().split()
        return {w.strip("?.,!:;\"'") for w in words if w.strip("?.,!:;\"'") not in stopwords and len(w) > 2}

# ---------------------------------------------------------------------------
# Topic switch detection (keyword‑based)
# ---------------------------------------------------------------------------
def detect_topic_switch(session_id: str, new_question: str, memory: EnhancedMemory, switch_threshold: float = 0.15) -> bool:
    history = memory.get_history(session_id)
    if not history:
        return False
    last_keywords = history[-1]["keywords"]
    new_keywords = memory._extract_keywords(new_question)
    if not last_keywords or not new_keywords:
        return False
    overlap = len(new_keywords & last_keywords) / len(new_keywords | last_keywords)
    return overlap < switch_threshold

# ---------------------------------------------------------------------------
# 1. Load & chunk ALL text files
# ---------------------------------------------------------------------------
print("Loading text files...")
all_documents = []

for file_path in TXT_PATHS:
    if not os.path.exists(file_path):
        print(f"  Warning: {file_path} not found – skipping")
        continue
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = os.path.basename(file_path)
    all_documents.extend(docs)
    print(f"  → Loaded {file_path} ({len(docs)} documents)")

if not all_documents:
    raise ValueError("No valid .txt files found in TXT_PATHS")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(all_documents)
print(f"  → Total chunks from all files: {len(splits)}")

# ---------------------------------------------------------------------------
# 2. Build per‑source retrievers (ensures multi‑file answers)
# ---------------------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(splits, embeddings, collection_name="multi_file_demo")

source_retrievers = {}
for file_path in TXT_PATHS:
    fname = os.path.basename(file_path)
    source_retrievers[fname] = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "filter": {"source": fname}}
    )

default_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.6}
)

class MultiSourceRetriever(BaseRetriever):
    source_retrievers: dict
    default_retriever: BaseRetriever
    per_source_k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        seen_content = set()
        merged = []
        for fname, ret in self.source_retrievers.items():
            try:
                docs = ret.invoke(query)
                for doc in docs:
                    key = doc.page_content.strip()[:200]
                    if key not in seen_content:
                        seen_content.add(key)
                        merged.append(doc)
            except Exception:
                pass
        if not merged:
            for doc in self.default_retriever.invoke(query):
                key = doc.page_content.strip()[:200]
                if key not in seen_content:
                    seen_content.add(key)
                    merged.append(doc)
        return merged

retriever = MultiSourceRetriever(
    source_retrievers=source_retrievers,
    default_retriever=default_retriever,
)

# ---------------------------------------------------------------------------
# 3. Build the RAG chain – with inline citations (section + source)
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant")

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, reformulate it as a standalone question. Do NOT answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# --- NEW PROMPT with inline citations (section + source) ---
system_prompt = (
    "You are a document support assistant with access to multiple documents.\n\n"
    "RULES:\n"
    "1. Answer ONLY from the provided context.\n"
    "2. Cite every fact using the format: [Section: <topic/section name>, Source: <filename>].\n"
    "   Example: [Section: vacation policy, Source: policies.txt]\n"
    "   Use a short descriptive section name (e.g., 'vacation policy', 'dog facts', 'remote work policy').\n"
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

store = {}
def get_session_history(sid: str):
    if sid not in store:
        store[sid] = ChatMessageHistory()
    return store[sid]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# ---------------------------------------------------------------------------
# 4. Helper: parse answer (keep citations, extract source filenames)
# ---------------------------------------------------------------------------

def parse_answer(raw: str, context_docs: list):
    """
    Removes all inline citations of the form [Section: ..., Source: ...]
    Returns clean answer text and a dict {source_filename: set(section_names)}.
    """
    import re
    from collections import defaultdict

    # Extract (section, source) pairs – matches everything up to the last ', Source:'
    extract_pattern = r'\[Section:\s*(.*?),\s*Source:\s*([^\],\n]+)\]'
    matches = re.findall(extract_pattern, raw, re.IGNORECASE | re.DOTALL)

    source_sections = defaultdict(set)
    for section, source in matches:
        source = source.strip().rstrip(']').rstrip(',')
        section = section.strip()
        if source and section:
            source_sections[source].add(section)

    # Remove the entire citation block – handles any internal commas
    clean = re.sub(r'\[[^\]]*?Section:.*?Source:[^\]]*\]', '', raw, flags=re.IGNORECASE | re.DOTALL)
    clean = re.sub(r'\s+', ' ', clean).strip()
    clean = clean.replace(' .', '.').replace(' ,', ',')

    # Fallback: if no citations were found, use metadata from retrieved docs
    if not source_sections:
        for doc in context_docs:
            src = doc.metadata.get("source", "")
            if src:
                source_sections[src] = set()

    return clean, source_sections

# ---------------------------------------------------------------------------
# 5. 10-turn question set
# ---------------------------------------------------------------------------
questions = [
    "What is the vacation policy?",
    "How many days of paid leave are allowed per year?",
    "Can I carry over unused vacation days to the next year?",
    "What are the uniqueness characteristics of dogs?",
    "Does the company provide equipment for remote work and its policy if any?",
    "I already asked about vacation days — are there any blackout dates?",
    "Tell me about cats?",
    "How long is the probation period for new hires?",
    "You mentioned vacation days earlier — do they expire at year end?",
    "Uniqueness of dogs?",
]

# ---------------------------------------------------------------------------
# 6. Run demo
# ---------------------------------------------------------------------------
memory = EnhancedMemory()
print("\n" + "="*65)
print("  10-Turn Conversational RAG Demo (multiple .txt files)")
print("="*65)

for i, q in enumerate(questions, 1):
    print(f"\n{'─'*65}\nTurn {i:02d}\n{'─'*65}")
    print(f"Q: {q}")

    if detect_topic_switch(SESSION_ID, q, memory):
        print("[⚡ Topic switch detected – trimming history]")
        hist = get_session_history(SESSION_ID)
        if len(hist.messages) > 4:
            hist.messages = hist.messages[-4:]

    if memory.is_repeated_question(SESSION_ID, q):
        print("[🔁 Similar question detected – model will avoid repetition]")

    recent_summary = memory.get_recent_answers_summary(SESSION_ID, n=3)
    response = conversational_rag_chain.invoke(
        {"input": q, "recent_summary": recent_summary},
        config={"configurable": {"session_id": SESSION_ID}},
    )

    raw_answer = response["answer"]
    context_docs = response.get("context", [])
    
    clean_answer, source_sections = parse_answer(raw_answer, context_docs)

    print(f"\nA: {clean_answer}")
    print("\nSources:")
    if source_sections:
        for src, sections in source_sections.items():
            if sections:
                sections_str = ", ".join(sorted(sections))
                print(f"  - {src} (sections: {sections_str})")
            else:
                print(f"  - {src}")
    else:
        print("  - (no sources identified)")
    
    memory.add_interaction(SESSION_ID, q, clean_answer)

print("\n" + "="*65)
print("Demo complete.")
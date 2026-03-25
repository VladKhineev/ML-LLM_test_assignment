import os
import re
import json

from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:4b")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_HOST, temperature=0, num_ctx=4096)
splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

prompt = PromptTemplate.from_template("""Ты — ассистент для извлечения данных из юридических документов.
Отвечай строго на основе предоставленного контекста, не придумывай ничего.

Контекст:
{context}

Вопрос: {question}

Ответь ТОЛЬКО в формате JSON, где ключ — краткое название запрошенных данных,
а значение — точный текст из документа.
Пример: {{"предмет договора": "поставка оборудования для нужд заказчика"}}
Никаких пояснений кроме JSON, никакого текста до или после JSON.
""")


def build_index(text: str) -> FAISS:
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)


def retrieve(index: FAISS, question: str, full_text: str, k: int = 5) -> list[str]:
    docs = index.similarity_search(question, k=k)
    chunks = [d.page_content for d in docs]

    header = full_text[:1500]
    if header not in chunks:
        chunks.insert(0, header)

    return chunks


def generate_answer(question: str, docs: list[str]) -> dict:
    context = "\n\n".join(docs)
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return _parse_json(response.content)


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"ответ": text}
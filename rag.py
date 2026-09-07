"""
RAG 핵심 로직 — 검색 + 답변 생성

ask.py(CLI)와 app.py(웹UI)가 공통으로 사용한다.
"""

import logging
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# SDK가 답변 스트림 중간에 출력하는 안내 메시지를 억제한다 (시연 화면 정리)
logging.getLogger("google_genai").setLevel(logging.ERROR)
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

BASE = Path(__file__).parent
DB_DIR = BASE / "vector_db"

EMBED_MODEL = "BAAI/bge-m3"
COLLECTION = "docs__bge-m3"
LLM_MODEL = "gemini-3.5-flash-lite"
TOP_K = 5

PROMPT_TEMPLATE = """아래 문서 내용만 참고해서 답변하세요.
문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

[문서]
{context}

[질문]
{question}"""


def load_embedder():
    """임베딩 모델 로딩 (로컬 실행, 외부 전송 없음)."""
    return SentenceTransformer(EMBED_MODEL)


def load_collection():
    """ChromaDB 컬렉션 열기."""
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False),  # 외부 통계 전송 차단
    )
    return client.get_collection(name=COLLECTION)


def load_llm():
    """Gemini 클라이언트."""
    load_dotenv()
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY 가 없습니다. setup_key.py 를 실행하세요.")
    return genai.Client(api_key=key)


def search(embedder, collection, question: str, k: int = TOP_K) -> list[dict]:
    """질문과 가장 유사한 청크 k개를 찾는다. (전 과정 로컬 실행)"""
    qvec = embedder.encode([question], normalize_embeddings=True)[0]
    res = collection.query(query_embeddings=[qvec.tolist()], n_results=k)

    return [
        {"text": doc, "source": meta["source"], "page": meta.get("page", 0)}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[문서: {c['source']}]\n{c['text']}" for c in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer_stream(client, prompt: str):
    """Gemini 답변을 스트리밍으로 받는다. (여기서만 외부 전송 발생)"""
    stream = client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW")
        ),
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text

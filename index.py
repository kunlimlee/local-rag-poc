"""
2단계 — 인덱싱: 문서 → 청크 → 임베딩 → ChromaDB 저장

이 단계가 RAG의 '준비' 과정이다. 한 번 실행해두면 이후 질문은 빠르게 처리된다.

중요 — 여기서 외부로 나가는 데이터는 없다:
  임베딩 모델(bge-m3)이 이 PC에서 직접 실행되므로 문서가 외부로 전송되지 않는다.
  ChromaDB의 사용 통계 전송(기본 켜짐)도 아래에서 끈다.

실행: .venv\\Scripts\\python.exe index.py
"""

import shutil
import sys
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from loader import load_documents

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
DB_DIR = BASE / "vector_db"

# docs/ 에 문서가 있으면 그것을, 없으면 저장소에 포함된 샘플을 쓴다.
# docs/ 는 .gitignore 에 등록되어 있다 — 고객사 실제 문서가 실수로 커밋되는 것을 막기 위함.
DOCS_DIR = BASE / "docs"
if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*")):
    DOCS_DIR = BASE / "sample_docs"

EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 컬렉션 이름에 임베딩 모델명을 넣는다.
# 임베딩 모델을 바꾸면 기존 벡터와 호환되지 않는데, 이름이 같으면 조용히 섞여서
# 검색 품질만 이상해진다. 이름을 분리해 두면 구조적으로 막힌다.
COLLECTION = "docs__bge-m3"


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*")):
        print(f"[실패] docs 폴더가 비어 있습니다: {DOCS_DIR}")
        sys.exit(1)

    total_start = time.time()

    # --- 1. 문서 읽기 ------------------------------------------------
    print("=" * 66)
    print("2단계 — 인덱싱")
    print("=" * 66)
    print("\n[1/4] 문서 읽는 중...")

    pieces, failures = load_documents(DOCS_DIR)
    if not pieces:
        print("[실패] 읽어들인 텍스트가 없습니다.")
        sys.exit(1)

    sources = {p["source"] for p in pieces}
    print(f"      파일 {len(sources)}개 / 조각 {len(pieces)}개")
    for name, reason in failures:
        print(f"      [건너뜀] {name}: {reason}")

    # --- 2. 청킹 ------------------------------------------------------
    print(f"\n[2/4] 청크로 분할 중... ({CHUNK_SIZE}자 단위, {CHUNK_OVERLAP}자 겹침)")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 앞쪽 구분자부터 시도한다. 문단 → 줄 → 문장 → 단어 순으로 끊어
        # 의미가 중간에 잘리는 것을 최대한 피한다.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for piece in pieces:
        for i, text in enumerate(splitter.split_text(piece["text"])):
            chunks.append(
                {
                    "text": text,
                    "source": piece["source"],
                    # ChromaDB 메타데이터는 None을 허용하지 않아 0으로 대체
                    "page": piece["page"] or 0,
                    "index": i,
                }
            )

    print(f"      청크 {len(chunks)}개 생성")
    avg = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"      평균 길이 {avg:.0f}자")

    # --- 3. 임베딩 (로컬 실행) ----------------------------------------
    print(f"\n[3/4] 임베딩 모델 로딩... ({EMBED_MODEL})")
    print("      ※ 최초 1회는 모델 약 2GB를 내려받으므로 시간이 걸립니다.")

    load_start = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    print(f"      모델 로딩 완료 ({time.time() - load_start:.1f}초)")

    print(f"\n      청크 {len(chunks)}개 벡터 변환 중... (이 PC에서 실행, 외부 전송 없음)")
    embed_start = time.time()
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=8,
        normalize_embeddings=True,  # 코사인 유사도 검색을 위해 정규화
        show_progress_bar=True,
    )
    embed_time = time.time() - embed_start
    print(f"      변환 완료 ({embed_time:.1f}초, 청크당 {embed_time / len(chunks):.2f}초)")
    print(f"      벡터 차원: {vectors.shape[1]}")

    # --- 4. ChromaDB 저장 ---------------------------------------------
    print("\n[4/4] 벡터 DB 저장 중...")

    # 재실행 시 중복을 막기 위해 기존 DB를 지우고 새로 만든다.
    # (PoC 규모에서는 전체 재생성이 가장 단순하다. 증분 인덱싱은 계획서 부록 참조)
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        # 기본값이 True 라서 사용 통계가 외부(PostHog)로 전송된다.
        # "데이터가 나가지 않는다"는 주장과 충돌하므로 반드시 끈다.
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(name=COLLECTION)

    collection.add(
        ids=[f"{c['source']}::{c['page']}::{c['index']}" for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"], "index": c["index"]}
            for c in chunks
        ],
    )

    print(f"      저장 완료: {DB_DIR}")

    # --- 결과 요약 ------------------------------------------------------
    total = time.time() - total_start
    print("\n" + "=" * 66)
    print("인덱싱 완료")
    print("=" * 66)
    print(f"  파일        : {len(sources)}개")
    print(f"  청크        : {len(chunks)}개")
    print(f"  임베딩 시간 : {embed_time:.1f}초")
    print(f"  전체 시간   : {total:.1f}초")
    print(f"  DB 위치     : {DB_DIR}")
    print("\n→ 계획서 '실측 기록표'의 '문서 N개 인덱싱 시간'에 기록하세요.")
    print("→ 다음: 3단계 검색 테스트 (step3_search_test.py)")


if __name__ == "__main__":
    main()

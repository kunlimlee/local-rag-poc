"""
2단계 보조 — 임베딩 모델 비교 (속도 + 검색 정확도)

배경: bge-m3 는 청크당 2.42초(i5-8265U, CPU)로 느리다.
      1,000청크면 40분, 질문마다 2.4초가 추가로 붙는다.
      더 가벼운 모델로 바꿔도 되는지, 정확도 손해는 없는지 확인한다.

속도만 보면 안 된다. 빨라도 엉뚱한 문서를 찾으면 RAG는 실패한다.
그래서 '정답 문서를 찾아내는가'를 함께 측정한다.

실행: .venv\\Scripts\\python.exe step2_embed_test.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from loader import load_documents

sys.stdout.reconfigure(encoding="utf-8")

DOCS_DIR = Path(__file__).parent / "docs"

MODELS = [
    "BAAI/bge-m3",                    # 현재 사용 중 (1024차원, 대형)
    "jhgan/ko-sroberta-multitask",    # 한국어 특화 소형 (768차원)
]

# 평가셋: 질문 → 답이 있어야 할 문서
# 실제 문서를 보고 사람이 직접 만든 정답이다.
EVAL = [
    ("연차휴가는 며칠 받나요?", "인사규정.txt"),
    ("USB 메모리를 사용해도 되나요?", "정보보안지침.md"),
    ("300만원짜리 장비를 사려면 누구 결재를 받아야 하나요?", "구매및지출규정.txt"),
    ("재택근무하면 통신비를 지원받을 수 있나요?", "재택근무지침.md"),
    ("결혼하면 축의금과 휴가를 얼마나 받나요?", "복리후생안내.md"),
    ("법정의무교육은 몇 시간을 들어야 하나요?", "교육훈련규정.txt"),
    ("비밀번호는 몇 자리로 만들어야 하나요?", "정보보안지침.md"),
    ("해외출장 숙박비 한도가 얼마인가요?", "인사규정.txt"),
]


def build_chunks():
    """문서를 읽어 청크로 나눈다 (index.py 와 동일한 설정)."""
    pieces, _ = load_documents(DOCS_DIR)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for p in pieces:
        for text in splitter.split_text(p["text"]):
            chunks.append({"text": text, "source": p["source"]})
    return chunks


def evaluate(model_name, chunks):
    print(f"\n{'─' * 66}")
    print(f"■ {model_name}")

    t0 = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - t0
    print(f"  모델 로딩     : {load_time:.1f}초")

    # --- 인덱싱 속도 ---
    t0 = time.time()
    vecs = model.encode(
        [c["text"] for c in chunks],
        batch_size=8, normalize_embeddings=True, show_progress_bar=False,
    )
    index_time = time.time() - t0
    per_chunk = index_time / len(chunks)
    print(f"  인덱싱        : {index_time:.1f}초 ({len(chunks)}청크, 청크당 {per_chunk:.2f}초)")
    print(f"  벡터 차원     : {vecs.shape[1]}")

    # --- 질문 임베딩 속도 (실사용 시 매 질문마다 발생) ---
    t0 = time.time()
    qvecs = model.encode(
        [q for q, _ in EVAL],
        batch_size=8, normalize_embeddings=True, show_progress_bar=False,
    )
    query_time = (time.time() - t0) / len(EVAL)
    print(f"  질문 1건 임베딩: {query_time:.2f}초  ← 답변 지연에 직접 더해짐")

    # --- 검색 정확도 ---
    # 정규화된 벡터이므로 내적 = 코사인 유사도
    sims = qvecs @ vecs.T
    top1 = top3 = 0
    misses = []
    for i, (question, expected) in enumerate(EVAL):
        ranked = np.argsort(-sims[i])
        sources = [chunks[j]["source"] for j in ranked[:3]]
        if sources[0] == expected:
            top1 += 1
        if expected in sources:
            top3 += 1
        else:
            misses.append((question, expected, sources[0]))

    n = len(EVAL)
    print(f"  Top-1 정확도  : {top1}/{n}")
    print(f"  Top-3 정확도  : {top3}/{n}  ← RAG는 보통 여러 개를 넘기므로 이쪽이 중요")

    for q, exp, got in misses:
        print(f"    [실패] {q}")
        print(f"           기대 {exp} / 실제 {got}")

    return {
        "model": model_name, "per_chunk": per_chunk, "query": query_time,
        "top1": top1, "top3": top3, "dim": vecs.shape[1],
    }


def main():
    chunks = build_chunks()
    print("=" * 66)
    print(f"임베딩 모델 비교 — 청크 {len(chunks)}개 / 평가 질문 {len(EVAL)}개")
    print("=" * 66)

    results = []
    for name in MODELS:
        try:
            results.append(evaluate(name, chunks))
        except Exception as e:
            print(f"\n■ {name}\n  [실패] {type(e).__name__}: {str(e)[:150]}")

    print("\n" + "=" * 66)
    print(f"{'모델':<32}{'청크당':>8}{'질문':>8}{'Top-1':>7}{'Top-3':>7}")
    print("-" * 66)
    for r in results:
        print(f"{r['model']:<32}{r['per_chunk']:>7.2f}s{r['query']:>7.2f}s"
              f"{r['top1']:>5}/{len(EVAL)}{r['top3']:>5}/{len(EVAL)}")

    if len(results) == 2:
        a, b = results
        speedup = a["per_chunk"] / b["per_chunk"]
        print(f"\n  → {b['model']} 가 {speedup:.1f}배 빠름")
        if b["top3"] >= a["top3"]:
            print("  → 검색 정확도 손해 없음. 교체 검토 가능")
        else:
            print(f"  → 다만 Top-3 정확도가 {a['top3']}→{b['top3']} 로 하락. 트레이드오프 판단 필요")


if __name__ == "__main__":
    main()

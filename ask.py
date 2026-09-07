"""
3~4단계 — 질문 → 검색 → 답변 (CLI)

실행:
    .venv\\Scripts\\python.exe ask.py "연차는 며칠인가요?"
    .venv\\Scripts\\python.exe ask.py --search "연차"     # 검색 결과만 확인
"""

import sys
import time

import rag

sys.stdout.reconfigure(encoding="utf-8")


def main():
    args = sys.argv[1:]
    search_only = "--search" in args
    args = [a for a in args if a != "--search"]

    if not args:
        print('사용법: ask.py "질문 내용"')
        sys.exit(1)

    question = " ".join(args)

    print("모델 로딩 중...")
    embedder = rag.load_embedder()
    collection = rag.load_collection()

    # --- 검색 (로컬) ---
    t0 = time.time()
    chunks = rag.search(embedder, collection, question)
    search_time = time.time() - t0

    print(f"\n[검색] {len(chunks)}개 청크 ({search_time:.2f}초)")
    for i, c in enumerate(chunks, 1):
        preview = c["text"][:70].replace("\n", " ")
        print(f"  {i}. {c['source']} — {preview}...")

    if search_only:
        return

    # --- 답변 생성 (외부 전송 발생) ---
    print(f"\n[질문] {question}")
    print("[답변] ", end="", flush=True)

    client = rag.load_llm()
    prompt = rag.build_prompt(question, chunks)

    t0 = time.time()
    first = None
    for piece in rag.answer_stream(client, prompt):
        if first is None:
            first = time.time() - t0
        print(piece, end="", flush=True)
    total = time.time() - t0

    # --- 출처 ---
    # 검색 결과는 유사도 순이므로 상위 3개에서만 뽑는다.
    # 5개 전부 표시하면 관련 없는 문서까지 출처로 보여 신뢰를 떨어뜨린다.
    sources = list(dict.fromkeys(c["source"] for c in chunks[:3]))
    print(f"\n\n[출처] {', '.join(sources)}")
    print(f"[시간] 첫 글자 {first:.2f}초 / 완료 {total:.2f}초")


if __name__ == "__main__":
    main()

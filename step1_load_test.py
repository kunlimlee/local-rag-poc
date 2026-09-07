"""
1단계 — 문서 텍스트 추출 확인

목적: 고객사 실제 문서 형식에서 텍스트가 제대로 뽑히는지 눈으로 확인한다.
      여기서 텍스트가 안 나오면 뒤 단계는 전부 무의미하다.

확인할 것:
  - 한글이 깨지지 않는가
  - 표·머리글이 어떻게 섞여 나오는가
  - 스캔 PDF(이미지)라서 글자가 아예 안 나오는 파일이 있는가  ← OCR 필요 신호

실행: .venv\\Scripts\\python.exe step1_load_test.py
"""

import sys
from pathlib import Path

from loader import load_documents

sys.stdout.reconfigure(encoding="utf-8")

DOCS_DIR = Path(__file__).parent / "docs"


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*")):
        print(f"[실패] docs 폴더가 비어 있습니다: {DOCS_DIR}")
        print("       읽어볼 문서를 넣고 다시 실행하세요.")
        sys.exit(1)

    pieces, failures = load_documents(DOCS_DIR)

    print("=" * 66)
    print("문서 텍스트 추출 확인")
    print("=" * 66)

    # 파일별로 묶어서 보여준다
    by_source: dict[str, list[dict]] = {}
    for p in pieces:
        by_source.setdefault(p["source"], []).append(p)

    for source, items in by_source.items():
        text = "\n".join(i["text"] for i in items)
        pages = [i["page"] for i in items if i["page"]]

        print(f"\n{'─' * 66}")
        print(f"■ {source}")
        print(f"  글자 수 : {len(text.strip()):,}자")
        if pages:
            print(f"  페이지  : {len(pages)}쪽")
        preview = text.strip()[:200].replace("\n", " ")
        print(f"  미리보기: {preview}...")

    print("\n" + "=" * 66)
    print(f"결과: 정상 {len(by_source)}개 파일 / 조각 {len(pieces)}개")
    print("=" * 66)

    if failures:
        print("\n확인이 필요한 파일:")
        for name, reason in failures:
            print(f"  - {name}: {reason}")
        print("\n→ 스캔 PDF가 많다면 OCR 도입을 검토해야 한다 (계획서 1단계 항목)")
    else:
        print("\n모든 파일에서 텍스트가 정상 추출되었습니다. 2단계로 진행 가능합니다.")


if __name__ == "__main__":
    main()

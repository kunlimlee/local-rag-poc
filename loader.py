"""
문서 로딩 — 파일에서 텍스트를 뽑아낸다.

LangChain 로더를 쓰지 않는 이유:
  langchain-community가 지원 종료(sunset) 예정이고, 그 로더들은 결국
  pypdf / docx2txt 를 감싼 얇은 껍데기일 뿐이다. 직접 쓰는 편이 의존성도 적고
  무슨 일이 일어나는지 눈에 보인다.

다른 단계에서 import 해서 쓴다:
    from loader import load_documents
"""

from pathlib import Path

import docx2txt
from pypdf import PdfReader

SUPPORTED = {".txt", ".md", ".pdf", ".docx"}


def _read_text(path: Path) -> list[tuple[str, int | None]]:
    """텍스트 파일을 읽는다. (내용, 페이지번호) 목록을 반환."""
    # Windows 기본 인코딩(cp949)으로 저장된 파일도 있으므로 순서대로 시도한다
    for encoding in ("utf-8", "cp949"):
        try:
            return [(path.read_text(encoding=encoding), None)]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8/cp949 모두 실패", b"", 0, 1, "인코딩을 알 수 없음")


def _read_pdf(path: Path) -> list[tuple[str, int | None]]:
    """PDF를 페이지 단위로 읽는다. 출처에 페이지 번호를 남기기 위함."""
    reader = PdfReader(str(path))
    out = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            out.append((text, i))
    return out


def _read_docx(path: Path) -> list[tuple[str, int | None]]:
    return [(docx2txt.process(str(path)) or "", None)]


READERS = {
    ".txt": _read_text,
    ".md": _read_text,
    ".pdf": _read_pdf,
    ".docx": _read_docx,
}


def load_file(path: Path) -> list[dict]:
    """
    파일 하나를 읽어 문서 조각 목록을 반환한다.

    각 조각: {"text": 본문, "source": 파일명, "page": 페이지번호 또는 None}
    """
    reader = READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"지원하지 않는 형식: {path.suffix}")

    return [
        {"text": text, "source": path.name, "page": page}
        for text, page in reader(path)
        if text.strip()
    ]


def load_documents(docs_dir: Path) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    폴더 전체를 읽는다.

    반환: (문서 조각 목록, 실패 목록[(파일명, 사유)])
    """
    chunks: list[dict] = []
    failures: list[tuple[str, str]] = []

    for path in sorted(p for p in docs_dir.rglob("*") if p.is_file()):
        rel = str(path.relative_to(docs_dir))

        if path.suffix.lower() not in SUPPORTED:
            failures.append((rel, f"지원하지 않는 형식 ({path.suffix})"))
            continue

        try:
            loaded = load_file(path)
        except Exception as e:
            failures.append((rel, f"{type(e).__name__}: {e}"))
            continue

        if not loaded:
            # 페이지는 있는데 글자가 없다 = 스캔본(이미지)일 가능성
            failures.append((rel, "텍스트 없음 (스캔 문서 의심 — OCR 필요)"))
            continue

        chunks.extend(loaded)

    return chunks, failures

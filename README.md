# 사내 문서 RAG 챗봇 (PoC)

기업 문서를 **사내 PC에 보관한 채** 질문에 답하는 RAG 챗봇. 일반 사무용 PC에서 동작한다.

- 문서 전체는 PC 밖으로 나가지 않는다 — 임베딩과 벡터 검색이 로컬에서 실행된다
- 질문 시 관련 발췌(상위 5개 청크)만 LLM API로 전송된다
- 답변 속도: **첫 글자 1.2초 / 완료 1.5초** (i5-8265U, GPU 없음)

## 데이터 경계

| 구성요소 | 실행 위치 | 외부 전송 |
|---|---|---|
| 원본 문서 (전체) | 로컬 디스크 | 없음 |
| 임베딩 (bge-m3) | 로컬 CPU | 없음 |
| 벡터 DB (ChromaDB) | 로컬 폴더 | 없음 |
| 답변 생성 (Gemini) | 클라우드 | **질문 + 청크 5개** |

> "데이터가 전혀 나가지 않는다"고 말할 수 없다. 정확히는 **"전체 문서는 사내에 남고,
> 질문 시 관련 발췌만 전송된다"**. 완전 오프라인이 필요하면 LLM을 Ollama로 교체한다
> (`rag.py`의 호출부만 변경).

## 설치

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Windows 사전 요건** — [VC++ 2015-2022 재배포 패키지 (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe)
`C:\Windows\System32\vcruntime140_1.dll` 이 없으면 설치해야 한다. 없으면 torch가
`OSError: [WinError 1114]` 로 실패한다. **설치에 관리자 권한이 필요하다.**

## 실행

```bash
# 1. API 키 설정 (https://aistudio.google.com/apikey)
.venv\Scripts\python.exe setup_key.py

# 2. 인덱싱 (문서 추가/변경 시 재실행)
.venv\Scripts\python.exe index.py

# 3-a. CLI 로 질문
.venv\Scripts\python.exe ask.py "연차는 며칠인가요?"

# 3-b. 웹 UI
.venv\Scripts\streamlit.exe run app.py
```

문서는 `docs/` 폴더에 넣는다. 비어 있으면 `sample_docs/`의 예시 문서를 사용한다.
(`docs/`는 `.gitignore`에 등록되어 있어 실제 문서가 커밋되지 않는다.)

## 구성

```
rag.py         검색 + 답변 생성 (핵심 로직)
index.py       문서 → 청크 → 임베딩 → ChromaDB
loader.py      PDF/DOCX/TXT/MD 텍스트 추출
ask.py         CLI
app.py         Streamlit 웹 UI
setup_key.py   API 키를 .env 에 저장

step0_api_test.py     API 연결 확인
step0_speed_test.py   LLM 모델 속도·정확도 비교
step1_load_test.py    문서 추출 확인
step2_embed_test.py   임베딩 모델 비교
```

`step*` 스크립트는 단계별 검증용이다. 고객사 PC에서 문제가 생겼을 때 어느 단계가
깨졌는지 하나씩 짚어보는 용도로 남겨두었다.

## 기술 선택 근거 (실측 기반)

**LLM: `gemini-3.5-flash-lite`** — 최신 모델(`3.7`/`3.8-flash`)은 호출 시 **503 과부하**가
발생해 시연에 부적합했다. `3.8-flash`는 thinking 수준 조절도 거부(400)되어 지연을
낮출 방법이 없다.

| 모델 | thinking | 첫 글자 | 완료 |
|---|---|---|---|
| **gemini-3.5-flash-lite** | LOW | **0.86초** | **3.22초** |
| gemini-3.5-flash | MINIMAL | 7.97초 | 9.00초 |
| gemini-3.6-flash | LOW | 25.70초 | 26.83초 |
| gemini-3.7 / 3.8-flash | — | 503 과부하 | — |

**임베딩: `BAAI/bge-m3`** — 문서량이 많으면 `jhgan/ko-sroberta-multitask`가
인덱싱 13.7배 빠르고(청크당 2.26초 → 0.17초) 모델 크기도 4.3GB → 423MB로 작다.
평가 질문 8개 기준 Top-3 정확도는 동일했다.

**LangChain 미사용** (`langchain-text-splitters` 제외) — `langchain-community`는 지원
종료 예정이고, 그 문서 로더는 `pypdf`/`docx2txt`를 감싼 껍데기다.

## ⚠️ 외부 전송 기본값 차단

주요 라이브러리는 **기본값이 외부 전송 켜짐**이다. 이 저장소에는 모두 차단되어 있다.

| 항목 | 기본값 | 조치 |
|---|---|---|
| ChromaDB 사용 통계 | 전송 켜짐 (PostHog) | `Settings(anonymized_telemetry=False)` |
| Streamlit 사용 통계 | 전송 켜짐 | `.streamlit/config.toml` |
| Streamlit 네트워크 바인딩 | **0.0.0.0 (전체 개방)** | `address = "localhost"` |

세 번째가 특히 위험하다. 기본 설정으로 띄우면 같은 망의 누구나 접속할 수 있다.
다른 기기에서 `http://<PC의 IP>:8501` 접속이 거부되는지 확인할 것.

## LLM 무료 티어 주의

Google 무료 티어는 제출한 내용을 **모델 개선에 사용하며 사람이 읽을 수 있다.**
약관에 "기밀정보를 무료 티어에 제출하지 말 것"이 명시되어 있다.

- 샘플/더미 문서 테스트 → 무료 티어 무방
- **실제 기밀문서 → 유료 티어 필수** (결제 등록 시 학습에 사용되지 않음)

출처: [Gemini API Terms](https://ai.google.dev/gemini-api/terms)

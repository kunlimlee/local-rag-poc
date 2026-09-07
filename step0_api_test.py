"""
0단계 — Gemini API 연결 확인

목적: 이 PC(그리고 고객사 PC)의 네트워크에서 Gemini API가 호출되는지 확인한다.
      기업 PC는 프록시/방화벽으로 외부 API를 막는 경우가 많아, 이것이 최대 리스크다.

실행: .venv\\Scripts\\python.exe step0_api_test.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai

# Windows 터미널에서 한글이 깨지지 않도록
sys.stdout.reconfigure(encoding="utf-8")

# 실측으로 확정한 모델 (step0_speed_test.py 결과 참조)
# 최신 모델(3.7/3.8-flash)은 503 과부하가 발생해 시연에 부적합
MODEL = "gemini-3.5-flash-lite"

# --- 1. API 키 읽기 -------------------------------------------------
load_dotenv()  # .env 파일을 읽어 환경변수로 올린다
api_key = os.getenv("GEMINI_API_KEY")

if not api_key or api_key.startswith("여기에"):
    print("[실패] API 키가 없습니다.")
    print()
    print("해결 방법:")
    print("  1. https://aistudio.google.com/apikey 에서 키를 발급받으세요")
    print("  2. 이 폴더에 '.env' 파일을 만드세요")
    print("  3. 파일 안에 다음 한 줄을 넣으세요:")
    print()
    print("     GEMINI_API_KEY=발급받은키")
    print()
    sys.exit(1)

print(f"[확인] API 키 로드됨 (앞 4자리: {api_key[:4]}...)")

# --- 2. API 호출 ----------------------------------------------------
print(f"[진행] {MODEL} 호출 중...")

client = genai.Client(api_key=api_key)

start = time.time()
try:
    result = client.interactions.create(
        model=MODEL,
        input="한국어로 '연결 성공'이라고만 답하세요.",
    )
except Exception as e:
    print()
    print(f"[실패] API 호출 실패: {type(e).__name__}")
    print(f"       {e}")
    print()
    print("확인할 것:")
    print("  - 인터넷 연결이 되어 있는가")
    print("  - 회사 프록시/방화벽이 googleapis.com 을 막고 있는가")
    print("    → 막혀 있다면 방화벽 허용 신청 필요 (승인에 수일 소요 가능)")
    print("  - API 키가 올바른가")
    sys.exit(1)

elapsed = time.time() - start

# --- 3. 결과 출력 ---------------------------------------------------
print()
print("=" * 50)
print("[성공] API 연결 확인됨")
print("=" * 50)
print(f"응답 내용 : {result.output_text.strip()}")
print(f"응답 시간 : {elapsed:.2f}초")
print()
print("→ 계획서 '실측 기록표'의 '사내망 API 연결' 칸에 '성공'으로 기록하세요.")

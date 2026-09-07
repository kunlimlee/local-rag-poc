"""
API 키를 .env 파일에 저장한다.

실행: .venv\\Scripts\\python.exe setup_key.py
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ENV_PATH = Path(__file__).parent / ".env"

print("=" * 55)
print("Gemini API 키 설정")
print("=" * 55)
print(f"저장 위치: {ENV_PATH}")
print()
print("키 발급: https://aistudio.google.com/apikey")
print()

key = input("API 키를 붙여넣고 Enter: ").strip()

if not key:
    print("\n[중단] 입력된 키가 없습니다.")
    sys.exit(1)

# 흔한 실수 정리: 따옴표나 'GEMINI_API_KEY=' 를 같이 붙여넣은 경우
key = key.strip("'\"")
if "=" in key:
    key = key.split("=", 1)[1].strip().strip("'\"")

# BOM 없이 저장 (BOM이 붙으면 첫 줄 키 이름을 못 읽는다)
ENV_PATH.write_text(f"GEMINI_API_KEY={key}\n", encoding="utf-8")

print()
print(f"[완료] .env 저장됨 (키 길이 {len(key)}자, 앞 4자리 {key[:4]}...)")
print()
print("다음 실행:")
print("  .venv\\Scripts\\python.exe step0_api_test.py")

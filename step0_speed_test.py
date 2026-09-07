"""
0단계 보조 — 모델/설정별 속도·정확도 실측 (2차)

1차 측정에서 밝혀진 것:
  - gemini-2.5-flash : 신규 사용자에게 제공 종료 (404)
  - gemini-3.8-flash : thinking 조절 불가(400), 고부하 시 503 발생
  - gemini-3.5-flash : thinking=기본 11.1초/정답, MINIMAL 8.4초/오답

이번 목표: 10초 이내 + 정답을 동시에 만족하는 조합 찾기.
          체감 속도를 좌우하는 '첫 글자까지 시간'도 함께 측정한다.

실행: .venv\\Scripts\\python.exe step0_speed_test.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

CONTEXT = """
[문서 1] 연차휴가 규정
직원은 입사 1년 경과 시 15일의 연차휴가를 부여받는다. 3년 이상 근속한 직원은
2년마다 1일씩 가산되며, 총 25일을 초과할 수 없다. 연차는 회계연도 기준으로
산정하며, 미사용 연차는 다음 해로 이월되지 않는다.

[문서 2] 휴가 신청 절차
연차 사용은 최소 3일 전까지 그룹웨어를 통해 신청해야 한다. 5일 이상 연속
사용 시에는 팀장 승인 외에 본부장 승인이 추가로 필요하다.
"""
QUESTION = "5년 근속한 직원의 연차는 며칠이고, 7일 연속으로 쓰려면 누구 승인이 필요한가요?"
PROMPT = f"아래 문서 내용만 참고해서 답변하세요.\n\n{CONTEXT}\n\n[질문]\n{QUESTION}"

# 정답: 연차 17일 / 본부장 승인
CASES = [
    ("gemini-3.5-flash", "MINIMAL"),
    ("gemini-3.5-flash", "LOW"),
    ("gemini-3.5-flash", "MEDIUM"),
    ("gemini-3.5-flash-lite", "LOW"),
    ("gemini-3.6-flash", "LOW"),
    ("gemini-3.7-flash", "LOW"),
]


def make_config(thinking):
    if not thinking:
        return None
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking)
    )


def measure_stream(model, thinking):
    """스트리밍으로 호출해 (첫글자까지, 전체완료, 답변) 측정."""
    start = time.time()
    first = None
    chunks = []
    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=PROMPT, config=make_config(thinking)
        ):
            if chunk.text:
                if first is None:
                    first = time.time() - start
                chunks.append(chunk.text)
        return first, time.time() - start, "".join(chunks).strip()
    except Exception as e:
        print(f"  [실패] {type(e).__name__}: {str(e)[:120]}")
        return None


print("=" * 66)
print("모델/설정별 속도·정확도 실측  (정답: 연차 17일, 본부장 승인)")
print("=" * 66)

print("\n[준비] 연결 예열 중...")
measure_stream("gemini-3.5-flash", "LOW")

results = []
for model, thinking in CASES:
    print(f"\n[측정] {model} / thinking={thinking}")
    out = measure_stream(model, thinking)
    if not out:
        continue
    first, total, answer = out
    correct = "17" in answer and "본부장" in answer
    results.append((model, thinking, first, total, correct, answer))
    print(f"  첫글자 {first:.2f}초 / 완료 {total:.2f}초 / 정답 {'O' if correct else 'X'}")

print("\n" + "=" * 66)
print(f"{'모델':<24} {'thinking':<9} {'첫글자':>7} {'완료':>7} {'정답':>5}")
print("-" * 66)
for model, thinking, first, total, correct, _ in sorted(results, key=lambda x: x[3]):
    print(f"{model:<24} {thinking:<9} {first:>6.2f}s {total:>6.2f}s {'O' if correct else 'X':>5}")

print("\n" + "=" * 66)
print("10초 이내 + 정답 동시 만족")
print("=" * 66)
ok = [r for r in results if r[3] <= 10 and r[4]]
if ok:
    for model, thinking, first, total, _, _ in sorted(ok, key=lambda x: x[3]):
        print(f"  ✔ {model} / thinking={thinking}  (첫글자 {first:.2f}초, 완료 {total:.2f}초)")
else:
    print("  없음 — 스트리밍 '첫글자' 기준으로 체감 속도를 확보해야 함")

print("\n" + "=" * 66)
print("오답 사례 확인")
print("=" * 66)
for model, thinking, _, _, correct, answer in results:
    if not correct:
        print(f"\n■ {model} / {thinking}")
        print(f"  {answer[:180]}")

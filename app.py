"""
5단계 — Streamlit 웹 UI (시연용)

실행:
    .venv\\Scripts\\streamlit.exe run app.py

브라우저가 자동으로 열린다. 종료는 터미널에서 Ctrl+C.
"""

import time

import streamlit as st

import rag

st.set_page_config(page_title="사내 문서 챗봇", page_icon="📄")


# 모델과 DB는 한 번만 로딩한다 (질문할 때마다 다시 읽으면 매번 10초씩 걸린다)
@st.cache_resource
def get_resources():
    return rag.load_embedder(), rag.load_collection(), rag.load_llm()


st.title("📄 사내 문서 챗봇")
st.caption(
    "문서는 이 PC에만 저장됩니다. 질문 시 관련 발췌만 외부 LLM으로 전송됩니다."
)

with st.spinner("모델 로딩 중... (최초 1회)"):
    embedder, collection, client = get_resources()

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"출처: {', '.join(msg['sources'])}")

# 질문 입력
if question := st.chat_input("문서에 대해 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1) 검색 (로컬 실행, 외부 전송 없음)
        chunks = rag.search(embedder, collection, question)

        # 2) 답변 생성 (여기서만 외부 전송)
        t0 = time.time()
        prompt = rag.build_prompt(question, chunks)
        answer = st.write_stream(rag.answer_stream(client, prompt))
        elapsed = time.time() - t0

        sources = list(dict.fromkeys(c["source"] for c in chunks[:3]))
        st.caption(f"출처: {', '.join(sources)}  ·  {elapsed:.1f}초")

        with st.expander("검색된 문서 발췌 보기"):
            for i, c in enumerate(chunks, 1):
                st.markdown(f"**{i}. {c['source']}**")
                st.text(c["text"][:300] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

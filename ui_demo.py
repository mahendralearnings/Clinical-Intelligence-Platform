"""
A simple visual demo UI - calls your existing FastAPI backend, shows
RAG and Agent results side by side, with the agent's step-by-step
reasoning displayed visually. This is a demo tool, not part of the
clean-architecture app itself.
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Clinical Intelligence Platform - Demo", layout="wide")
st.title("Clinical Intelligence Platform — Live Demo")

# --- Login (sidebar) ---
st.sidebar.header("Login")
role_credentials = {
    "Doctor": ("bob@clinic.dev", "doctor_pass_2!"),
}
role = st.sidebar.selectbox("Login as", list(role_credentials.keys()))

if "token" not in st.session_state:
    st.session_state.token = None

if st.sidebar.button("Log in"):
    email, password = role_credentials[role]
    response = requests.post(f"{API_BASE}/auth/login", json={"email": email, "password": password})
    if response.status_code == 200:
        st.session_state.token = response.json()["access_token"]
        st.sidebar.success("Logged in!")
    else:
        st.sidebar.error(f"Login failed: {response.text}")

if not st.session_state.token:
    st.warning("Please log in using the sidebar first.")
    st.stop()

headers = {"Authorization": f"Bearer {st.session_state.token}"}

# --- Mode selector ---
mode = st.radio("Choose mode", ["Plain RAG (single search + answer)", "Agent (multi-step reasoning)"])

question = st.text_input(
    "Ask a question",
    value="What is the maximum daily metformin dose, divided evenly across 3 doses?"
    if mode.startswith("Agent")
    else "What is the lactic acidosis risk with metformin?",
)

if st.button("Ask", type="primary"):
    with st.spinner("Thinking..."):
        if mode.startswith("Plain RAG"):
            response = requests.post(
                f"{API_BASE}/rag/query", json={"question": question, "top_k": 3}, headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer")
                st.write(data["answer"])
                st.subheader("Sources")
                for src in data["sources"]:
                    st.markdown(f"- **{src['source']}** — {src['section_title']} (score: {src['score']:.2f})")
            else:
                st.error(response.text)

        else:
            response = requests.post(
                f"{API_BASE}/agent/query", json={"question": question}, headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Agent's reasoning steps")
                icons = {"tool_call": "🔧", "tool_result": "📄", "ai_message": "💭"}
                for step in data["steps"]:
                    icon = icons.get(step["step_type"], "•")
                    st.markdown(f"{icon} **{step['step_type']}**: {step['content']}")
                st.subheader("Final Answer")
                st.success(data["final_answer"])
            else:
                st.error(response.text)
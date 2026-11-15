"""Optional Streamlit UI for the document retrieval slice; no synthetic answers."""

import os
import httpx
import streamlit as st

st.set_page_config(page_title="SupportLens", page_icon="📚")
st.title("SupportLens")
st.caption("Day-one scaffold · Document evidence first")
mode = st.radio("Mode", ["docs_qa", "banking_triage"], horizontal=True)
question = st.text_area("Question")
if st.button("Submit", disabled=not question.strip()):
    try:
        response = httpx.post(os.environ.get("SUPPORTLENS_API", "http://127.0.0.1:8000") + "/v1/assist",
                              json={"mode": mode, "question": question}, timeout=30)
        if response.status_code != 200:
            st.warning(response.json().get("detail", response.text))
        else:
            value = response.json()
            st.write(value["decision"], "·", value["reason"])
            for item in value["evidence"]:
                with st.expander(item["title"]):
                    st.write(item["text"])
                    st.caption(f"{item['evidence_id']} · {item['unit']} · score {item['score']:.3f}")
            st.caption(f"Artifact: {value['artifact_version']} · {value['latency']['total_ms']:.1f} ms")
    except httpx.HTTPError as error:
        st.error(f"API unavailable: {error}")

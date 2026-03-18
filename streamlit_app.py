import streamlit as st

from rag import DEFAULT_MODEL, DEFAULT_TOP_K, build_index, load_index, generate_answer

st.set_page_config(page_title="RAG Dragon Ball", layout="wide")

st.title("RAG - Dragon Ball Saga Freezer")

st.markdown(
    """
    This app lets you query the Dragon Ball Saga Freezer corpus using a Retrieval-Augmented-Generation pipeline.

    1. Build (or reuse) a FAISS index from the `chunks/` files.
    2. Ask a question in English.
    3. Get an answer produced by the LLM using the top-k retrieved passages.
    """
)

# Sidebar settings
st.sidebar.header("Settings")
chunks_dir = st.sidebar.text_input("Chunks directory", value="chunks/saga_freezer")
index_dir = st.sidebar.text_input("Index directory", value="rag_index")

st.sidebar.markdown("---")
model = st.sidebar.text_input("OpenRouter model", value=DEFAULT_MODEL)
top_k = st.sidebar.number_input("Top-k passages", value=DEFAULT_TOP_K, min_value=1, max_value=20, step=1)

tab_build, tab_query = st.tabs(["Build index", "Ask a question"])

with tab_build:
    st.subheader("Build FAISS index")
    st.write("This reads the chunk text files and builds a local FAISS vector index.")

    if st.button("Build index"):
        with st.spinner("Building index (this may take a while)…"):
            build_index(chunks_dir=chunks_dir, index_dir=index_dir)
        st.success("Index built successfully.")

with tab_query:
    st.subheader("Query the index")
    question = st.text_input("Ask a question (in English for best results)", value="Who is Freezer?")

    if st.button("Run query"):
        if not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer…"):
                store = load_index(index_dir)
                docs = store.similarity_search(question, k=top_k)
                answer = generate_answer(question, docs, model=model)

            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### Retrieved context (top-k)")
            for i, d in enumerate(docs, start=1):
                st.markdown(f"**[{i}]** `{d.metadata.get('source')}`  ")
                st.write(d.page_content)

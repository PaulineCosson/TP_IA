import os
import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env")

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "microsoft/wizardlm-2-8x22b:free"
DEFAULT_TOP_K = 4


def load_texts(chunks_dir: str) -> list[Document]:
    p = Path(chunks_dir)
    if not p.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    docs = []
    for path in sorted(p.rglob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs


def build_index(chunks_dir: str, index_dir: str, model_name: str = "all-MiniLM-L6-v2"):
    docs = load_texts(chunks_dir)
    if not docs:
        raise SystemExit("No documents found in chunks directory.")

    emb = HuggingFaceEmbeddings(model_name=model_name)
    store = FAISS.from_documents(docs, emb)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    store.save_local(index_dir)
    print(f"Index saved to {index_dir} ({len(docs)} docs).")


def load_index(index_dir: str, model_name: str = "all-MiniLM-L6-v2") -> FAISS:
    emb = HuggingFaceEmbeddings(model_name=model_name)
    # FAISS uses pickle to store the index; allow_dangerous_deserialization must be set to True
    # when loading from a file you created locally.
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)


def build_prompt(question: str, contexts: list[Document]) -> str:
    ctx_text = "\n\n---\n\n".join(
        [f"Source: {d.metadata.get('source', '')}\n{d.page_content}" for d in contexts]
    )
    return (
        "Tu es un assistant expert qui répond de manière concise en utilisant uniquement les informations fournies.\n\n"
        "Utilise les passages suivants comme contexte (n'invente rien) :\n\n"
        f"{ctx_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Réponse :"
    )


def generate_answer(question: str, contexts: list[Document], model: str, temperature: float = 0.2) -> str:
    prompt = build_prompt(question, contexts)
    client = OpenAI(base_url=OPENROUTER_URL, api_key=API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )

    if resp.choices:
        return resp.choices[0].message.content.strip()
    return "No response generated."


def query_index(index_dir: str, question: str, top_k: int = DEFAULT_TOP_K, model: str = DEFAULT_MODEL, temperature: float = 0.2):
    store = load_index(index_dir)
    docs = store.similarity_search(question, k=top_k)
    answer = generate_answer(question, docs, model=model, temperature=temperature)

    print("\n--- CONTEXTE (top-k) ---")
    for i, d in enumerate(docs, start=1):
        print(f"[{i}] {d.metadata.get('source')} ({len(d.page_content)} chars)")

    print("\n--- RÉPONSE ---\n")
    print(answer)


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline (build index + query).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Build a FAISS index from chunk files.")
    p_build.add_argument("--chunks-dir", default="chunks", help="Directory containing chunk .txt files.")
    p_build.add_argument("--index-dir", default="rag_index", help="Directory where to store the FAISS index.")
    p_build.add_argument("--embed-model", default="all-MiniLM-L6-v2", help="HuggingFace embedding model name.")

    p_query = sub.add_parser("query", help="Query a built FAISS index.")
    p_query.add_argument("--index-dir", default="rag_index", help="Directory where FAISS index is stored.")
    p_query.add_argument("--question", required=True, help="The question to ask.")
    p_query.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    p_query.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model to use for generation.")
    p_query.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")

    args = parser.parse_args()

    if args.command == "build":
        build_index(args.chunks_dir, args.index_dir, model_name=args.embed_model)
    elif args.command == "query":
        query_index(args.index_dir, args.question, top_k=args.top_k, model=args.model, temperature=args.temperature)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

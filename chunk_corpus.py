"""Découper un corpus texte (wiki) en passages plus courts.

Ce script illustre l'étape 7 du TP : découper les pages en fragments prêts à être
vectorisés/embeddés.

Usage:
  python chunk_corpus.py corpus/saga_freezer --out chunks/saga_freezer --chunk-size 500 --overlap 50

Le script produit un fichier par page + un identifiant de fragment, par exemple :
  chunks/saga_freezer/Goku_0001.txt
  chunks/saga_freezer/Goku_0002.txt

Niveau RAG, ces fragments servent ensuite à calculer des embeddings et à construire
une base vectorielle (FAISS, Chroma, etc.).

"""

import argparse
import glob
import pathlib
import re


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Découpe un texte en chunks basés sur des mots.

    chunk_size et overlap sont exprimés en nombre de mots.
    """
    words = re.findall(r"\w+|[^	\n\r\f\v\w]+", text)
    # Si on veut rester sur une découpe propre, on peut travailler sur les mots uniquement.
    # Ici on découpe en tokens simples pour avoir un découpage rapide.
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words).strip())
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Découper un corpus de pages wiki en passages plus courts.")
    parser.add_argument("src", help="Répertoire source contenant les .txt du corpus")
    parser.add_argument("--out", default="chunks", help="Répertoire de sortie")
    parser.add_argument("--chunk-size", type=int, default=500, help="Nombre de mots par chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Nombre de mots en chevauchement")
    args = parser.parse_args()

    src_dir = pathlib.Path(args.src)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src_dir / "**" / "*.txt"), recursive=True))
    if not files:
        print(f"Aucun fichier trouvé dans {src_dir}")
        return

    for f in files:
        text = pathlib.Path(f).read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = pathlib.Path(f).stem
        chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
        for i, chunk in enumerate(chunks, start=1):
            out_file = out_dir / f"{title}_{i:04d}.txt"
            out_file.write_text(chunk, encoding="utf-8")

    print(f"Découpage terminé : {len(files)} pages -> {len(list(out_dir.glob('*.txt')))} chunks")


if __name__ == "__main__":
    main()

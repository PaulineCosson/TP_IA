"""Verify a downloaded wiki corpus.

This script implements the "Étape 5" du TP : vérifier que le corpus n'est pas vide
et qu'il contient un nombre de pages / mots cohérent.

Usage:
  python verify_corpus.py corpus/saga_freezer

It prints:
- nombre de fichiers
- fichiers vides
- taille (chars / mots) par fichier
- stats globales
"""

import argparse
import glob
import pathlib
import re


def analyze_dir(path: str):
    files = sorted(glob.glob(f"{path.rstrip('\\/')}/**/*.txt", recursive=True))
    if not files:
        print(f"Aucun fichier .txt trouvé dans {path}")
        return

    total_words = 0
    total_chars = 0
    empty_files = []

    for f in files:
        text = pathlib.Path(f).read_text(encoding="utf-8")
        chars = len(text)
        words = len(re.findall(r"\w+", text))
        total_chars += chars
        total_words += words
        if chars == 0 or words == 0:
            empty_files.append(f)
        print(f"{f}\t{chars} chars\t{words} mots")

    print("\n--- STATISTIQUES GLOBALES ---")
    print(f"Fichiers totaux        : {len(files)}")
    print(f"Fichiers vides         : {len(empty_files)}")
    if empty_files:
        for f in empty_files:
            print(f"  - {f}")
    print(f"Total mots             : {total_words}")
    print(f"Total caractères       : {total_chars}")


def main():
    parser = argparse.ArgumentParser(description="Vérifie un corpus de pages wiki téléchargées.")
    parser.add_argument("path", help="Répertoire contenant les fichiers .txt du corpus")
    args = parser.parse_args()

    analyze_dir(args.path)


if __name__ == "__main__":
    main()

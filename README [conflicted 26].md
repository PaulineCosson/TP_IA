# TP RAG — Retrieval Augmented Generation (ESIEE)

Ce dépôt contient une implémentation simple d'un pipeline RAG (Retrieval Augmented Generation) basé sur :

- **Corpus** : pages d'un wiki (Dragon Ball Fandom) téléchargées avec `wiki_downloader.py`
- **Chunking** : découpage en passages (chunks) via `chunk_corpus.py`
- **Embeddings + indexation** : `rag.py build` (FAISS + all-MiniLM-L6-v2)
- **Recherche + génération** : `rag.py query` (OpenRouter via `openai`)

---

## Prérequis

1. Créez un environnement Python (recommandé) :

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

3. Ajoutez votre clé OpenRouter (ou `OPENAI_API_KEY`) dans `.env` :

```text
OPENROUTER_API_KEY=sk-or-v1-...
```

> 🚨 **Ne publiez jamais votre clé en clair** dans un dépôt public.

---

## Étapes (TP)

### 1) Construire / vérifier le corpus (déjà fait)

- Téléchargement des pages : `wiki_downloader.py` (étape TP “construire corpus wiki”)
- Vérification du corpus : `python verify_corpus.py corpus/saga_freezer`
- Chunking (découpage) : `python chunk_corpus.py corpus/saga_freezer --out chunks/saga_freezer`

### 2) Indexer le corpus (ingestion / vectorisation)

```bash
python rag.py build --chunks-dir chunks/saga_freezer --index-dir rag_index
```

### 3) Interroger le RAG (retrieval + génération)

```bash
python rag.py query --question "Qui est Freezer ?" --index-dir rag_index
```

Options utiles :
- `--top-k 5` : nombre de chunks retournés
- `--model openai/gpt-5.4-mini` : modèle OpenRouter utilisé

---

## Pour aller plus loin

- Évaluer la qualité du RAG avec des questions/réponses de test
- Utiliser un cross-encoder pour reranker (meilleure précision)
- Ajouter une interface Streamlit ou Gradio

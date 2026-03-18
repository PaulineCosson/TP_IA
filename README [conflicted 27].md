# TP RAG - Dragon Ball Saga Freezer

Ce projet implémente un système RAG (Retrieval Augmented Generation) basé sur le corpus de la Saga Freezer de Dragon Ball, extrait du wiki Fandom.

## Installation

1. Cloner le repo
2. Installer les dépendances :
   ```bash
   pip install langchain langchain-community faiss-cpu sentence-transformers openai python-dotenv
   ```
3. Créer un fichier `.env` avec votre clé API OpenRouter complète :
   ```
   OPENROUTER_API_KEY=sk-or-v1-2def8c4157456ed36e4e4920a1759194cf91bed0d4f2980eb514c7cd0de...
   ```
   (La clé fournie dans le TP est incomplète ; complétez-la avec les 5 derniers caractères donnés en cours.)

## Utilisation

### Construire l'index
```bash
python rag.py build --chunks-dir chunks/saga_freezer --index-dir rag_index
```

### Poser une question
```bash
python rag.py query --index-dir rag_index --question "Comment Goku devient-il Super Saiyan ?"
```

Note : Les questions en anglais peuvent donner de meilleurs résultats.

## Corpus

- Source : Dragon Ball Fandom (wiki français)
- Thème : Saga Freezer
- Chunks : 572 passages (~500 mots chacun)

## Tech Stack

- LangChain pour l'orchestration
- FAISS pour la base vectorielle
- Sentence Transformers (all-MiniLM-L6-v2) pour les embeddings
- OpenRouter pour le LLM (modèle Minimax par défaut)

# Assistant Code du travail RAG

Projet final du module MD5 Data & IA : construire un assistant RAG capable de repondre a des questions sur le droit du travail francais en citant les articles utilises comme sources.

Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection du travail pour votre situation personnelle.

## Objectif

Mettre en place progressivement un pipeline RAG complet :

- constitution et nettoyage d'un corpus juridique ;
- decoupage en chunks coherents ;
- indexation dans une base vectorielle persistante ;
- recherche documentaire ;
- generation de reponses avec citations d'articles ;
- interface en ligne de commande.

## Technologies utilisees

- Python 3.10+
- ChromaDB
- Sentence Transformers
- Groq
- pypdf
- python-dotenv
- pytest

Frameworks RAG interdits pour ce projet : LangChain et LlamaIndex.

## Architecture

Architecture a completer pendant la phase de conception.

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── docs/
├── prompts/
├── src/
└── tests/
```

## Workflow Git

Le projet suit le workflow enseigne dans le TP Scribe :

```text
feature/* -> dev -> main
```

- chaque fonctionnalite est developpee sur une branche `feature/*` ;
- chaque branche `feature/*` fait l'objet d'une Pull Request vers `dev` ;
- `main` ne recoit que les versions validees depuis `dev` ;
- les branches sont conservees jusqu'a la fin du projet pour permettre le controle de l'historique Git.

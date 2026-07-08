# Assistant Code du travail RAG

Projet final du module MD5 Data & IA : construire un assistant RAG capable de repondre
a des questions sur le droit du travail francais en citant les articles utilises comme
sources.

Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou l'inspection
du travail pour votre situation personnelle.

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

Architecture generale cible, a enrichir au fil des branches :

```text
.
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- chroma/
+-- docs/
+-- prompts/
+-- src/
|   +-- config.py
+-- tests/
```

La configuration est centralisee dans `src/config.py`. Elle lit les variables
d'environnement, expose des objets de configuration types et ne demarre aucun service
externe.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copier ensuite le fichier d'exemple :

```bash
copy .env.example .env
```

Les cles API restent locales dans `.env` et ne doivent jamais etre commitees.

## Execution

Les scripts d'indexation et d'interrogation seront ajoutes dans les prochaines branches.
Pour l'instant, seule la configuration applicative est disponible.

## Workflow Git

Le projet suit le workflow enseigne dans le TP Scribe :

```text
feature/* -> dev -> main
```

- chaque fonctionnalite est developpee sur une branche `feature/*` ;
- chaque branche `feature/*` fait l'objet d'une Pull Request vers `dev` ;
- `main` ne recoit que les versions validees depuis `dev` ;
- les branches sont conservees jusqu'a la fin du projet pour permettre le controle de l'historique Git.

## Conventions de developpement

- developper uniquement sur une branche `feature/*` ;
- ouvrir une Pull Request vers `dev` ;
- ne jamais commiter de secrets ;
- tester chaque brique avant de l'integrer ;
- garder les commits courts, atomiques et explicites.

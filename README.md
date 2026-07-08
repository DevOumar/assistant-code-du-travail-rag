# Assistant Code du travail RAG

Projet final du module MD5 Data & IA. L'objectif est de construire un assistant
RAG capable de repondre a des questions sur le droit du travail francais en citant
les articles utilises comme sources.

Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou
l'inspection du travail pour votre situation personnelle.

## Objectifs

- charger et preparer un corpus juridique lie au Code du travail ;
- transformer les textes en documents structures avec metadonnees ;
- decouper les documents en chunks coherents ;
- indexer les chunks dans une base vectorielle persistante ;
- rechercher les chunks pertinents pour une question ;
- generer une reponse sourcee avec citations d'articles ;
- refuser de repondre quand l'information n'est pas dans la base ;
- afficher systematiquement l'avertissement juridique obligatoire ;
- fournir une interface en ligne de commande.

## Etat actuel

Les briques suivantes sont deja preparees :

- configuration applicative centralisee ;
- chunking article-preserving ;
- construction des prompts juridiques ;
- orchestration RAG abstraite par injection de dependances ;
- moderation locale des entrees utilisateur ;
- boucle CLI testable.

Les implementations concretes suivantes sont attendues sur les branches dediees :

- chargement du corpus ;
- parsing des documents ;
- base vectorielle ChromaDB ;
- retrieval Top-k ;
- integration concrete du generateur Groq.

## Technologies

- Python 3.10+
- ChromaDB
- Sentence Transformers
- Groq
- pypdf
- python-dotenv
- pytest

Frameworks RAG interdits : LangChain et LlamaIndex.

## Structure du projet

```text
.
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- chroma/
+-- docs/
+-- prompts/
|   +-- rag_prompt_system.txt
+-- src/
|   +-- config.py
|   +-- chunking.py
|   +-- prompting.py
|   +-- rag.py
|   +-- moderator.py
|   +-- cli.py
+-- tests/
```

## Architecture

`src/config.py` charge les variables d'environnement et expose une configuration
typee. Ce module ne demarre aucun service externe.

`src/chunking.py` recoit des documents deja normalises et produit des chunks. Il ne
lit pas le corpus brut.

`src/prompting.py` assemble les messages systeme et utilisateur a partir d'une
question et de chunks deja retrouves. Le prompt systeme est stocke dans
`prompts/rag_prompt_system.txt` et rendu avec le contexte recupere. Le module ne fait
aucun appel LLM.

`src/rag.py` orchestre le retrieval, la construction du prompt et la generation via
des interfaces injectees. Il ne depend pas directement de ChromaDB ni de Groq.

`src/moderator.py` filtre localement les questions vides, les tentatives evidentes
de prompt injection et les demandes hors perimetre du droit du travail.

`src/cli.py` contient une boucle interactive testable. Le point d'entree concret sera
active lorsque les implementations de retrieval et de generation seront integrees.

## Choix de conception

### Granularite du chunking

Les articles du Code du travail sont courts, denses et doivent rester citables. La
strategie retenue consiste a conserver un article dans un seul chunk lorsqu'il tient
dans la limite configuree.

Si un article est trop long, il est decoupe a l'interieur de l'article, en priorite
sur les paragraphes, puis sur les phrases, avec un leger chevauchement. Cette approche
hybride evite de melanger plusieurs articles tout en gardant des chunks exploitables
pour l'indexation vectorielle.

### Tracabilite

Le numero d'article doit etre conserve dans les metadonnees et rendu visible dans le
contexte fourni au LLM. Le prompt interdit d'inventer des articles et demande de ne
citer que les articles presents dans le contexte.

### Fraicheur du corpus

La configuration prevoit `CORPUS_SOURCE` et `CORPUS_DATE`. La date du corpus est
transmise au prompt afin que le systeme puisse signaler le risque d'obsolescence.

### Reponses conditionnelles

Le prompt demande de signaler les limites quand une reponse depend d'une convention
collective, de la taille de l'entreprise ou d'une situation personnelle.

### Frontiere du conseil juridique

Le systeme doit distinguer une question factuelle couverte par le corpus d'une demande
d'interpretation personnelle. Dans le second cas, il doit rester general, signaler les
limites et rappeler l'avertissement juridique.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copier le fichier d'exemple :

```bash
copy .env.example .env
```

Renseigner ensuite les variables utiles localement. Les cles API ne doivent jamais
etre commitees.

## Configuration

Variables principales :

- `GROQ_API_KEY`
- `GROQ_MODEL`
- `GROQ_TEMPERATURE`
- `GROQ_MAX_TOKENS`
- `RAW_DATA_DIR`
- `PROCESSED_DATA_DIR`
- `CHROMA_DB_DIR`
- `PROMPTS_DIR`
- `EMBEDDING_MODEL_NAME`
- `CHROMA_COLLECTION_NAME`
- `RETRIEVAL_TOP_K`
- `CORPUS_SOURCE`
- `CORPUS_DATE`

## Tests

```bash
pytest
```

Les tests actuels couvrent la configuration, le chunking, la construction de prompts,
l'orchestration RAG abstraite, la moderation et la boucle CLI.

## Execution

L'execution finale dependra des branches d'indexation, retrieval et generation. Pour
l'instant, les modules sont testables independamment.

Le point d'entree CLI actuel indique explicitement que les dependances concretes ne
sont pas encore branchees :

```bash
python src/cli.py
```

## Workflow Git

Workflow impose :

```text
feature/* -> dev -> main
```

- chaque fonctionnalite est developpee sur une branche `feature/*` ;
- chaque branche fait l'objet d'une Pull Request vers `dev` ;
- `main` ne recoit que les versions validees depuis `dev` ;
- les branches sont conservees jusqu'a la fin du projet.

## Repartition des branches

- Preparation donnees : `feature/corpus-loader`, `feature/document-parser`
- Recherche vectorielle : `feature/vector-db`, `feature/retrieval`
- Orchestration : `feature/config`, `feature/chunking`, `feature/prompt`,
  `feature/rag`, `feature/moderator`, `feature/cli`, `feature/readme`

## Conventions de developpement

- ne jamais commiter de secrets ;
- garder les commits courts, atomiques et explicites ;
- tester chaque brique avant Pull Request ;
- ne pas merger sa propre Pull Request sans revue ;
- ne pas utiliser LangChain ni LlamaIndex ;
- conserver les branches pour permettre le controle de l'historique Git.

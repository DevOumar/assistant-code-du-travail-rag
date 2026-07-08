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
|   +-- chunking.py
|   +-- prompting.py
|   +-- rag.py
|   +-- moderator.py
|   +-- cli.py
+-- tests/
```

La configuration est centralisee dans `src/config.py`. Elle lit les variables
d'environnement, expose des objets de configuration types et ne demarre aucun service
externe.

Le chunking est implemente dans `src/chunking.py`. Il recoit des documents deja
normalises par la future branche `feature/document-parser` et ne lit pas directement
le corpus brut.

La preparation des prompts est isolee dans `src/prompting.py`. Elle assemble les
messages systeme et utilisateur a partir d'une question et de chunks deja retrouves,
sans appeler de modele LLM.

L'orchestration RAG est definie dans `src/rag.py`. Elle depend d'interfaces injectees
pour le retrieval et la generation, afin de rester independante des implementations
concretes ChromaDB et Groq.

Le filtrage d'entree est gere dans `src/moderator.py`. Il detecte localement les
questions vides, les tentatives evidentes de prompt injection et les demandes hors
perimetre du droit du travail.

La boucle de ligne de commande est preparee dans `src/cli.py`. Elle orchestre la saisie
utilisateur, la moderation et l'affichage des reponses, mais attend encore les
implementations concretes du retrieval et de la generation.

## Choix de conception

### Granularite du chunking

Les articles du Code du travail sont courts, denses et doivent rester citables. La
strategie retenue est donc de conserver un article dans un seul chunk lorsqu'il tient
dans la limite configuree. Cela maximise la tracabilite : le numero d'article reste
associe a tout le texte transmis aux etapes suivantes.

Si un article est trop long, il est decoupe a l'interieur de cet article, en priorite
sur les paragraphes, puis sur les phrases, avec un leger chevauchement. Cette approche
hybride evite de melanger plusieurs articles tout en gardant des chunks exploitables
pour l'indexation vectorielle.

### Prompt et citations

Le prompt impose de repondre uniquement a partir du contexte transmis par le retrieval.
Chaque source est numerotee et affiche ses metadonnees, notamment le numero d'article
quand il est disponible. Le LLM ne devra citer que les articles presents dans ce
contexte.

Le cas d'echec est explicite : si l'information n'est pas dans la base, la reponse doit
indiquer "Je ne trouve pas cette information dans ma base." L'avertissement juridique
obligatoire est injecte dans le prompt systeme et dans le message utilisateur pour
reduire le risque d'oubli.

### Orchestration RAG

La couche RAG assemble trois responsabilites deja separees : retrieval, construction
du prompt et generation. Elle ne connait pas ChromaDB ni Groq directement. Ces
dependances sont injectees via des interfaces, ce qui permet de tester le comportement
du pipeline avant l'arrivee des implementations concretes.

L'avertissement juridique est garanti une derniere fois dans la couche RAG : si le
generateur l'oublie, il est ajoute avant de retourner la reponse a l'utilisateur.

### Moderation

Le moderateur intervient avant le pipeline RAG. Il ne remplace pas le prompt systeme :
il sert a refuser les entrees manifestement dangereuses ou hors sujet avant toute
recherche ou generation. La premiere version est volontairement deterministe pour
rester explicable en soutenance.

### Interface CLI

La CLI est concue comme une boucle interactive simple : lire une question, appliquer la
moderation, appeler le pipeline RAG injecte, afficher la reponse et les sources. Le point
d'entree concret sera active quand les implementations de retrieval et de generation
seront disponibles.

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
Pour l'instant, la configuration applicative, la brique de chunking, la preparation des
prompts, l'orchestration RAG abstraite, la moderation locale et la boucle CLI testable
sont disponibles.

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

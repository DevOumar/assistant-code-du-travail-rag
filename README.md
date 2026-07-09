# Assistant Code du travail RAG

Projet final du module **MD5 Data & IA**.

L'objectif est de construire un assistant RAG spécialisé dans le droit du travail français.  
L'outil doit répondre à des questions à partir d'un corpus contrôlé, retrouver les articles utiles, citer ses sources et rester honnête quand l'information manque ou quand le corpus est trop ancien.

> Cet assistant ne fournit pas de conseil juridique. Pour une situation personnelle, il faut toujours vérifier avec un professionnel du droit ou l'inspection du travail.

## Présentation du projet

Nous avons choisi une architecture simple à relire et facile à tester :

- un chargement du corpus depuis Legifrance ;
- un parsing des articles en documents propres ;
- un chunking qui reste centré sur les articles du Code du travail ;
- un stockage vectoriel persistant avec ChromaDB ;
- une couche de retrieval capable de gérer les questions simples et composées ;
- un générateur de réponses basé sur Groq ;
- une modération locale pour filtrer les demandes hors sujet ou les tentatives de prompt injection ;
- une interface Streamlit de type chat ;
- une interface CLI pour tester le pipeline sans passer par le web.

Le projet est organisé pour que chaque brique soit lisible séparément. C'était important pour notre équipe, parce que le professeur regarde aussi la qualité du workflow Git et la clarté de l'historique.

## Objectifs fonctionnels

- répondre sur le Code du travail français ;
- renvoyer des réponses sourcées ;
- afficher les articles utilisés ;
- signaler le risque d'obsolescence du corpus ;
- rester poli et clair sur les questions hors périmètre ;
- garder une séparation nette entre préparation des données, retrieval, génération et interface.

## Architecture générale

```text
Question utilisateur
        |
        v
Modération locale
        |
        v
Reformulation / décomposition
        |
        v
Retrieval vectoriel et hybride
        |
        v
Construction du prompt avec contexte
        |
        v
Génération Groq
        |
        v
Réponse sourcée + avertissement juridique
```

## Technologies utilisées

- Python 3.10+
- Streamlit
- ChromaDB
- Sentence Transformers
- Groq
- pytest
- python-dotenv
- pydantic / dataclasses selon les modules

Nous n'utilisons pas LangChain ni LlamaIndex dans ce projet.

## Workflow Git

Le dépôt suit un workflow en quatre niveaux :

1. `feature/<nom>`
2. `feature/orchestrateur`
3. `dev`
4. `main`

Règles appliquées pendant le projet :

- chaque fonctionnalité est développée sur sa propre branche `feature/<nom>` ;
- les branches de travail ne sont pas mélangées entre elles ;
- les Pull Requests sont ouvertes vers `feature/orchestrateur` pendant le développement ;
- `feature/orchestrateur` sert de branche d'intégration technique ;
- une fois validé, le contenu est remonté vers `dev` ;
- `main` reste la branche de référence finale ;
- on garde l'historique complet, sans suppression automatique des branches après merge.

Le principe est simple : on code proprement sur une feature, on vérifie sur la branche d'orchestration, puis on remonte vers `dev`, puis vers `main` quand l'ensemble est validé.

## Structure du projet

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── chroma/
├── docs/
├── prompts/
├── src/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Rôle des dossiers

- `data/raw` : corpus brut récupéré depuis la source ;
- `data/processed` : documents normalisés et prêts à être chunkés ;
- `data/chroma` : persistance de la base vectorielle ;
- `prompts` : prompts système externalisés ;
- `src` : code applicatif ;
- `tests` : tests unitaires et de comportement ;
- `docs` : documents de travail éventuels.

## Architecture technique

### `src/config.py`
Charge la configuration depuis `.env` ou les variables d'environnement.  
On y trouve notamment :

- les chemins projet ;
- la configuration Legifrance ;
- la configuration Groq ;
- la configuration du retrieval ;
- la configuration d'embedding ;
- la date et la source du corpus ;
- l'avertissement juridique.

### `src/corpus_loader.py`
Récupère les articles depuis Legifrance, filtre ceux qui appartiennent au périmètre du sujet et produit un corpus brut exploitable.

### `src/document_parser.py`
Transforme les articles bruts en documents propres et garde les métadonnées utiles :

- numéro d'article ;
- thème ;
- source ;
- date du corpus ;
- identifiant Legifrance.

### `src/chunking.py`
Découpe les documents en chunks tout en essayant de préserver la lisibilité juridique.  
L'idée n'est pas de fragmenter inutilement les articles, mais de garder des unités utiles pour le retrieval.

### `src/vector_store.py`
Gère la base vectorielle persistante :

- chargement du modèle d'embedding ;
- création de la collection ChromaDB ;
- indexation des chunks ;
- requêtes vectorielles ;
- vérification de l'état de l'index.

### `src/question_processing.py`
Nettoie les questions :

- suppression des formulations parasites ;
- normalisation ;
- découpage en sous-questions atomiques.

### `src/question_agents.py`
Centralise la logique de préparation de question :

- reformulation ;
- détection des petits messages de courtoisie ;
- routage vers le retrieval ;
- délégation de la recherche de références.

### `src/retrieval.py`
Adapte les résultats de la base vectorielle au format attendu par le pipeline :

- retrieval Top-k ;
- décomposition des questions composées ;
- fusion des résultats ;
- affichage des chunks.

### `src/prompting.py`
Construit le prompt système et le contexte transmis au générateur.  
Les prompts sont externalisés dans `prompts/` pour rester faciles à relire et à ajuster.

### `src/rag.py`
Orchestre le pipeline de bout en bout :

- récupération des chunks ;
- construction du prompt ;
- génération de la réponse ;
- ajout du disclaimer juridique.

### `src/moderator.py`
Bloque les demandes évidentes hors sujet, les prompts d'injection et les questions trop éloignées du périmètre.

### `src/cli.py`
Permet de tester le pipeline directement dans le terminal.

### `src/web_app.py`
Fournit l'interface Streamlit :

- chat ;
- historique de conversation ;
- corpus ;
- fraîcheur ;
- toggle d'expérimentation ;
- affichage des sources cliquables.

## Fraîcheur du corpus

Le droit du travail évolue. On ne peut pas faire comme si une base de données restait valable indéfiniment.

Le projet affiche donc :

- la source du corpus ;
- sa date ;
- un indicateur d'âge ;
- un risque d'obsolescence.

L'objectif est de prévenir l'utilisateur quand le corpus est ancien ou incomplet, plutôt que de donner une réponse trop sûre d'elle.

## Réponses attendues

Le comportement visé est le suivant :

- si la question est clairement hors sujet, l'assistant reste courtois et recentre vers le droit du travail ;
- si la question est dans le domaine mais sans contexte suffisant, il indique la limite ;
- si la question est composée, il peut la découper ;
- si des articles sont trouvés, il affiche les sources ;
- si rien n'est trouvé, il le dit clairement.

## Installation

Créer un environnement virtuel :

```bash
python -m venv .venv
.venv\Scripts\activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Créer le fichier local d'environnement :

```bash
copy .env.example .env
```

Le fichier `.env` ne doit jamais être poussé sur GitHub.

## Configuration

Exemple de configuration :

```env
APP_ENV=development
GROQ_API_KEY=
GROQ_MODEL=llama-3.1-8b-instant
GROQ_TEMPERATURE=0.2
GROQ_MAX_TOKENS=1024

RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
CHROMA_DB_DIR=data/chroma
PROMPTS_DIR=prompts

EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_COLLECTION_NAME=code_du_travail
RETRIEVAL_TOP_K=5
RETRIEVAL_ENABLE_HYBRID_SEARCH=false

CORPUS_SOURCE=legifrance-sandbox
CORPUS_DATE=2026-07-09
```

## Exécution

Lancer l'indexation complète :

```bash
python src/index_pipeline.py --fetch-raw --parse-raw --recreate
```

Si le corpus brut existe déjà :

```bash
python src/index_pipeline.py --parse-raw --recreate
```

Lancer la CLI :

```bash
python src/cli.py
```

Lancer l'interface Streamlit :

```bash
streamlit run src/web_app.py
```

## Conventions de développement

- garder les commits petits et lisibles ;
- utiliser des messages de commit clairs ;
- ne pas mélanger plusieurs sujets dans le même commit ;
- écrire des tests quand on touche au comportement ;
- conserver les métadonnées utiles dans les chunks ;
- ne pas inventer d'articles ou de sources ;
- garder le ton de l'assistant sobre et professionnel ;
- faire passer `dev` avant `main`.

## État actuel du projet

Le dépôt est déjà avancé. Les briques principales sont en place et le travail se poursuit par étapes :

- préparation des données ;
- retrieval ;
- modération ;
- prompt ;
- interface ;
- amélioration progressive de la qualité des réponses.

L'objectif final n'est pas juste d'avoir un prototype qui tourne, mais un dépôt propre, compréhensible et défendable devant le professeur.

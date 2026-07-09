# Assistant Code du travail RAG

Projet final du module **MD5 Data & IA**.

L'objectif est de construire un assistant RAG capable de répondre à des questions
sur le Code du travail français à partir d'un corpus contrôlé. L'assistant doit
retrouver les passages utiles, générer une réponse sourcée et citer les articles
utilisés.

> Cet assistant ne fournit pas de conseil juridique. Consultez un avocat ou
> l'inspection du travail pour votre situation personnelle.

## État du projet

Le dépôt contient déjà les fondations techniques du projet :

- configuration centralisée avec variables d'environnement ;
- chunking orienté articles du Code du travail ;
- prompts externalisés dans des fichiers texte ;
- orchestration RAG découplée du moteur vectoriel et du LLM ;
- modération locale des questions utilisateur ;
- interface CLI testable ;
- interface web Streamlit sous forme de chat ;
- tests unitaires sur les briques déjà livrées.

Les parties qui dépendent encore de l'intégration finale sont le chargement réel du
corpus, le parsing complet des documents juridiques, l'indexation ChromaDB, le
retrieval Top-k et le client Groq concret.

## Fonctionnement attendu

Le pipeline final suivra ce parcours :

```text
Question utilisateur
        |
        v
Modération locale
        |
        v
Retrieval Top-k dans ChromaDB
        |
        v
Construction du prompt avec contexte et date du corpus
        |
        v
Génération Groq
        |
        v
Réponse avec articles cités, sources et avertissement juridique
```

Le code actuel prépare cette architecture sans dépendre directement des
implémentations concrètes de ChromaDB et Groq. Ces dépendances seront branchées au
moment de l'intégration.

## Technologies

- Python 3.10+
- ChromaDB pour la base vectorielle persistante
- Sentence Transformers pour les embeddings
- Groq pour la génération
- pypdf pour l'extraction de PDF si nécessaire
- python-dotenv pour la configuration locale
- Streamlit pour l'interface web
- pytest pour les tests

Contrainte du sujet : **LangChain et LlamaIndex ne sont pas utilisés**.

## Structure du dépôt

```text
.
+-- data/
|   +-- raw/          # corpus brut
|   +-- processed/    # données normalisées
|   +-- chroma/       # persistance ChromaDB
+-- docs/
+-- prompts/
|   +-- moderator_prompt_system.txt
|   +-- rag_prompt_system.txt
+-- src/
|   +-- config.py
|   +-- chunking.py
|   +-- prompting.py
|   +-- rag.py
|   +-- moderator.py
|   +-- cli.py
|   +-- web_app.py
+-- tests/
```

## Architecture technique

`src/config.py` charge la configuration depuis `.env` ou les variables
d'environnement. Il expose les chemins, le modèle d'embedding, les paramètres de
retrieval, les paramètres Groq, la date du corpus et l'avertissement juridique.

`src/chunking.py` reçoit des documents déjà normalisés et produit des chunks. Le
module cherche à préserver les articles : un article court reste dans un seul chunk,
un article trop long est découpé avec chevauchement.

`src/prompting.py` construit les messages envoyés au LLM. Le prompt système est
externalisé dans `prompts/rag_prompt_system.txt`, ce qui permet de le relire et de le
modifier sans toucher au code.

`src/rag.py` orchestre le retrieval, la construction du prompt et la génération. Il
utilise des interfaces injectées pour éviter de coupler le coeur RAG à ChromaDB ou à
Groq.

`src/moderator.py` bloque les questions vides, les tentatives évidentes de prompt
injection et les demandes hors périmètre du droit du travail français.

`src/cli.py` contient la boucle de discussion en ligne de commande.

`src/web_app.py` fournit une interface Streamlit type chat avec historique, statut du
corpus, affichage du disclaimer et rendu des sources. Tant que le pipeline complet
n'est pas connecté, l'interface indique clairement que le RAG final n'est pas encore
disponible.

## Choix liés au sujet

### Granularité du chunking

Les articles du Code du travail sont denses et doivent rester citables. La stratégie
retenue conserve donc l'article comme unité principale. Si le texte dépasse la taille
maximale configurée, il est découpé en fragments plus petits avec overlap pour ne pas
perdre le contexte.

### Traçabilité

Chaque chunk doit conserver ses métadonnées : source, article, thème et score de
retrieval lorsque disponible. Le prompt interdit d'inventer des articles et demande
de citer uniquement les références présentes dans le contexte.

### Fraîcheur du corpus

Le droit du travail évolue. La configuration prévoit donc `CORPUS_SOURCE` et
`CORPUS_DATE`. Cette date est transmise au prompt afin que l'assistant puisse
indiquer honnêtement le risque d'obsolescence si le corpus est ancien ou non daté.

### Réponses conditionnelles

Certaines réponses dépendent d'éléments non contenus dans le corpus : convention
collective, taille de l'entreprise, statut du salarié, situation personnelle. Le
prompt demande de signaler ces limites au lieu de donner une réponse trop absolue.

### Frontière du conseil juridique

L'assistant doit rester informatif. Il ne doit pas remplacer un avocat, un syndicat,
un service RH ou l'inspection du travail. L'avertissement juridique est centralisé
dans la configuration et ajouté aux réponses.

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

Le fichier `.env` est ignoré par Git et ne doit jamais être poussé.

## Configuration

Variables principales :

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

CORPUS_SOURCE=
CORPUS_DATE=
```

`CORPUS_DATE` doit être renseignée avec la date réelle du corpus utilisé, par exemple
`2026-07-08`.

## Exécution

Lancer la CLI :

```bash
python src/cli.py
```

Lancer l'interface web :

```bash
streamlit run src/web_app.py
```

Pendant la phase actuelle, l'interface web s'ouvre mais affiche un message indiquant
que le pipeline RAG complet n'est pas encore connecté. C'est volontaire : le projet
attend encore l'intégration du corpus, du retrieval et du générateur concret.

## Tests

Lancer la suite de tests :

```bash
pytest
```

Les tests couvrent actuellement la configuration, le chunking, les prompts, la
modération, l'orchestration RAG abstraite, la CLI et l'interface web.

## Corpus Loader (feature/corpus-loader)

### Source retenue

Le corpus est recupere via l'**API Legifrance (environnement sandbox PISTE)**, plutot
qu'un scraping HTML ou un import manuel de PDF. Un test manuel (OAuth2 + un appel
`consult/legiPart`) a confirme que les identifiants du projet sont des identifiants
sandbox : `sandbox-oauth.piste.gouv.fr` et `sandbox-api.piste.gouv.fr` doivent donc
etre utilises tant que des identifiants production ne sont pas fournis.

### Authentification

Le flux est un **OAuth2 client credentials** :

```
POST https://sandbox-oauth.piste.gouv.fr/api/oauth/token
Content-Type: application/x-www-form-urlencoded
grant_type=client_credentials&client_id=...&client_secret=...&scope=openid
```

`src/corpus_loader.py` gere l'obtention et le renouvellement automatique du token :
le jeton est mis en cache et rafraichi automatiquement des qu'il approche de son
expiration (`expires_in`, avec une marge de securite), sans appel reseau superflu
a chaque requete.

### Endpoint utilise : `consult/legiPart`

```
POST https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app/consult/legiPart
{"textId": "LEGITEXT000006072050", "date": "<date du jour, ISO>"}
```

`LEGITEXT000006072050` est l'identifiant du Code du travail. Ce choix (plutot qu'un
`/search` article par article via `typeChamp: NUM_ARTICLE`) permet de recuperer en un
seul appel l'arborescence complete du code (sections, sous-sections, articles), avec
pour chaque article son numero (`num`), son identifiant `LEGIARTI` et son contenu.
Cela evite de multiplier les appels API (un par article) et donne directement la
correspondance numero <-> LEGIARTI pour tout le corpus.

### Themes couverts

Seuls les articles dont le numero appartient aux plages suivantes sont conserves :

- `L3121-1` a `L3121-36` : duree du travail
- `L3141-1` a `L3141-32` : conges payes
- `L1221-1` a `L1248-11` : contrat de travail
- `L1231-1` a `L1237-20` : rupture du contrat de travail
- `L1237-11` a `L1237-19` : rupture conventionnelle

### Format de sortie brute

Le resultat est ecrit tel quel (sans normalisation ni chunking) dans
`data/raw/code_du_travail_raw.json` :

```json
{
  "retrieved_at": "2026-07-08T12:00:00+00:00",
  "source": "legifrance-sandbox",
  "text_id": "LEGITEXT000006072050",
  "endpoint": "https://sandbox-api.piste.gouv.fr/dila/legifrance/lf-engine-app/consult/legiPart",
  "article_count": 123,
  "articles": [
    {
      "num": "L3121-1",
      "id": "LEGIARTI000018487817",
      "content": "<p>...</p>",
      "section_path": ["Partie legislative", "Livre Ier : Duree du travail", "..."]
    }
  ]
}
```

Ce fichier brut sert d'entree a la future branche `feature/document-parser`, qui se
charge de le normaliser vers le format `id`/`text`/`metadata` attendu par
`chunking.py`.

## Workflow Git

Workflow imposé :

```text
feature/* -> dev -> main
```

Règles appliquées dans ce dépôt :

- chaque fonctionnalité est développée sur une branche `feature/*` ;
- chaque branche est poussée sur GitHub ;
- chaque intégration passe par une Pull Request vers `dev` ;
- `main` reste la branche stable ;
- les branches sont conservées jusqu'à la fin du projet pour permettre le contrôle de
  l'historique Git.

Branches du projet :

- `feature/bootstrap`
- `feature/config`
- `feature/corpus-loader`
- `feature/document-parser`
- `feature/chunking`
- `feature/vector-db`
- `feature/retrieval`
- `feature/prompt`
- `feature/rag`
- `feature/moderator`
- `feature/cli`
- `feature/readme`

## Conventions de développement

- ne jamais commiter `.env` ni une clé API ;
- garder des commits courts, atomiques et lisibles ;
- ajouter ou mettre à jour les tests avec chaque brique importante ;
- ne pas mélanger plusieurs responsabilités dans une même branche ;
- ne pas merger directement dans `main` ;
- conserver les messages de commit explicites pour faciliter la revue de l'historique.

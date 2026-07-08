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
+-- tests/
```

La configuration est centralisee dans `src/config.py`. Elle lit les variables
d'environnement, expose des objets de configuration types et ne demarre aucun service
externe.

Le chunking est implemente dans `src/chunking.py`. Il recoit des documents deja
normalises par la future branche `feature/document-parser` et ne lit pas directement
le corpus brut.

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
Pour l'instant, seules la configuration applicative et la brique de chunking sont
disponibles.

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

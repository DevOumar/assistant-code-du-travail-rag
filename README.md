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

## Document Parser (feature/document-parser)

### Role

`src/document_parser.py` lit le corpus brut produit par la branche `corpus-loader`
(`data/raw/code_du_travail_raw.json`, un article par entree avec `num`, `id` LEGIARTI,
`content` HTML et `section_path`) et le transforme en documents normalises au format
`id`/`text`/`metadata` attendu par `chunking.py`. Il ne decoupe rien lui-meme : cette
etape reste la responsabilite de `chunking.py`.

### Nettoyage du HTML

`clean_html()` retire les balises (`p`, `div`, `br`, `table`/`tr`/`td`/`th`, `a`, ...)
sans dependance externe (regex + `html.unescape` de la bibliotheque standard), tout en
preservant les coupures de paragraphes et de lignes de tableau pour garder un texte
lisible, puis normalise les espaces superflus.

### Identifiant et metadonnees

- **Identifiant du document** : construit a partir du numero d'article (`article-L3121-1`),
  stable et lisible pour la tracabilite des citations.
- **Section thematique** : deduite du numero d'article via des plages calquees sur celles
  du `corpus-loader`, evaluees de la plus specifique a la plus large pour que les plages
  imbriquees (`rupture_conventionnelle` et `licenciement` sont toutes deux incluses dans
  `contrat_travail`) resolvent vers le theme le plus precis :
  - `L3121-1` a `L3121-36` : duree_travail
  - `L3141-1` a `L3141-32` : conges_payes
  - `L1237-11` a `L1237-19` : rupture_conventionnelle
  - `L1231-1` a `L1237-20` : licenciement
  - `L1221-1` a `L1248-11` : contrat_travail
- **Autres champs** : `legiarti` (identifiant officiel), `source` et `corpus_date` (issus
  de `config.corpus.source`/`config.corpus.date`), et `title` (derniere section du
  `section_path`, si disponible).

### Sortie et controle qualite

Le resultat est ecrit dans `processed_data_dir` (`data/processed/code_du_travail_documents.json`)
avec un horodatage de generation et le nombre d'articles ignores. Les articles sans `num`
ou `content` exploitable (vide une fois nettoye) sont ignores individuellement, sans faire
echouer l'ensemble du traitement ; un fichier source manquant ou un JSON malforme, en
revanche, font echouer le traitement (`RawCorpusNotFoundError`, `RawCorpusFormatError`).

Executer le module directement (`python src/document_parser.py`) affiche un echantillon
aleatoire de 10 documents pour verification manuelle, comme demande par le sujet.

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

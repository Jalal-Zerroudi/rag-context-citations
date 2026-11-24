# rag-context-citations

Indexation et recherche locales (**.txt / .pdf → chunks → embeddings → FAISS**) avec un mode **EXTRACTION STRICTE** : la réponse doit être **uniquement** composée d’extraits **copiés-collés** depuis les **SOURCES**, et **chaque ligne** doit finir par une citation `[1]`, `[2]`, etc.  
Si le modèle oublie les citations, un post-traitement tente de **recoller automatiquement** le bon `[id]` par matching dans les chunks récupérés.
## 🏗️ Architecture

<p align="center">
  <img src="architecture.png" alt="RAG Architecture" width="900">
</p>

✅ Trois modes sont prévus :
- **Notebook** (`main.ipynb`) : tester rapidement la logique
- **CLI** (`app.py --ask ...`) : interroger le corpus depuis le terminal
- **Serveur HTTP** (`app.py --web`) : exposer l’API JSON

> [!NOTE]
> La route d’interface `/` attend `templates/index.html`, mais ce fichier n’est pas actuellement versionné. L’API `POST /api/ask` reste définie dans `app.py` ; l’interface web complète nécessite l’ajout du template manquant.

---

## ✨ Fonctionnalités

- Lecture de documents **`.txt`** et **`.pdf`** (PDF page par page).
- Découpage en **chunks** avec **overlap** (contexte conservé).
- Embeddings via `sentence-transformers` + **normalisation L2**.
- Recherche top-k via **FAISS** (cosine avec `IndexFlatIP`).
- Mode **strict** :
  - pas d’invention
  - pas de reformulation
  - citations obligatoires
  - validation des titres et des extraits contre les chunks récupérés
  - sinon : `❌ Information non disponible dans mes documents.`
- Mise en forme Markdown :
  - `Titre:` → blocs + sous-puces
  - `X : - A - B` → puces multi-lignes
  - suppression de guillemets ajoutés par certains modèles
- Cache par fichier et configuration (clé SHA256) :
  - `cache/chunks/<cache_key>.jsonl`
  - `cache/embeddings/<cache_key>.npy`
  - `cache/file_hashes.json`

---

## 🗂️ Structure du projet

```text
.
├── .env.example            # modèle de configuration sans secret
├── app.py                  # CLI et serveur Flask
├── architecture.png        # schéma affiché dans ce README
├── main.ipynb              # démonstration interactive
├── requirements.txt
├── data/                   # 7 documents texte d'exemple sur MCP
│   ├── 01_definition_mcp.txt
│   ├── 02_objectifs_mcp.txt
│   ├── ...
│   └── 07_sans_mcp_vs_avec_mcp.txt
├── cache/                  # auto-généré localement et ignoré par Git
│   ├── chunks/
│   ├── embeddings/
│   └── file_hashes.json
└── rag/
    ├── atlascloud.py       # appel du fournisseur LLM
    ├── indexer.py          # chargement, découpage, embeddings et cache
    ├── loaders.py          # extraction TXT/PDF
    └── retriever.py        # recherche vectorielle FAISS
```

---

## ✅ Prérequis

- Python **3.10+** (recommandé)
- Une clé API AtlasCloud : `ATLASCLOUD_API_KEY`

---

## ⚙️ Installation

```bash
# 1) Créer un environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 2) Installer les dépendances
pip install -r requirements.txt
```

---

## 🔐 Configuration (.env)

Copie le modèle versionné, puis renseigne ta propre clé :

```bash
# macOS / Linux
cp .env.example .env
```

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

Le fichier `.env` obtenu contient :

```env
ATLASCLOUD_API_KEY=
ATLAS_MODEL=openai/gpt-oss-20b
```

`ATLASCLOUD_API_KEY` est obligatoire pour générer une réponse. `ATLAS_MODEL` est
optionnel : si la variable est absente, le code utilise
`openai/gpt-oss-20b`. Ne place jamais une vraie clé dans `.env.example` ; seul le
fichier local `.env`, ignoré par Git, doit contenir le secret.


## 🌐 Premier lancement et dépendances réseau

Au démarrage, `build_or_load_index` et `Retriever` initialisent le modèle d'embedding `sentence-transformers/all-MiniLM-L6-v2`. S'il n'est pas déjà présent dans le cache local de Sentence Transformers, son premier chargement peut nécessiter une connexion Internet et prendre plus de temps. Les lancements suivants réutilisent ce cache de modèle ainsi que les chunks et embeddings conservés dans le dossier `cache/`.

Une fois le modèle d'embedding disponible localement, l'indexation et la recherche FAISS peuvent fonctionner sans réseau. La génération d'une réponse appelle toutefois toujours AtlasCloud et nécessite donc une connexion, une clé `ATLASCLOUD_API_KEY` valide et un fournisseur disponible.

`ATLAS_MODEL` sélectionne uniquement le modèle de génération envoyé à AtlasCloud ; cette variable ne change pas le modèle d'embedding. Pour utiliser un autre embedder, modifie de manière cohérente `embedding_model_name` lors de la construction de l'index et `embedder_name` dans `Retriever`, puis reconstruis l'index.

## 🔒 Confidentialité et frontière locale

L'indexation, le calcul des embeddings et la recherche FAISS s'exécutent localement. La génération de la réponse n'est toutefois pas entièrement locale : `app.py` inclut la question, le texte des chunks récupérés et leurs métadonnées dans la requête HTTPS envoyée à `https://api.atlascloud.ai/v1/chat/completions`.

- N'indexe pas de documents confidentiels, personnels ou réglementés sans avoir vérifié que leur transmission à AtlasCloud est autorisée.
- Le mode `strict=True` limite la réponse aux sources récupérées, mais n'empêche pas leur envoi au fournisseur.
- Le serveur écoute par défaut sur `127.0.0.1`. L'API n'intègre ni authentification ni limitation de débit : ne l'expose pas avec `--host 0.0.0.0` sans contrôle réseau, terminaison TLS et authentification en amont.
- Le fichier `.env` est ignoré par Git ; conserve néanmoins `ATLASCLOUD_API_KEY` hors des journaux, captures et fichiers partagés.

---

## 📥 Ajouter tes documents

1) Mets tes fichiers **.txt** et **.pdf** dans `data/`  
2) Lance le notebook ou l’app : le projet va indexer automatiquement  
3) Si tu modifies un fichier, le hash change ⇒ chunks + embeddings sont recalculés

---

## 🧪 Notebook (main.ipynb)

Dans `main.ipynb`, après création du retriever :

```python
q = "Le Model Context Protocol ?"
res = answer_with_rag(retriever, q, topk=6, strict=True)
print(res["answer"])
```

---

## 💻 Mode CLI (terminal)

```bash
python app.py --ask "Le Model Context Protocol ?" --topk 6 --strict
```

Aide :

```bash
python app.py --help
```

---

## 🌐 Serveur API

```bash
python app.py --web --host 127.0.0.1 --port 8000
```

Le serveur expose `POST /api/ask` sur `http://127.0.0.1:8000`. La page `http://127.0.0.1:8000/` ne peut pas être rendue tant que `templates/index.html` n’a pas été ajouté au dépôt.

---

## 🔌 API

### `POST /api/ask`

Body JSON :

```json
{
  "question": "Le Model Context Protocol ?",
  "topk": 6,
  "strict": true
}
```

Exemple `curl` :

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Le Model Context Protocol ?","topk":6,"strict":true}'
```

Réponse :
- `answer` : Markdown + section **Sources consultées**
- `retrieved` : liste des chunks (id, score, fichier, page, preview)

### Validation des requêtes et erreurs

| Entrée | Comportement de l'API |
| --- | --- |
| `question` absente, vide ou composée d'espaces | Réponse `400` avec `{"error":"❌ Question vide"}` |
| `topk` omis | Utilise `6` |
| `topk` convertible en entier | Ramène la valeur dans l'intervalle `1` à `12` avant la recherche |
| `topk` non convertible en entier | Réponse `400` avec un message indiquant l'intervalle accepté |
| `strict` omis | Utilise `true` |
| `strict` autre qu'un booléen JSON | Réponse `400` avec `{"error":"❌ strict doit être un booléen"}` |

Une requête valide renvoie `200` avec `answer` et `retrieved`. Une erreur inattendue
pendant la recherche ou la génération renvoie `500` dans un objet
`error`. Les clients doivent donc vérifier le code HTTP avant de lire les champs
de succès.

---

## 🎛️ Valeurs par défaut selon l’interface

Le mode strict n’a pas la même valeur par défaut selon le point d’entrée :

- **CLI** : le mode strict est désactivé par défaut ; ajoute `--strict` pour limiter la réponse aux extraits vérifiés du corpus.
- **API `POST /api/ask`** : `strict` vaut `true` lorsqu’il est omis. La valeur fournie doit être un booléen JSON (`true` ou `false`).
- **Mode non strict** : le modèle peut compléter la réponse avec ses connaissances générales. Les citations sont demandées, mais elles ne sont pas validées ni imposées par le post-traitement strict.

Pour des réponses reproductibles et traçables uniquement vers les documents indexés, active explicitement le mode strict dans le CLI et conserve `"strict": true` dans les appels API.

---

## 🧠 Règles du mode strict (important)

En `strict=True` :
- La réponse doit être **extraction-only** (copier-coller depuis les sources).
- Chaque ligne doit finir par une citation `[id]`.
- Si aucune info fiable dans les chunks ⇒  
  `❌ Information non disponible dans mes documents.`

---

## 🛠️ Personnalisation

Dans `build_or_load_index(...)` :
- `chunk_size` (défaut: `900`)
- `overlap` (défaut: `150`)
- `embedding_model_name` (défaut: `sentence-transformers/all-MiniLM-L6-v2`)

---

### Découpage et réutilisation du cache

`chunk_size` et `overlap` sont exprimés en **caractères Python**, pas en tokens du modèle. Les espaces et retours à la ligne sont d'abord normalisés ; le découpage utilise ensuite des tranches de texte et peut couper un mot. Les valeurs doivent respecter `chunk_size > 0` et `0 <= overlap < chunk_size`.

La clé de cache combine le **chemin absolu du document**, son **SHA256**, le **nom du modèle d'embedding**, `chunk_size` et `overlap`. Modifier l'un de ces éléments produit une autre clé : les anciens chunks et vecteurs ne sont pas réutilisés pour cette configuration. Le modèle choisi dans `Retriever` doit correspondre à celui utilisé pour construire l'index.

À chaque construction, l'index FAISS est recréé en mémoire à partir des fichiers actuellement présents dans le corpus. Les anciennes entrées de cache ne sont pas automatiquement supprimées du disque. Les fichiers JSONL contiennent le texte des chunks et leurs métadonnées : protégez donc `cache/` comme le corpus source, même si ses noms de fichiers sont hachés.

---

## 🧯 Dépannage

- **ATLASCLOUD_API_KEY manquante** : copie `.env.example` vers `.env`, puis renseigne la clé
- **Index vide** : vérifie `data/` (fichiers `.txt` / `.pdf`)
- **PDF scanné** : pas de texte extractible (OCR non inclus)
- **Cache incohérent** : supprime `cache/` puis relance

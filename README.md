# rag-context-citations

RAG local (**.txt / .pdf → chunks → embeddings → FAISS**) avec un mode **EXTRACTION STRICTE** : la réponse doit être **uniquement** composée d’extraits **copiés-collés** depuis les **SOURCES**, et **chaque ligne** doit finir par une citation `[1]`, `[2]`, etc.  
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
├── app.py
├── main.ipynb
├── requirements.txt
├── data/
│   ├── 01_definition_mcp.txt
│   ├── 02_objectifs_mcp.txt
│   ├── ...
├── cache/                  # auto-généré
│   ├── chunks/
│   ├── embeddings/
│   └── file_hashes.json
└── rag/
    ├── atlascloud.py
    ├── indexer.py
    ├── loaders.py
    └── retriever.py
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

Crée un fichier `.env` à la racine :

```env
ATLASCLOUD_API_KEY=VOTRE_CLE_ICI
# Optionnel
ATLAS_MODEL=openai/gpt-oss-20b
```

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

- **ATLASCLOUD_API_KEY manquante** : vérifie `.env`
- **Index vide** : vérifie `data/` (fichiers `.txt` / `.pdf`)
- **PDF scanné** : pas de texte extractible (OCR non inclus)
- **Cache incohérent** : supprime `cache/` puis relance

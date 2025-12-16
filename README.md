# rag-context-citations

RAG local (**.txt / .pdf → chunks → embeddings → FAISS**) avec un mode **EXTRACTION STRICTE** : la réponse doit être **uniquement** composée d’extraits **copiés-collés** depuis les **SOURCES**, et **chaque ligne** doit finir par une citation `[1]`, `[2]`, etc.  
Si le modèle oublie les citations, un post-traitement tente de **recoller automatiquement** le bon `[id]` par matching dans les chunks récupérés.

✅ Deux façons d’utiliser le projet :
- **Notebook** (`main.ipynb`) : tester rapidement la logique
- **Application Web + CLI** (`app.py`) : UI web + API JSON + mode terminal

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
- Cache par fichier (SHA256) :
  - `cache/chunks/<hash>.jsonl`
  - `cache/embeddings/<hash>.npy`
  - `cache/file_hashes.json`

---

## 🗂️ Structure du projet

```text
.
├── app.py
├── main.ipynb
├── requirements.txt
├── .env
├── data/
│   ├── 01_definition_mcp.txt
│   ├── 02_objectifs_mcp.txt
│   ├── ...
├── cache/                  # auto-généré
│   ├── chunks/
│   ├── embeddings/
│   └── file_hashes.json
├── rag/
│   ├── atlascloud.py
│   ├── indexer.py
│   ├── loaders.py
│   └── retriever.py
└── templates/
    └── index.html
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

## 🌐 Application Web

```bash
python app.py --web --host 127.0.0.1 --port 8000
```

Puis ouvre :

- `http://127.0.0.1:8000/`

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

## 🧯 Dépannage

- **ATLASCLOUD_API_KEY manquante** : vérifie `.env`
- **Index vide** : vérifie `data/` (fichiers `.txt` / `.pdf`)
- **PDF scanné** : pas de texte extractible (OCR non inclus)
- **Cache incohérent** : supprime `cache/` puis relance

---

## 📄 Licence

À définir (MIT / Apache-2.0 / GPL…).

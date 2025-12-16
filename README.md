```md
# rag-context-citations

Un **RAG local** (PDF/TXT → chunks → embeddings → **FAISS**) avec un mode **EXTRACTION STRICTE** : l’assistant ne fait que **copier-coller** des passages des sources et **force des citations** `[1]`, `[2]`… (avec une étape de post-processing qui peut **recoller des citations manquantes** par matching dans les chunks).

✅ Deux modes d’usage :
- **Notebook** (`main.ipynb`) : tester la logique rapidement
- **App Web + CLI** (`app.py`) : UI web + endpoint API JSON

---

## Fonctionnalités
- Indexation de documents **`.txt`** et **`.pdf`** (PDF page par page).
- Découpage en chunks (avec overlap).
- Embeddings via `sentence-transformers` + normalisation L2.
- Recherche top-k avec **FAISS** (cosine via `IndexFlatIP`).
- Mode **strict** : si rien n’est trouvable dans les sources → `❌ Information non disponible dans mes documents.`
- Mise en forme Markdown :
  - titres `xxx:` → blocs + sous-puces
  - conversion de `X : - A - B` → puces multi-lignes
  - suppression des guillemets ajoutés par certains modèles
- Cache par fichier (hash SHA256) pour éviter de recalculer à chaque run.

---

## Arborescence
```

.
├── app.py
├── main.ipynb
├── requirements.txt
├── .env
├── data/
│   ├── 01_definition_mcp.txt
│   ├── 02_objectifs_mcp.txt
│   └── ...
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

````

---

## Prérequis
- Python **3.10+** (recommandé)
- Une clé API AtlasCloud (variable `ATLASCLOUD_API_KEY`)

---

## Installation
```bash
# 1) Créer un venv
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Installer les dépendances
pip install -r requirements.txt
````

---

## Configuration (.env)

Crée un fichier `.env` à la racine :

```env
ATLASCLOUD_API_KEY=xxxxx
# Optionnel
ATLAS_MODEL=openai/gpt-oss-20b
```

---

## Ajouter / modifier des documents

* Mets tes fichiers **.txt** et **.pdf** dans `data/`.
* Au lancement, le projet calcule un **SHA256** :

  * si le fichier change → chunks + embeddings sont recalculés
  * sinon → rechargés depuis `cache/`

---

## Tester dans le notebook (main.ipynb)

1. Ouvre `main.ipynb`
2. Lance les cellules (chargement index, création retriever)
3. Teste une question :

```python
q = "Le Model Context Protocol ?"
res = answer_with_rag(retriever, q, topk=6, strict=True)
print(res["answer"])
```

---

## CLI (terminal)

Exemple en mode strict :

```bash
python app.py --ask "Le Model Context Protocol ?" --topk 6 --strict
```

Aide :

```bash
python app.py --help
```

---

## Lancer l’App Web

```bash
python app.py --web --host 127.0.0.1 --port 8000
```

Puis ouvre :

* UI : `http://127.0.0.1:8000/`

### Endpoints

* `GET /` → page web (`templates/index.html`)
* `POST /api/ask` → JSON

Exemple requête :

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Le Model Context Protocol ?", "topk":6, "strict":true}'
```

Réponse (format) :

* `answer` : réponse markdown + section sources consultées
* `retrieved` : liste des chunks utilisés (id, score, fichier, page, preview)

---

## Comment marche la citation automatique ?

En mode strict, chaque ligne doit finir par `[id]`.
Si le LLM oublie la citation, le post-processing tente de :

* retrouver la ligne dans les chunks récupérés (substring match)
* ajouter automatiquement le bon `[ref_id]`
  Si aucune correspondance n’est trouvée (en strict), la ligne peut être supprimée.

---

## Personnaliser

Dans `build_or_load_index(...)` :

* `chunk_size` (par défaut 900)
* `overlap` (par défaut 150)
* `embedding_model_name` (par défaut `all-MiniLM-L6-v2`)

Dans `chat_stream(...)` :

* `ATLAS_MODEL` via `.env`
* `max_tokens`, `temperature`

---

## Dépannage

* **Erreur clé API** : vérifie `.env` et `ATLASCLOUD_API_KEY`.
* **Index vide** : vérifie que `data/` contient bien des `.txt`/`.pdf`.
* **PDF sans texte** : certains PDFs scannés n’ont pas de texte extractible (il faudrait OCR, non inclus).
* **Cache incohérent** : supprime `cache/` puis relance.

---

## Licence

À définir (MIT / Apache-2.0 / GPL…).

```

Si tu veux, je peux aussi te proposer une version README **plus courte** (style “Quickstart” uniquement) ou une version **plus pro** (badges, roadmap, exemples JSON, etc.).
```

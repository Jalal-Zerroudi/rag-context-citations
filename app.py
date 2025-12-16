from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from rag.indexer import build_or_load_index
from rag.retriever import Retriever
from rag.atlascloud import chat_stream


# ============================================================
# PROMPTS
# ============================================================

SYSTEM_PROMPT_STRICT = """Tu es un assistant RAG en mode EXTRACTION PURE + MISE EN FORME.

RÈGLES ABSOLUES:
1) Tu dois UNIQUEMENT copier-coller des passages EXACTS présents dans les SOURCES.
2) Interdit: reformuler, expliquer, résumer, compléter, déduire ou corriger le texte.
3) Chaque extrait doit avoir une citation [1], [2], etc.
4) Si aucune information exacte dans les SOURCES ne répond: réponds EXACTEMENT
   "❌ Information non disponible dans mes documents."
5) N'ajoute PAS de section Sources à la fin (le serveur gère ça).
6) Maximum 6 extraits.

FORMAT:
- 1 phrase par ligne (court).
- Tu peux garder des lignes "titre:" si elles existent dans les sources.
- Pas de tableaux, pas de guillemets ajoutés.
- Si tu as "X : - A - B", mets chaque "- ..." sur une nouvelle ligne.
"""

SYSTEM_PROMPT_FALLBACK = """Tu es un assistant.
- Réponds en français.
- Si des SOURCES sont fournies, cite [1], [2], etc.
- Sinon réponds avec connaissances générales.
"""


# ============================================================
# TEXT CLEANING (SAFE UTF-8)
# ============================================================

def clean_text(text: str) -> str:
    """Nettoie le texte sans casser les accents. Corrige mojibake seulement si détecté."""
    if not text:
        return text

    if any(x in text for x in ("Ã", "Â", "â€", "â€™", "â€œ", "â€")):
        try:
            fixed = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if fixed:
                text = fixed
        except Exception:
            pass

    text = (text
            .replace("\ufeff", "")
            .replace("\u200b", "")
            )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(retrieved) -> str:
    if not retrieved:
        return "SOURCES:\n(Aucune source trouvée)\n"

    lines = ["SOURCES:\n"]
    for r in retrieved:
        filename = Path(r.chunk.meta.get("path", "unknown")).name
        page = r.chunk.meta.get("page", None)
        page_str = f"page {page}" if page else "toutes pages"
        chunk = clean_text(r.chunk.text)
        lines.append(f"[{r.ref_id}] {filename} ({page_str})\n{chunk}\n")
    return "\n".join(lines)


def run_llm(messages, max_tokens: int = 800, temperature: float = 0.0) -> str:
    out: List[str] = []
    for tok in chat_stream(messages, max_tokens=max_tokens, temperature=temperature):
        out.append(tok)
    return clean_text("".join(out).strip())


# ============================================================
# POST-PROCESSING
#  - strip quotes added by LLM
#  - split ": - ..." into multiline bullets
#  - convert "title:" lines into nested markdown
#  - FIX: add missing citations by matching extracted lines in retrieved chunks
# ============================================================

CIT_RE = re.compile(r"\[(\d{1,2})\]")
CIT_END_RE = re.compile(r"\[(\d{1,2})\]\s*$")
TITLE_RE = re.compile(r"^.{2,160}:\s*$")
QUOTED_LINE_RE = re.compile(r'^\s*["“](.*?)["”]\s*(\[\d{1,2}\])\s*$')


def extract_cited_ids(answer: str) -> List[int]:
    ids = re.findall(r"\[(\d{1,2})\]", answer)
    uniq: List[int] = []
    for x in ids:
        i = int(x)
        if i not in uniq:
            uniq.append(i)
    return uniq


def _strip_line_quotes(line: str) -> str:
    """
    Transform:
      "Texte ..." [1]  ->  Texte ... [1]
    """
    m = QUOTED_LINE_RE.match(line.strip())
    if not m:
        return line.strip()
    return f"{m.group(1).strip()} {m.group(2)}".strip()


def _split_colon_dash_to_bullets(text: str) -> str:
    """
    Transform (format only):
      Avec MCP : - On écrit ... - Les 3 modèles ...
    into:
      Avec MCP :
      - On écrit ...
      - Les 3 modèles ...
    """
    text = text.replace(": -", " :\n- ")
    # aussi " - " -> nouvelle puce si ça ressemble à une liste inline
    text = re.sub(r"\s-\s(?=[A-ZÉÈÀÂÎÔÙÇ])", "\n- ", text)
    return text


def _norm_for_match(s: str) -> str:
    s = clean_text(s)
    s = s.lower()
    s = s.replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _remove_trailing_citation(line: str) -> str:
    return re.sub(r"\s*\[\d{1,2}\]\s*$", "", line).strip()


def _find_best_ref_id_for_line(line_no_cit: str, retrieved) -> Optional[int]:
    """Cherche la ligne dans les chunks (substring). Retourne ref_id si trouvé."""
    target = _norm_for_match(line_no_cit)
    if not target:
        return None

    for r in retrieved:
        chunk = _norm_for_match(r.chunk.text)
        if target in chunk:
            return int(r.ref_id)
    return None


def _ensure_citation(line: str, retrieved, strict: bool) -> Optional[str]:
    """
    Si pas de [id], on essaie de le déduire par matching dans les sources.
    Si impossible en strict -> None (on drop la ligne).
    """
    line = line.strip()
    if not line:
        return None

    if CIT_END_RE.search(line):
        return line

    # pas une phrase (ex: "—" etc.) -> drop en strict
    base = _remove_trailing_citation(line)
    ref_id = _find_best_ref_id_for_line(base, retrieved)
    if ref_id is None:
        return None if strict else line
    return f"{base} [{ref_id}]"


def normalize_extraction_markdown(raw: str) -> str:
    if not raw:
        return raw

    raw = clean_text(raw)
    if raw == "❌ Information non disponible dans mes documents.":
        return raw

    raw = _split_colon_dash_to_bullets(raw)

    # strip quotes line by line
    lines = []
    for ln in raw.splitlines():
        ln2 = _strip_line_quotes(ln)
        if ln2.strip():
            lines.append(ln2.strip())
    return "\n".join(lines).strip()


def structure_nested_markdown(answer: str, retrieved, strict: bool) -> str:
    """
    Output:
    **Réponse extraite**
    - **Titre :**
      - phrase... [1]
      - phrase... [1]
    - **Autre :**
      - ...
    """
    answer = normalize_extraction_markdown(answer)
    if answer == "❌ Information non disponible dans mes documents.":
        return answer

    lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
    if not lines:
        return "❌ Information non disponible dans mes documents."

    # enlever une éventuelle ligne "Réponse extraite" si le modèle l'a mise
    if lines and lines[0].lower().startswith("réponse extraite"):
        lines = lines[1:]

    # build blocks: title -> items
    blocks: List[Dict[str, Any]] = []
    current_title: Optional[str] = None
    current_items: List[str] = []

    def flush():
        nonlocal current_title, current_items
        if current_title is None and not current_items:
            return
        # ensure citations on items
        items_ok: List[str] = []
        for it in current_items:
            it = it.strip()
            if it.startswith("- "):
                it = it[2:].strip()
            it2 = _ensure_citation(it, retrieved, strict=strict)
            if it2:
                items_ok.append(it2)

        if items_ok:
            blocks.append({"title": current_title, "items": items_ok})

        current_title = None
        current_items = []

    for ln in lines:
        # title line ends with ":" and has no citation at end
        if TITLE_RE.match(ln) and not CIT_END_RE.search(ln) and not ln.startswith("- "):
            flush()
            current_title = ln  # keep exact text
            continue
        current_items.append(ln)

    flush()

    # if strict and nothing survived => not available
    if strict and not blocks:
        return "❌ Information non disponible dans mes documents."

    out: List[str] = ["**Réponse extraite**"]

    for b in blocks:
        title = b["title"]
        items = b["items"]

        if title:
            out.append(f"- **{title}**")
            for it in items:
                out.append(f"  - {it}")
        else:
            # no title: simple list
            for it in items:
                out.append(f"- {it}")

    return "\n".join(out).strip()


# ============================================================
# SOURCES FOOTER (server-managed)
# ============================================================
def format_sources_section(retrieved, cited_ids: List[int]) -> str:
    if not retrieved:
        return "\n\n---\n\n📚 **Sources consultées**\n\n❌ Aucune source disponible\n"

    by_id = {r.ref_id: r for r in retrieved}
    ids_cited = [i for i in cited_ids if i in by_id]

    lines = ["\n\n---\n\n📚 **Sources consultées**\n"]
    if ids_cited:
        for i in ids_cited:
            r = by_id[i]
            filename = Path(r.chunk.meta.get("path", "unknown")).name
            page = r.chunk.meta.get("page", None)
            page_str = f"page {page}" if page else "toutes pages"
            lines.append(f"- **[{i}]** `{clean_text(filename)}` ({page_str})")
    else:
        # fallback: show TopK anyway
        for r in retrieved:
            filename = Path(r.chunk.meta.get("path", "unknown")).name
            page = r.chunk.meta.get("page", None)
            page_str = f"page {page}" if page else "toutes pages"
            lines.append(f"- **[{r.ref_id}]** `{clean_text(filename)}` ({page_str})")

    return "\n".join(lines) + "\n"


# ============================================================
# RAG ANSWER
# ============================================================
def answer_with_rag(retriever: Retriever, question: str, topk: int = 6, strict: bool = True) -> Dict[str, Any]:
    question = clean_text(question)
    retrieved = retriever.search(question, topk=topk)

    if strict and not retrieved:
        return {"answer": "❌ Information non disponible dans mes documents.", "retrieved": []}

    context = build_context(retrieved)

    user_prompt = f"""QUESTION: {question}

                        {context}

                        CONSIGNE:
                        - Extrais uniquement des passages EXACTS des SOURCES (copier-coller).
                        - 1 phrase par ligne.
                        - Chaque ligne doit finir par [id] (ex: [1]).
                        - Interdit: reformuler/expliquer.
                        - Si tu as "X : - A - B", mets chaque "- ..." sur une nouvelle ligne.
                        - Si rien: "❌ Information non disponible dans mes documents."
                    """

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT_FALLBACK},
        {"role": "user", "content": user_prompt},
    ]

    raw = run_llm(messages, max_tokens=900, temperature=0.0)

    # ✅ IMPORTANT: on ne renvoie PLUS "❌" juste parce que le LLM a oublié les [id]
    # -> on structure + on recolle les citations par matching dans retrieved
    answer = structure_nested_markdown(raw, retrieved, strict=strict)

    cited = extract_cited_ids(answer)
    if strict and answer != "❌ Information non disponible dans mes documents." and not cited:
        # sécurité finale (normalement rare)
        answer = "❌ Information non disponible dans mes documents."
        cited = []

    sources_section = format_sources_section(retrieved, cited)
    final_answer = answer + sources_section

    retrieved_ui = []
    for r in retrieved:
        filename = clean_text(Path(r.chunk.meta.get("path", "unknown")).name)
        preview = clean_text(r.chunk.text[:220].replace("\n", " "))
        if len(r.chunk.text) > 220:
            preview += "..."
        retrieved_ui.append({
            "id": r.ref_id,
            "score": round(float(getattr(r, "score", 0.0)), 4),
            "path": filename,
            "page": r.chunk.meta.get("page", None),
            "preview": preview,
        })

    return {"answer": final_answer, "retrieved": retrieved_ui}


# ============================================================
# WEB (Flask + Waitress)
# ============================================================
def run_web(retriever: Retriever, root: Path, host: str, port: int):
    try:
        from flask import Flask, jsonify, request, render_template
    except ImportError:
        print("❌ Flask non installé. Fais: pip install flask")
        return

    app = Flask(__name__, template_folder=str(root / "templates"))

    try:
        app.json.ensure_ascii = False  # Flask >= 2.3
    except Exception:
        app.config["JSON_AS_ASCII"] = False

    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def home():
        return render_template("index.html")

    @app.post("/api/ask")
    def api_ask():
        data = request.get_json(force=True, silent=True) or {}
        question = (data.get("question") or "").strip()
        topk = int(data.get("topk") or 6)
        strict = bool(data.get("strict", True))

        if not question:
            return jsonify({"error": "❌ Question vide"}), 400

        topk = max(1, min(12, topk))

        try:
            result = answer_with_rag(retriever, question, topk=topk, strict=strict)
            resp = jsonify(result)
            resp.headers["Content-Type"] = "application/json; charset=utf-8"
            return resp
        except Exception as e:
            return jsonify({"error": f"❌ Erreur: {str(e)}"}), 500

    print(f"\n✅ Web UI: http://{host}:{port}")
    print(f"📁 templates/index.html : {(root / 'templates' / 'index.html')}")
    print(f"📁 data/ : {(root / 'data')}")
    print(f"💾 cache/ : {(root / 'cache')}\n")

    try:
        from waitress import serve
        serve(app, host=host, port=port, threads=6)
    except Exception:
        app.run(host=host, port=port, debug=False, threaded=True)


# ============================================================
# MAIN
# ============================================================
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Local RAG (CLI + Web)")
    parser.add_argument("--ask", type=str, default=None, help="Question (mode CLI)")
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--strict", action="store_true", help="Mode strict (docs uniquement)")
    parser.add_argument("--web", action="store_true", help="Lancer l'interface web")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    root = Path(__file__).parent
    data_dir = root / "data"
    datat_dir = root / "datat"
    cache_dir = root / "cache"

    print("🔄 Chargement index RAG...")
    rag_index = build_or_load_index(data_dir=[data_dir, datat_dir], cache_dir=cache_dir)
    retriever = Retriever(rag_index)
    print("✅ Index prêt.\n")

    if args.web:
        run_web(retriever, root=root, host=args.host, port=args.port)
        return

    if not args.ask:
        print('CLI: python app.py --ask "ta question" --topk 6 --strict')
        print('WEB: python app.py --web --port 8000')
        return

    result = answer_with_rag(retriever, args.ask, topk=args.topk, strict=args.strict)
    print(result["answer"])


if __name__ == "__main__":
    main()

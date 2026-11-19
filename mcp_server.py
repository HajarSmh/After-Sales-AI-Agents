# =========================================================
# mcp_server.py : Serveur MCP du SAV
# Outil publié : rag_multimodal
# Derrière l'outil : un agent RAG (passages + figures) -> réponse sourcée
# Lancement : uv run mcp_server.py   (ou python mcp_server.py)
# .env : GOOGLE_API_KEY=...  (optionnel : GEMINI_MODEL, GEMINI_EMBEDDING_MODEL)
# =========================================================
import base64
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
MANUELS_DIR = BASE / "manuels"
FIGURES_DIR = BASE / "figures"
MANUELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Modèles Gemini (modifiables dans .env : GEMINI_MODEL, GEMINI_EMBEDDING_MODEL)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

# Vision incluse. timeout et max_retries évitent les blocages silencieux.
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0, timeout=60, max_retries=3)


def texte(message) -> str:
    """Retourne le texte d'un message (content peut être une chaîne ou une liste de blocs)."""
    c = message.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


# =========================================================
# 1) Vision : décrire une image
# =========================================================
CACHE = BASE / "descriptions_cache.json"
_cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def decrire_image(path: Path, consigne: str) -> str:
    """Envoie l'image au LLM vision (résultat mis en cache sur disque)."""
    cle = hashlib.md5(path.read_bytes() + consigne.encode()).hexdigest()
    if cle in _cache:
        return _cache[cle]
    b64 = base64.b64encode(path.read_bytes()).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    message = HumanMessage(
        content=[
            {"type": "text", "text": consigne},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ]
    )
    description = texte(llm.invoke([message]))
    _cache[cle] = description
    CACHE.write_text(json.dumps(_cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return description


# =========================================================
# 2) Indexation : passages des manuels + figures
# =========================================================
logger.info("Indexation des manuels...")
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=200
)
passages = PyPDFDirectoryLoader(path=str(MANUELS_DIR)).load_and_split(
    text_splitter=text_splitter
)
for d in passages:
    d.metadata["type"] = "passage"

figures = []
for img in sorted(FIGURES_DIR.glob("*.*")):
    if img.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        continue
    logger.info(f"Description de la figure {img.name}")
    description = decrire_image(
        img,
        "Décris cette figure de manuel d'appareil électronique en français : "
        "pictogrammes, voyants, schémas, légendes et signification éventuelle.",
    )
    figures.append(
        Document(
            page_content=description,
            metadata={"type": "figure", "source": img.name, "ref": img.stem},
        )
    )

documents = passages + figures
if not documents:
    logger.warning("Aucun manuel ni figure trouvé : ajoutez des fichiers dans manuels/ et figures/")
    documents = [
        Document(
            page_content="Base documentaire vide.",
            metadata={"type": "passage", "source": "vide", "page": 0},
        )
    ]
logger.info(f"{len(passages)} passages et {len(figures)} figures indexés")

embedding_model = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="manuels_sav",
)
retriever_passages = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 4, "filter": {"type": "passage"}}
)
retriever_figures = vectorstore.as_retriever(
    search_type="similarity", search_kwargs={"k": 2, "filter": {"type": "figure"}}
)


# =========================================================
# 3) Outils internes de l'agent RAG
# =========================================================
@tool
def rechercher_passages(query: str) -> str:
    """Recherche des passages de texte dans les manuels (avec source et page)."""
    try:
        docs = retriever_passages.invoke(query)
        return "\n\n".join(
            f"[P{i}] ({d.metadata.get('source')}, p.{d.metadata.get('page')}) {d.page_content}"
            for i, d in enumerate(docs, 1)
        ) or "Aucun passage trouvé."
    except Exception as e:
        return f"Erreur de recherche de passages : {e}"


@tool
def rechercher_figures(query: str) -> str:
    """Recherche des figures et pictogrammes dans les manuels (avec fichier source)."""
    try:
        docs = retriever_figures.invoke(query)
        return "\n\n".join(
            f"[F{i}] ({d.metadata.get('source')}) {d.page_content}"
            for i, d in enumerate(docs, 1)
        ) or "Aucune figure trouvée."
    except Exception as e:
        return f"Erreur de recherche de figures : {e}"


rag_system_message = """
Vous êtes un agent RAG spécialisé dans les manuels d'appareils électroniques.
1. Utilisez rechercher_passages ET rechercher_figures avant de répondre.
2. Répondez uniquement à partir des éléments retrouvés.
3. Citez chaque affirmation avec sa source : [P1], [F1], etc.
4. Terminez par une section « Sources » listant fichier et page de chaque élément cité.
Si la réponse ne peut pas être trouvée, répondez : "Je ne sais pas - cette information n'est pas disponible dans les manuels."
"""

rag_agent = create_agent(
    model=llm,
    tools=[rechercher_passages, rechercher_figures],
    system_prompt=rag_system_message,
)

# =========================================================
# 4) Serveur MCP
# =========================================================
mcp = FastMCP(name="sav-mcp-server", host="0.0.0.0", port=23000)


@mcp.tool()
def rag_multimodal(question: str, image_ref: str | None = None) -> str:
    """
    Répond à une question de dépannage à partir des manuels (passages et figures)
    et fournit une réponse sourcée.
    Si le client a joint une photo, passer sa référence dans image_ref (ex: img_px200_01).
    Args:
        question: question du client
        image_ref: référence de la photo jointe (optionnel)
    """
    logger.info(f"rag_multimodal : question={question!r}, image_ref={image_ref!r}")
    contexte = question
    if image_ref:
        fichier = next(FIGURES_DIR.glob(f"{image_ref}.*"), None)
        if fichier is None:
            return f"Image introuvable pour la référence : {image_ref}"
        description = decrire_image(
            fichier,
            "Décris précisément le pictogramme ou le voyant visible sur cette photo d'appareil.",
        )
        contexte += f"\n\nDescription de la photo jointe ({image_ref}) : {description}"
    result = rag_agent.invoke({"messages": [HumanMessage(contexte)]})
    return texte(result["messages"][-1])


if __name__ == "__main__":
    mcp.run(transport="streamable-http")   # http://localhost:23000/mcp
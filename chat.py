"""
chat.py
-------
Esta es la SEGUNDA PARTE del sistema RAG: el asistente conversacional.

Por cada pregunta del usuario:
  1. Convierte la pregunta en un embedding (mismo modelo usado en ingest.py).
  2. Busca en ChromaDB los fragmentos del documento más parecidos a la pregunta.
  3. Construye un prompt con esos fragmentos como "contexto".
  4. Le pide a Gemini que responda SOLO con base en ese contexto.
  5. Muestra la respuesta y, al final, las páginas del documento consultadas.

Se ejecuta tantas veces como quieras hacer preguntas:
    python chat.py
"""

import os
from dotenv import load_dotenv
import chromadb
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ No se encontró GEMINI_API_KEY. Revisa que el archivo se llame exactamente "
        "'.env' (no '.env.example') y que contenga la línea GEMINI_API_KEY=tu_clave."
    )

# ------------------------------------------------------------------
# CONFIGURACIÓN (debe coincidir con la usada en ingest.py)
# ------------------------------------------------------------------
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "resolucion_2674"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
CHAT_MODEL = "gemini-2.5-flash"
N_RESULTS = 5  # Cuántos fragmentos recupera por cada pregunta

# ------------------------------------------------------------------
# IDENTIDAD DEL ASISTENTE (requisito del prompt)
# ------------------------------------------------------------------
NOMBRE_ASISTENTE = "Sanit"
ROL_ASISTENTE = (
    "asistente experto en la Resolución 2674 de 2013 del Ministerio de Salud y "
    "Protección Social de Colombia, que reglamenta los requisitos sanitarios para "
    "la fabricación, procesamiento, envase, almacenamiento, transporte, "
    "distribución, comercialización y expendio de alimentos"
)

SYSTEM_PROMPT = f"""
Eres {NOMBRE_ASISTENTE}, un {ROL_ASISTENTE}.

Reglas que debes seguir SIEMPRE, sin excepción:
1. Responde ÚNICAMENTE con la información contenida en el CONTEXTO entregado en cada
   pregunta. No utilices conocimiento externo ni supuestos propios.
2. Si la respuesta no se encuentra en el CONTEXTO, dilo explícitamente con una frase
   como "No encontré esa información en la Resolución 2674 de 2013." No inventes datos.
3. Responde en español, de forma clara, concisa y profesional. Si el contexto menciona
   artículos o capítulos específicos, mencionálos en tu respuesta.
4. No uses las palabras "contexto" ni "fragmento" en tu respuesta; responde de forma
   natural, como lo haría un experto explicando la norma a otra persona.
"""

# ------------------------------------------------------------------
# CONEXIONES
# ------------------------------------------------------------------
client = genai.Client(api_key=API_KEY)
db = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    coleccion = db.get_collection(COLLECTION_NAME)
except Exception:
    raise SystemExit(
        "❌ No encontré la base vectorial. Ejecuta primero 'python ingest.py' "
        "para procesar el documento antes de usar el chat."
    )


# ------------------------------------------------------------------
# RECUPERACIÓN (Retrieval)
# ------------------------------------------------------------------
def recuperar_contexto(pregunta: str):
    """Convierte la pregunta en un embedding y busca los fragmentos más parecidos."""
    resultado = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=pregunta,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    vector_pregunta = resultado.embeddings[0].values

    resultados = coleccion.query(query_embeddings=[vector_pregunta], n_results=N_RESULTS)
    documentos = resultados["documents"][0]
    metadatas = resultados["metadatas"][0]
    return documentos, metadatas


# ------------------------------------------------------------------
# GENERACIÓN (Augmented Generation)
# ------------------------------------------------------------------
def construir_prompt(pregunta: str, documentos):
    contexto = "\n\n---\n\n".join(documentos)
    return f"""CONTEXTO (fragmentos recuperados del documento):
{contexto}

PREGUNTA DEL USUARIO:
{pregunta}
"""


def responder(pregunta: str) -> str:
    documentos, metadatas = recuperar_contexto(pregunta)
    prompt = construir_prompt(pregunta, documentos)

    respuesta = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
        ),
    )

    paginas = sorted({m["pagina"] for m in metadatas})
    paginas_texto = ", ".join(str(p) for p in paginas)

    return f"{respuesta.text}\n\n📄 Páginas consultadas: {paginas_texto}"


# ------------------------------------------------------------------
# BUCLE CONVERSACIONAL
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"🤖 {NOMBRE_ASISTENTE} — {ROL_ASISTENTE}")
    print("Escribe tu pregunta sobre la Resolución 2674 de 2013.")
    print("Escribe 'salir' para terminar.")
    print("=" * 70)

    while True:
        pregunta = input("\n🧑 Tú: ").strip()
        if pregunta.lower() in {"salir", "exit", "quit"}:
            print("👋 ¡Hasta luego!")
            break
        if not pregunta:
            continue

        respuesta = responder(pregunta)
        print(f"\n{NOMBRE_ASISTENTE}: {respuesta}")


if __name__ == "__main__":
    main()
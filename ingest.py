"""
ingest.py
---------
Este script hace la PRIMERA PARTE del sistema RAG:

  1. Carga el documento PDF (Resolución 2674 de 2013).
  2. Lo divide en fragmentos pequeños ("chunks"), guardando en qué página
     aparece cada fragmento (para poder citar la fuente después).
  3. Genera un embedding (vector numérico) de cada fragmento usando Gemini.
  4. Guarda esos vectores en una base de datos vectorial local (ChromaDB).

Se ejecuta UNA SOLA VEZ (o cada vez que cambies el PDF):
    python ingest.py
"""

import time
import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai
from google.genai import types

# ------------------------------------------------------------------
# 1. CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------
load_dotenv()  # Lee el archivo .env y carga GEMINI_API_KEY como variable de entorno

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise SystemExit(
        "❌ No se encontró GEMINI_API_KEY. Revisa que el archivo se llame exactamente "
        "'.env' (no '.env.example') y que contenga la línea GEMINI_API_KEY=tu_clave."
    )

PDF_PATH = "data/resolucion_2674_2013.pdf"   # Ruta al PDF dentro del proyecto
CHROMA_DIR = "chroma_db"                     # Carpeta donde se guardará la base vectorial
COLLECTION_NAME = "resolucion_2674"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768          # Opciones recomendadas: 768, 1536 o 3072

CHUNK_SIZE = 900             # Caracteres por fragmento
CHUNK_OVERLAP = 150          # Caracteres repetidos entre fragmentos consecutivos
BATCH_SIZE = 50              # Fragmentos enviados juntos a la API en cada llamada
                             # (un lote más grande = menos peticiones = menos riesgo
                             #  de chocar con el límite gratuito de peticiones por minuto)
SLEEP_ENTRE_LOTES = 5        # Segundos de espera entre cada llamada a la API
SLEEP_SI_CUOTA_EXCEDIDA = 60 # Segundos de espera si Google responde "cuota excedida"

# Cliente de la API de Gemini, con la key explícita (evita conflictos con otras
# credenciales de Google que pueda tener configuradas tu sistema)
client = genai.Client(api_key=API_KEY)


# ------------------------------------------------------------------
# 2. CARGA DEL PDF -> TEXTO POR PÁGINA
# ------------------------------------------------------------------
def cargar_paginas(pdf_path: str):
    """Devuelve una lista de tuplas (numero_de_pagina, texto_de_la_pagina)."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(
            f"No encontré el archivo '{pdf_path}'. "
            f"Descarga el PDF de la Resolución 2674 de 2013 y guárdalo en esa ruta."
        )

    reader = PdfReader(pdf_path)
    paginas = []
    for i, page in enumerate(reader.pages, start=1):
        texto = (page.extract_text() or "").strip()
        if texto:
            paginas.append((i, texto))
    return paginas


# ------------------------------------------------------------------
# 3. CHUNKING -> FRAGMENTOS PEQUEÑOS CON SOLAPAMIENTO
# ------------------------------------------------------------------
def dividir_en_chunks(texto: str, chunk_size: int, overlap: int):
    """Divide un texto largo en fragmentos de tamaño chunk_size,
    repitiendo 'overlap' caracteres entre fragmentos consecutivos
    (esto evita cortar una idea justo a la mitad)."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + chunk_size
        chunks.append(texto[inicio:fin])
        inicio += chunk_size - overlap
    return chunks


def construir_chunks(paginas):
    """Convierte cada página en uno o varios chunks, conservando el número de página."""
    chunks = []
    for num_pagina, texto in paginas:
        for fragmento in dividir_en_chunks(texto, CHUNK_SIZE, CHUNK_OVERLAP):
            if fragmento.strip():
                chunks.append({"texto": fragmento, "pagina": num_pagina})
    return chunks


# ------------------------------------------------------------------
# 4. EMBEDDINGS CON GEMINI (con reintento simple por si falla la red/cuota)
# ------------------------------------------------------------------
def generar_embeddings(textos, intentos=5):
    ultimo_error = None
    for intento in range(1, intentos + 1):
        try:
            resultado = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=textos,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            return [e.values for e in resultado.embeddings]
        except Exception as e:
            ultimo_error = e
            mensaje = str(e)
            if "RESOURCE_EXHAUSTED" in mensaje or "429" in mensaje:
                print(f"   ⏳ Cuota gratuita alcanzada (intento {intento}/{intentos}). "
                      f"Esperando {SLEEP_SI_CUOTA_EXCEDIDA}s antes de reintentar...")
                time.sleep(SLEEP_SI_CUOTA_EXCEDIDA)
            else:
                print(f"   ⚠️  Error generando embeddings (intento {intento}/{intentos}): {e}")
                time.sleep(5)
    raise RuntimeError(
        f"No fue posible generar embeddings tras varios intentos. Último error: {ultimo_error}"
    )


# ------------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL
# ------------------------------------------------------------------
def main():
    print(f"📄 Leyendo PDF: {PDF_PATH}")
    paginas = cargar_paginas(PDF_PATH)
    print(f"   -> {len(paginas)} páginas con texto encontradas.")

    print("✂️  Generando chunks...")
    chunks = construir_chunks(paginas)
    print(f"   -> {len(chunks)} fragmentos generados.")

    print("🧠 Conectando con ChromaDB...")
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        db.delete_collection(COLLECTION_NAME)  # Si ya existía, la recreamos desde cero
    except Exception:
        pass
    coleccion = db.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("🔢 Generando embeddings y guardando en la base vectorial...")
    for i in range(0, len(chunks), BATCH_SIZE):
        lote = chunks[i: i + BATCH_SIZE]
        textos = [c["texto"] for c in lote]
        vectores = generar_embeddings(textos)

        ids = [f"chunk_{i + j}" for j in range(len(lote))]
        metadatas = [{"pagina": c["pagina"]} for c in lote]

        coleccion.add(ids=ids, embeddings=vectores, documents=textos, metadatas=metadatas)
        print(f"   -> Procesados {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} fragmentos")
        time.sleep(SLEEP_ENTRE_LOTES)  # Pausa para respetar el límite gratuito de peticiones por minuto

    print("\n✅ Ingesta completa. La base vectorial quedó guardada en la carpeta 'chroma_db/'.")
    print("   Ahora puedes ejecutar: python chat.py")


if __name__ == "__main__":
    main()
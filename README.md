# Sanit — Asistente RAG sobre la Resolución 2674 de 2013

## 👤 Estudiante y fecha
- **Nombre:** _<completa tu nombre>_
- **Fecha:** _<completa la fecha de entrega>_

## 📄 Documento seleccionado y justificación
**Documento:** Resolución 2674 de 2013 del Ministerio de Salud y Protección Social de
Colombia — "Por la cual se reglamenta el artículo 126 del Decreto Ley 019 de 2012 y se
dictan otras disposiciones" (requisitos sanitarios para fabricación, procesamiento,
envase, almacenamiento, transporte, distribución, comercialización y expendio de
alimentos). Categoría: **Salud y seguridad**.

**Justificación:** _<completa: por qué elegiste este documento — por ejemplo, relevancia
para tu carrera/trabajo, interés personal en seguridad alimentaria, disponibilidad del
PDF oficial, extensión adecuada (37 páginas), etc.>_

Fuente oficial del PDF:
https://www.minsalud.gov.co/Normatividad_Nuevo/Resoluci%C3%B3n%202674%20de%202013.pdf

## 🎯 Persona usuaria objetivo y caso de uso
**Persona usuaria objetivo:** _<completa, por ejemplo: propietarios y administradores de
restaurantes, plantas de alimentos o tiendas de víveres en Colombia que necesitan
verificar rápidamente si cumplen con los requisitos sanitarios exigidos por la ley,
sin tener que leer las 37 páginas de la resolución.>_

**Caso de uso:** _<completa, por ejemplo: un encargado de control de calidad usa el
asistente para consultar, en lenguaje natural, requisitos específicos (dotación del
personal manipulador, condiciones de almacenamiento, rotulado, etc.) antes de una
visita de inspección del INVIMA, obteniendo respuestas trazables a artículos y
páginas concretas de la norma.>_

## 💬 Cinco preguntas y respuestas generadas por el sistema
> Ejecuta `python chat.py`, haz estas preguntas y pega aquí la respuesta real que te
> dé el asistente (incluyendo la línea "📄 Páginas consultadas: ...").

**1. ¿Qué se entiende por alimento de mayor riesgo en salud pública según la resolución?**
> _<pega aquí la respuesta del asistente>_

**2. ¿Qué requisitos de dotación e higiene debe cumplir el personal que manipula alimentos?**
> _<pega aquí la respuesta del asistente>_

**3. ¿Qué condiciones deben cumplir los establecimientos para el almacenamiento de alimentos?**
> _<pega aquí la respuesta del asistente>_

**4. ¿Qué información obligatoria debe llevar el rotulado o etiquetado de un alimento?**
> _<pega aquí la respuesta del asistente>_

**5. ¿Qué dice la resolución sobre el transporte aéreo de alimentos? (pregunta fuera del
documento, para probar que el asistente no inventa)**
> _<pega aquí la respuesta — debería indicar que no encontró esa información>_

## ⚙️ Instrucciones para ejecutar el sistema

### 1. Requisitos previos
- Python 3.10 o superior instalado.
- Una cuenta de Google y una API key gratuita de Gemini: créala en
  https://aistudio.google.com/app/apikey

### 2. Clonar el repositorio
```bash
git clone https://github.com/<tu-usuario>/<tu-repo>.git
cd <tu-repo>
```

### 3. Crear y activar un entorno virtual
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 4. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 5. Configurar la API key
```bash
# Copia la plantilla y edita el archivo .env con tu clave real
cp .env.example .env
```
Abre `.env` y reemplaza `tu_api_key_aqui` con tu clave real de Gemini.

### 6. Descargar el documento
Descarga el PDF oficial desde el enlace de Minsalud (sección de arriba) y guárdalo como:
```
data/resolucion_2674_2013.pdf
```

### 7. Orden de ejecución
```bash
# Paso 1: procesar el documento (solo se hace una vez)
python ingest.py

# Paso 2: conversar con el asistente
python chat.py
```

## 🏗️ Arquitectura del sistema
1. **Carga del documento** (`ingest.py`): se extrae el texto del PDF página por página
   con `pypdf`.
2. **Chunking**: cada página se divide en fragmentos de ~900 caracteres con 150 de
   solapamiento, conservando el número de página de origen.
3. **Embeddings**: cada fragmento se convierte en un vector con el modelo
   `gemini-embedding-001` de Google.
4. **Base vectorial**: los vectores se guardan en **ChromaDB** (local, persistente en
   la carpeta `chroma_db/`).
5. **Bucle conversacional** (`chat.py`): cada pregunta del usuario se convierte en un
   embedding, se buscan los fragmentos más similares en ChromaDB, y se envían como
   contexto al modelo `gemini-2.5-flash`, que responde solo con esa información y cita
   las páginas consultadas.

## ⚠️ Limitaciones conocidas
- El asistente solo conoce lo que está en el PDF cargado; no responde sobre normas
  derogatorias, modificaciones posteriores ni jurisprudencia relacionada.
- La extracción de texto con `pypdf` puede fallar en PDFs escaneados como imagen (sin
  texto seleccionable); en ese caso se necesitaría OCR.
- _<agrega aquí la limitación que tú identificaste al construir el sistema, según lo
  pide el video>_

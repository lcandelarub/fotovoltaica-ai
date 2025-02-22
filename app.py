
import streamlit as st
import openai
import os
import json
import pdfplumber

# Configurar la API Key de OpenAI
api_key = os.getenv("OPENAI_API_KEY")

# Configurar el cliente OpenAI
client = openai.OpenAI(api_key=api_key)

# Directorio donde se almacenan los documentos
DOCUMENTS_FOLDER = "documents"

# Función para cargar documentos almacenados
def load_documents():
    knowledge = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".pdf"):
            with pdfplumber.open(os.path.join(DOCUMENTS_FOLDER, filename)) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                knowledge[filename] = text
    return knowledge

# Cargar documentos en memoria
knowledge_base = load_documents()

# Función para buscar en documentos
def search_documents(query):
    results = []
    for doc, content in knowledge_base.items():
        if query.lower() in content.lower():
            results.append((doc, content))
    return results

# Interfaz en Streamlit
st.title("📖 Consulta sobre Energía Fotovoltaica")
st.write("Pregunta sobre los documentos cargados en la base de datos.")

# Entrada del usuario
query = st.text_input("Escribe tu pregunta aquí:")

if query:
    results = search_documents(query)
    
    if results:
        # Construcción del prompt para OpenAI
        context = "\n\n".join([f"Documento: {doc}\nContenido: {content[:2000]}" for doc, content in results])  # Limitar contenido
        prompt = f"Basado en los siguientes documentos sobre energía fotovoltaica, responde la pregunta de forma precisa y utilizando datos textuales exactos:

{context}

Pregunta: {query}

Respuesta:"

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Eres un experto en energía fotovoltaica."},
                          {"role": "user", "content": prompt}],
                max_tokens=200
            )
            st.write("📄 **Fuente del documento:**", results[0][0])
            st.write("💡 **Respuesta:**", response.choices[0].message["content"])
        except Exception as e:
            st.error(f"❌ ERROR: {str(e)}")
    else:
        st.warning("No se encontró información relevante en los documentos almacenados.")

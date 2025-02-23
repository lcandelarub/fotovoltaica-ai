import os
import json
import streamlit as st
import openai
from PyPDF2 import PdfReader

# Cargar clave de API de OpenAI desde secretos de Streamlit
api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=api_key)

# Carpeta donde se almacenan los documentos
DOCUMENTS_FOLDER = "documents"

# Función para cargar y leer PDFs
def load_documents():
    docs = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".pdf"):
            path = os.path.join(DOCUMENTS_FOLDER, filename)
            with open(path, "rb") as f:
                reader = PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                docs[filename] = text
    return docs

# Cargar documentos una vez y almacenarlos en memoria
if "document_texts" not in st.session_state:
    st.session_state.document_texts = load_documents()

# Interfaz de usuario en Streamlit
st.title("Chat sobre Fotovoltaica")

# Entrada del usuario
pregunta = st.text_input("Haz tu pregunta sobre energía fotovoltaica:")

if st.button("Preguntar"):
    if not pregunta:
        st.warning("Por favor, escribe una pregunta.")
    else:
        # Concatenar los textos de los documentos
        context = " ".join(st.session_state.document_texts.values())

        # Crear el prompt para OpenAI
        prompt = f"Basado en los siguientes documentos sobre energía fotovoltaica, responde la pregunta de forma precisa y utilizando datos textuales exactos:\n\n{context}\n\nPregunta: {pregunta}\n\nRespuesta:"

        try:
            response = client.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500
            )
            st.success(response.choices[0].message["content"])
        except openai.OpenAIError as e:
            st.error(f"❌ ERROR inesperado: {e}")

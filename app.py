import streamlit as st
import openai
import os
import json
import pdfplumber

# Configurar clave de API de OpenAI desde secrets
api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

# Función para extraer texto de los PDFs
def extract_text_from_pdfs(pdf_folder):
    text_data = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            with pdfplumber.open(file_path) as pdf:
                text = "
".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                text_data[filename] = text
    return text_data

# Carpeta donde están los documentos almacenados
pdf_folder = "documents"

# Extraer texto de los documentos PDF
document_texts = extract_text_from_pdfs(pdf_folder)

# Guardar los textos en un JSON
json_path = "knowledge.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(document_texts, f, ensure_ascii=False, indent=4)

# Cargar el contenido del JSON en memoria
with open(json_path, "r", encoding="utf-8") as f:
    knowledge_base = json.load(f)

# Interfaz de Streamlit
st.title("Chat de Fotovoltaica AI")
st.write("Haz preguntas sobre energía solar y obtén respuestas basadas en los documentos almacenados.")

# Entrada del usuario
user_question = st.text_input("Introduce tu pregunta:")

if st.button("Preguntar"):
    if user_question:
        combined_text = " ".join(knowledge_base.values())

        # Construir el prompt
        prompt = f"""Basado en el siguiente contenido extraído de documentos sobre fotovoltaica, responde la pregunta de forma clara y estructurada.
        
        CONTENIDO:
        {combined_text}
        
        Pregunta: {user_question}
        
        Respuesta:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "Eres un asistente experto en energía fotovoltaica."},
                          {"role": "user", "content": prompt}]
            )
            answer = response["choices"][0]["message"]["content"]
            st.write("**Respuesta:**", answer)

        except openai.error.OpenAIError as e:
            st.error(f"❌ ERROR de OpenAI:

{str(e)}")

    else:
        st.warning("⚠️ Por favor, introduce una pregunta.")

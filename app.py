import os
import json
import openai
import streamlit as st
from PyPDF2 import PdfReader

# Configuración de la clave API de OpenAI desde Streamlit Secrets
api_key = os.getenv("OPENAI_API_KEY")

# Verificar si la clave API está presente
if not api_key:
    st.error("Error: La clave API de OpenAI no está configurada.")
    st.stop()

# Configurar el cliente de OpenAI
client = openai.OpenAI(api_key=api_key)

# Función para extraer texto de archivos PDF
def extract_text_from_pdfs(pdf_folder):
    extracted_texts = {}
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            pdf_reader = PdfReader(file_path)
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            extracted_texts[file] = text
    return extracted_texts

# Cargar documentos al iniciar
document_folder = "documents"
documents_text = extract_text_from_pdfs(document_folder)

# Guardar el contenido indexado en JSON
index_file_path = os.path.join(document_folder, "knowledge.json")
with open(index_file_path, "w", encoding="utf-8") as index_file:
    json.dump(documents_text, index_file, ensure_ascii=False, indent=4)

# Cargar el índice en memoria
def load_index():
    with open(index_file_path, "r", encoding="utf-8") as index_file:
        return json.load(index_file)

index_data = load_index()

# Interfaz de la aplicación Streamlit
st.title("Asistente de Fotovoltaica")
st.write("Haz preguntas sobre energía fotovoltaica basadas en los documentos cargados.")

# Entrada del usuario
question = st.text_input("Pregunta:")

if question:
    # Concatenar todo el contenido indexado para realizar la consulta
    indexed_text = "\n".join([f"{key}: {value[:2000]}" for key, value in index_data.items()]) # Limitamos el tamaño del contenido

    # Generar respuesta con OpenAI
    prompt = (
        f"Basado en los siguientes documentos sobre energía fotovoltaica, "
        f"responde la pregunta de forma precisa y utilizando datos textuales exactos:

"
        f"Documentos disponibles:
{indexed_text}

"
        f"Pregunta: {question}

"
        f"Respuesta:"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "Eres un experto en energía fotovoltaica."},
                      {"role": "user", "content": prompt}],
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        st.write("### Respuesta:")
        st.write(answer)

    except openai.OpenAIError as e:
        st.error(f"Error con OpenAI: {str(e)}")

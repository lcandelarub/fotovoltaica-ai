import os
import streamlit as st
import openai
import PyPDF2
import json

# Configuración de la clave de API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("No se encontró la clave de API de OpenAI. Asegúrate de configurarla en Streamlit secrets.")
    st.stop()

# Configurar cliente de OpenAI
client = openai.OpenAI(api_key=api_key)

# Definir carpeta de documentos
DOCUMENT_FOLDER = "documents"

# Función para leer y extraer texto de los PDFs almacenados
def cargar_documentos():
    textos = {}
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)

    for file_name in os.listdir(DOCUMENT_FOLDER):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(DOCUMENT_FOLDER, file_name)
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                textos[file_name] = text.strip()
    
    return textos

# Cargar documentos al iniciar
documentos = cargar_documentos()

# Función para responder preguntas en base a los documentos
def responder_pregunta(pregunta, documentos):
    contenido = "\n\n".join([f"{nombre}: {texto[:2000]}" for nombre, texto in documentos.items()])  # Limitamos el contenido para evitar exceder tokens

    if not contenido.strip():
        return "No hay documentos almacenados o no contienen información relevante."

    prompt = f"Basado en el siguiente contenido extraído de documentos sobre fotovoltaica, responde la pregunta de forma clara y estructurada.\n\n{contenido}\n\nPregunta: {pregunta}\n\nRespuesta:"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Eres un experto en energía fotovoltaica."},
                      {"role": "user", "content": prompt}],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"❌ ERROR inesperado: {str(e)}"

# Interfaz de la aplicación con Streamlit
st.title("📡 Asistente Fotovoltaico AI")
st.write("Haz preguntas basadas en documentos almacenados.")

# Campo de entrada de pregunta
pregunta = st.text_input("Escribe tu pregunta aquí:")

if st.button("Buscar respuesta"):
    if pregunta.strip():
        respuesta = responder_pregunta(pregunta, documentos)
        st.write("📖 **Respuesta:**")
        st.write(respuesta)
    else:
        st.warning("Por favor, introduce una pregunta válida.")

import streamlit as st
import PyPDF2
import os
import json
from openai import OpenAI

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CONFIGURAR ARCHIVO DE MEMORIA ---
knowledge_file = "knowledge.json"
if not os.path.exists(knowledge_file):
    with open(knowledge_file, "w") as f:
        json.dump({}, f)

# --- INTERFAZ DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI (Solo responde con documentos)")
st.write("Haz preguntas y obtén respuestas basadas únicamente en los documentos que has subido.")

# --- SUBIR DOCUMENTOS Y GUARDAR SU CONTENIDO ---
st.sidebar.header("📂 Subir nuevos documentos")
uploaded_files = st.sidebar.file_uploader("Sube PDFs para añadir conocimiento", accept_multiple_files=True, type="pdf")

def process_and_store_documents(files):
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)

    for file in files:
        text = extract_text_from_pdf(file)
        knowledge[file.name] = text

    with open(knowledge_file, "w") as f:
        json.dump(knowledge, f, indent=4)

    st.sidebar.success("📚 Documentos añadidos al conocimiento de la IA.")

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

if uploaded_files:
    process_and_store_documents(uploaded_files)

# --- INTERFAZ DE PREGUNTAS ---
query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
if query:
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)

    if not knowledge:
        st.warning("⚠️ No hay documentos cargados. Por favor, sube documentos antes de hacer preguntas.")
    else:
        combined_text = "\n\n".join(knowledge.values())

        prompt = f""" 
        Basado en la siguiente información obtenida exclusivamente de los documentos proporcionados,
        responde la pregunta de forma clara y estructurada. 

        Si la información no está en los documentos, responde con: "No lo sé".

        Pregunta: {query}

        Información disponible:
        {combined_text[:2000]}

        Respuesta:
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Eres un asistente técnico en energía solar fotovoltaica y solo respondes con la información proporcionada en los documentos."},
                    {"role": "user", "content": prompt}
                ]
            )

            generated_response = response.choices[0].message.content

            # Verificar si la respuesta generada contiene información o si debe responder "No lo sé"
            if "No lo sé" in generated_response or len(generated_response.strip()) < 10:
                st.warning("🤔 La información no está en los documentos. Respuesta: 'No lo sé'.")
            else:
                st.subheader("💡 Respuesta generada por IA:")
                st.write(generated_response)

        except Exception as e:
            st.error(f"❌ ERROR inesperado: {str(e)}")


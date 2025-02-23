import streamlit as st
import PyPDF2
import os
import json
import requests
from openai import OpenAI

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CONFIGURAR ARCHIVO DE MEMORIA ---
knowledge_file = "knowledge.json"

def download_pdfs_from_github():
    GITHUB_REPO = "https://raw.githubusercontent.com/lcandelarub/fotovoltaica-ai/main/documents/"
    pdf_files = [file['name'] for file in requests.get('https://api.github.com/repos/lcandelarub/fotovoltaica-ai/contents/documents').json() if file['name'].endswith('.pdf')]  # Lista de archivos en el repo
    
    if not os.path.exists("documents"):
        os.makedirs("documents")
    
    for pdf in pdf_files:
        url = GITHUB_REPO + pdf
        response = requests.get(url)
        if response.status_code == 200:
            with open(os.path.join("documents", pdf), "wb") as f:
                f.write(response.content)
        else:
            st.warning(f"No se pudo descargar {pdf} desde GitHub.")

def process_and_store_documents():
    knowledge = {}
    for file in os.listdir("documents"):
        if file.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join("documents", file))
            knowledge[file] = text
    
    with open(knowledge_file, "w") as f:
        json.dump(knowledge, f, indent=4)

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# --- DESCARGAR Y PROCESAR DOCUMENTOS AL INICIAR ---
download_pdfs_from_github()
process_and_store_documents()

# --- INTERFAZ DE PREGUNTAS ---
st.title("🔆 Chat de Fotovoltaica AI")
st.write("Haz preguntas y obtén respuestas basadas en documentos pre-cargados.")

query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
if query:
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)
    
    combined_text = "\n\n".join(knowledge.values())
    prompt = f"""
    Basado en el siguiente conocimiento almacenado sobre energía solar fotovoltaica,
    responde la pregunta de forma clara, estructurada y precisa.
    
    Pregunta: {query}
    
    Conocimiento disponible:
    {combined_text[:2000]}
    
    Respuesta:
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en energía solar fotovoltaica."},
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_response = response.choices[0].message.content.strip()
        if generated_response:
            st.subheader("💡 Respuesta generada por IA:")
            st.write(generated_response)
        else:
            st.warning("No lo sé.")
    except Exception as e:
        st.error(f"❌ ERROR inesperado: {str(e)}")

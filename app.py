import streamlit as st
import os
import openai
import json
import chromadb

# Configuración de la API Key de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ No se ha encontrado la API Key de OpenAI. Asegúrate de configurarla en Streamlit.")
    st.stop()

client = openai.OpenAI(api_key=api_key)

# Inicializar la base de datos ChromaDB para almacenamiento
db = chromadb.PersistentClient(path="./chromadb_storage")
collection = db.get_or_create_collection(name="documents")

# Función para extraer texto de documentos PDF
import PyPDF2

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

# Procesar y almacenar documentos en la base de datos
def process_and_store_documents():
    docs_path = "./documents"
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    stored_docs = [doc["id"] for doc in collection.get()["documents"]]
    
    for filename in os.listdir(docs_path):
        file_path = os.path.join(docs_path, filename)
        if filename.endswith(".pdf") and filename not in stored_docs:
            text = extract_text_from_pdf(file_path)
            collection.add(documents=[{"id": filename, "content": text}])
            st.sidebar.success(f"📂 Documento {filename} procesado y almacenado.")

# Recuperar información relevante de los documentos
def search_documents(query):
    results = collection.query(query_texts=[query], n_results=3)
    if results["documents"]:
        return results["documents"]
    return []

# Interfaz de usuario con Streamlit
st.title("🔆 Chat IA Fotovoltaica")
st.sidebar.header("📄 Gestión de Documentos")

# Procesar y almacenar documentos en la base de datos
process_and_store_documents()

st.sidebar.write("Los documentos cargados están almacenados y serán utilizados para responder tus preguntas.")

question = st.text_input("❓ Haz tu pregunta sobre fotovoltaica:")

if question:
    results = search_documents(question)
    if results:
        selected_doc = results[0]
        context = selected_doc["content"][:4000]  # Limitar la cantidad de texto procesado
        prompt = f"Basado en el siguiente contenido extraído de un documento sobre fotovoltaica, responde la pregunta de forma clara y estructurada. 

{context}

Pregunta: {question}

Respuesta:"
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "system", "content": "Eres un experto en energía fotovoltaica."},
                          {"role": "user", "content": prompt}],
                max_tokens=300
            )
            st.success(response.choices[0].message["content"])
        except Exception as e:
            st.error(f"❌ ERROR inesperado: {str(e)}")
    else:
        st.warning("La información no está en los documentos. Respuesta: 'No lo sé'.")

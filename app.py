import streamlit as st
import PyPDF2
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import pickle

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CARGAR EL MODELO DE EMBEDDINGS PARA INDEXAR CONOCIMIENTO ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- CONFIGURACIÓN DE LA BASE DE DATOS VECTORIAL ---
index_file = "knowledge_index.faiss"
doc_store_file = "document_store.pkl"

if os.path.exists(index_file) and os.path.exists(doc_store_file):
    index = faiss.read_index(index_file)
    with open(doc_store_file, "rb") as f:
        document_store = pickle.load(f)
else:
    index = faiss.IndexFlatL2(384)  # Dimensión de embeddings
    document_store = {}

# --- INTERFAZ DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI con Memoria")
st.write("Haz preguntas y obtén respuestas basadas en el conocimiento almacenado.")

# --- SUBIR DOCUMENTOS Y PROCESARLOS ---
st.sidebar.header("📂 Subir nuevos documentos")
uploaded_files = st.sidebar.file_uploader("Sube PDFs para añadir conocimiento", accept_multiple_files=True, type="pdf")

def process_and_store_documents(files):
    global index, document_store
    for file in files:
        text = extract_text_from_pdf(file)
        doc_embedding = embedding_model.encode(text, convert_to_numpy=True)

        # Guardar en la base de datos
        document_store[file.name] = text
        index.add(np.array([doc_embedding]))

    # Guardar la base de datos actualizada
    faiss.write_index(index, index_file)
    with open(doc_store_file, "wb") as f:
        pickle.dump(document_store, f)

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

if uploaded_files:
    process_and_store_documents(uploaded_files)
    st.sidebar.success("📚 Documentos añadidos al conocimiento de la IA.")

# --- INTERFAZ DE PREGUNTAS ---
query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
if query:
    query_embedding = embedding_model.encode(query, convert_to_numpy=True)
    D, I = index.search(np.array([query_embedding]), k=3)  # Buscar en la base de datos vectorial

    retrieved_docs = [list(document_store.values())[i] for i in I[0] if i < len(document_store)]
    combined_text = "\n\n".join(retrieved_docs)

    prompt = f"""
    Basado en el siguiente conocimiento almacenado en la IA sobre energía solar fotovoltaica,
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

        st.subheader("💡 Respuesta generada por IA:")
        st.write(response.choices[0].message.content)
    except Exception as e:
        st.error(f"❌ ERROR inesperado: {str(e)}")


import streamlit as st
import PyPDF2
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CARGAR EL MODELO DE EMBEDDINGS PARA INDEXAR CONOCIMIENTO ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- CONFIGURACIÓN DE LA BASE DE DATOS VECTORIAL (ChromaDB) ---
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("fotovoltaica_knowledge")

# --- INTERFAZ DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI con Memoria")
st.write("Haz preguntas y obtén respuestas basadas en el conocimiento almacenado.")

# --- SUBIR DOCUMENTOS Y PROCESARLOS ---
st.sidebar.header("📂 Subir nuevos documentos")
uploaded_files = st.sidebar.file_uploader("Sube PDFs para añadir conocimiento", accept_multiple_files=True, type="pdf")

def process_and_store_documents(files):
    for file in files:
        text = extract_text_from_pdf(file)
        doc_embedding = embedding_model.encode(text).tolist()

        # Guardar en ChromaDB
        collection.add(documents=[text], embeddings=[doc_embedding], ids=[file.name])

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
    query_embedding = embedding_model.encode(query).tolist()
    search_results = collection.query(query_embeddings=[query_embedding], n_results=3)

    retrieved_docs = search_results["documents"][0] if search_results["documents"] else []
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


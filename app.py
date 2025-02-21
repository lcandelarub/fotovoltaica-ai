import streamlit as st
import PyPDF2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- CONFIGURACIÓN DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI")
st.write("Haz preguntas sobre energía solar y obtén respuestas basadas en la información cargada.")

# --- FUNCIÓN PARA EXTRAER TEXTO DE PDF ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# --- CARGA AUTOMÁTICA DE DOCUMENTOS DESDE GITHUB ---
st.sidebar.header("📂 Documentos disponibles")
document_folder = "documents"  # Carpeta donde estarán los PDFs dentro del repositorio

# Obtener la lista de documentos en la carpeta
pdf_files = [f for f in os.listdir(document_folder) if f.endswith(".pdf")]

# Diccionario para almacenar el texto de los documentos
documents_text = {}
for pdf in pdf_files:
    pdf_path = os.path.join(document_folder, pdf)
    documents_text[pdf] = extract_text_from_pdf(pdf_path)

# Mostrar los documentos cargados en la barra lateral
st.sidebar.write("Los siguientes documentos están en la base de datos:")
for doc_name in documents_text.keys():
    st.sidebar.write(f"📄 {doc_name}")

# --- PROCESAR DOCUMENTOS Y CONFIGURAR BÚSQUEDA ---
if documents_text:
    docs = list(documents_text.values())
    doc_names = list(documents_text.keys())

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)

    def search_query(query, top_n=1):
        """Busca la consulta en los documentos y devuelve el más relevante."""
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        results = [(doc_names[i], similarities[i]) for i in top_indices]
        return results

    # --- INTERFAZ DE CHAT ---
    query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
    if query:
        results = search_query(query)
        best_match, confidence = results[0]
        st.subheader(f"📄 Documento más relevante: {best_match}")
        st.write(f"🔎 Coincidencia: {confidence:.2%}")
        st.write("💡 Respuesta basada en el documento:")
        st.write(documents_text[best_match][:1000])  # Muestra los primeros 1000 caracteres

else:
    st.sidebar.warning("🔺 No hay documentos disponibles. Sube archivos a GitHub para que sean accesibles.")

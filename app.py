import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2

# --- CONFIGURACIÓN DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI")
st.write("Haz preguntas sobre energía solar y obtén respuestas basadas en los documentos subidos.")

# --- FUNCIÓN PARA EXTRAER TEXTO DE PDF ---
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# --- CARGA DE DOCUMENTOS ---
st.sidebar.header("📂 Subir Documentos")
uploaded_files = st.sidebar.file_uploader("Sube tus PDFs aquí", accept_multiple_files=True, type=["pdf"])

# Procesar documentos
documents_text = {}
if uploaded_files:
    for file in uploaded_files:
        documents_text[file.name] = extract_text_from_pdf(file)

    # Vectorizar documentos
    docs = list(documents_text.values())
    doc_names = list(documents_text.keys())

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)

    # --- BÚSQUEDA SEMÁNTICA ---
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
    st.sidebar.warning("🔺 Sube al menos un documento para hacer consultas.")

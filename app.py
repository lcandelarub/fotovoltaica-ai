import streamlit as st
import PyPDF2
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# --- CONFIGURACIÓN DE LA APP ---
st.title("🔆 Chat de Fotovoltaica AI")
st.write("Haz preguntas sobre energía solar y obtén respuestas basadas en toda la información cargada.")

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
document_folder = "documents"

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

# --- PROCESAR DOCUMENTOS Y CONFIGURAR BÚSQUEDA MEJORADA ---
if documents_text:
    docs = list(documents_text.values())
    doc_names = list(documents_text.keys())

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(docs)

    def search_query(query, top_n=3):
        """Busca la consulta en los documentos y devuelve los fragmentos más relevantes."""
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        results = [(doc_names[i], similarities[i], documents_text[doc_names[i]]) for i in top_indices]
        return results

    # --- INTERFAZ DE CHAT ---
    query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
    if query:
        results = search_query(query, top_n=3)  # Ahora analiza varios documentos
        combined_text = "\n\n".join([f"📄 {doc}:\n{text[:2000]}" for doc, _, text in results])

        prompt = f"""
        Basado en la siguiente información recopilada de varios documentos técnicos de energía solar fotovoltaica,
        responde la pregunta de forma clara, estructurada y precisa. Usa toda la información relevante para generar
        una respuesta lo más completa posible.

        Pregunta: {query}

        Información disponible:
        {combined_text}

        Respuesta:
        """

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Puedes cambiar a "gpt-4-turbo" si tienes acceso
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en energía solar fotovoltaica."},
                    {"role": "user", "content": prompt}
                ]
            )

            st.subheader("📄 Documentos utilizados para la respuesta:")
            for doc, confidence, _ in results:
                st.write(f"🔎 {doc} (Relevancia: {confidence:.2%})")

            st.write("💡 Respuesta generada por IA:")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ ERROR inesperado: {str(e)}")

else:
    st.sidebar.warning("🔺 No hay documentos disponibles. Sube archivos a GitHub para que sean accesibles.")

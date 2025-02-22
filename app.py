import streamlit as st
import pdfplumber
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
st.title("🔆 Chat de Fotovoltaica AI (Mejor Búsqueda)")
st.write("Haz preguntas y obtén respuestas basadas únicamente en los documentos que has subido.")

# --- SUBIR DOCUMENTOS Y GUARDAR SU CONTENIDO ---
st.sidebar.header("📂 Subir nuevos documentos")
uploaded_files = st.sidebar.file_uploader("Sube PDFs para añadir conocimiento", accept_multiple_files=True, type="pdf")

def process_and_store_documents(files):
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)

    for file in files:
        text_fragments = extract_text_from_pdf(file)
        knowledge[file.name] = text_fragments

    with open(knowledge_file, "w") as f:
        json.dump(knowledge, f, indent=4)

    st.sidebar.success("📚 Documentos añadidos al conocimiento de la IA.")
    return knowledge

def extract_text_from_pdf(pdf_file):
    fragments = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                words = page_text.split()
                for i in range(0, len(words), 100):  # Fragmentos más grandes para mejorar contexto
                    fragment = " ".join(words[i:i+100])
                    fragments.append(fragment)
    return fragments

if uploaded_files:
    knowledge_data = process_and_store_documents(uploaded_files)

    # 📌 MOSTRAR EN PANTALLA TODO EL CONTENIDO EXTRAÍDO PARA DEPURACIÓN
    st.subheader("📂 CONTENIDO EXTRAÍDO DEL DOCUMENTO:")
    st.write(knowledge_data)

# --- INTERFAZ DE PREGUNTAS ---
query = st.text_input("🔍 Escribe tu pregunta sobre fotovoltaica:")
if query:
    with open(knowledge_file, "r") as f:
        knowledge_data = json.load(f)

    if not knowledge_data:
        st.warning("⚠️ No hay documentos cargados. Por favor, sube documentos antes de hacer preguntas.")
    else:
        # Filtramos fragmentos que tengan palabras clave de la pregunta
        relevant_fragments = []
        query_words = set(query.lower().split())

        for doc in knowledge_data.values():
            for fragment in doc:
                # Priorizamos los fragmentos que contienen términos clave y números
                if any(word in fragment.lower() for word in query_words) or any(char.isdigit() for char in fragment):
                    relevant_fragments.append(fragment)

        if not relevant_fragments:
            st.warning("🤔 La información no está en los documentos. Respuesta: 'No lo sé'.")
        else:
            combined_text = "\n\n".join(relevant_fragments[:10])  # Máximo 10 fragmentos para análisis

            prompt = f""" 
            Basado exclusivamente en la siguiente información obtenida de los documentos subidos,
            responde la pregunta de forma clara y estructurada. 

            Si la respuesta no está en los documentos, responde con: "No lo sé".
            
            Prioriza los datos numéricos y porcentajes si están disponibles.

            Pregunta: {query}

            Información disponible:
            {combined_text}

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

                if "No lo sé" in generated_response or len(generated_response.strip()) < 10:
                    st.warning("🤔 La información no está en los documentos. Respuesta: 'No lo sé'.")
                else:
                    st.subheader("💡 Respuesta generada por IA:")
                    st.write(generated_response)

            except Exception as e:
                st.error(f"❌ ERROR inesperado: {str(e)}")


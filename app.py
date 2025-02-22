import streamlit as st
import openai
import os
import json
import PyPDF2

# Configuración de la API de OpenAI
api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=api_key)

# Carpeta donde se almacenan los documentos en GitHub
DOCUMENTS_FOLDER = "documents"

# Función para leer y procesar documentos PDF
def load_documents():
    knowledge_base = {}
    for filename in os.listdir(DOCUMENTS_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(DOCUMENTS_FOLDER, filename)
            with open(file_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                knowledge_base[filename] = text
    return knowledge_base

# Cargar documentos al iniciar la app
knowledge = load_documents()

# Guardar el conocimiento en un archivo JSON para referencia
knowledge_json = "knowledge.json"
with open(knowledge_json, "w") as json_file:
    json.dump(knowledge, json_file)

# Interfaz de usuario con Streamlit
st.title("Fotovoltaica AI 📡☀️")

# Mostrar documentos cargados
st.sidebar.header("📂 Documentos en memoria")
for doc in knowledge.keys():
    st.sidebar.write(f"📄 {doc}")

# Input del usuario
query = st.text_input("Haz una pregunta sobre fotovoltaica:")

if st.button("Buscar respuesta"):
    if not query:
        st.warning("Por favor, ingresa una pregunta.")
    else:
        # Buscar información en los documentos
        context = "\n".join(list(knowledge.values()))[:5000]  # Limitar caracteres para evitar errores

        # Generar respuesta con OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": "Responde solo con información de los documentos proporcionados. Si no sabes la respuesta, di 'No lo sé'."},
                      {"role": "user", "content": f"Contexto: {context}\n\nPregunta: {query}"}]
        )

        answer = response.choices[0].message.content

        # Mostrar respuesta
        st.subheader("📜 Respuesta:")
        st.write(answer)

import PyPDF2
import os
import json
import requests
import openai

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("❌ ERROR: No se encontró la API Key de OpenAI. Configúrala en el entorno de ejecución.")

client = openai

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
            print(f"⚠️ Advertencia: No se pudo descargar {pdf} desde GitHub.")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text.split("\n\n")  # Dividir el texto en párrafos

def process_and_store_documents():
    knowledge = {}
    for file in os.listdir("documents"):
        if file.endswith(".pdf"):
            paragraphs = extract_text_from_pdf(os.path.join("documents", file))
            for i, paragraph in enumerate(paragraphs):
                key = f"{file}_part_{i}"
                knowledge[key] = paragraph.strip()
    
    with open(knowledge_file, "w") as f:
        json.dump(knowledge, f, indent=4)

# --- DESCARGAR Y PROCESAR DOCUMENTOS AL INICIAR ---
download_pdfs_from_github()
process_and_store_documents()

# --- INTERFAZ DE PREGUNTAS ---
print("🔆 Chat de Fotovoltaica AI")
print("Haz preguntas y obtén respuestas basadas en documentos pre-cargados.")

query = input("🔍 Escribe tu pregunta sobre fotovoltaica: ")
if query:
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)
    
    relevant_fragments = [text for key, text in knowledge.items() if query.lower() in text.lower()]
    combined_text = "\n\n".join(relevant_fragments[:5])  # Limitar a 5 fragmentos relevantes
    prompt = f"""
    Eres un asistente experto en energía solar fotovoltaica. Usa la siguiente información almacenada en documentos técnicos para responder a la pregunta del usuario.

    ### Pregunta del usuario:
    {query}

    ### Información relevante encontrada:
    {combined_text}

    ### Instrucciones para responder:
    - Si encuentras información útil, responde de forma estructurada y con detalles técnicos.
    - Si no encuentras información relevante en los documentos, responde con 'No lo sé'.
    - No inventes información que no esté en los documentos.

    ### Respuesta esperada:
    """
    
    try:
        response = client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en energía solar fotovoltaica."},
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_response = response["choices"][0]["message"]["content"].strip()
        if generated_response:
            print("💡 Respuesta generada por IA:")
            print(generated_response)
        else:
            print("⚠️ No lo sé.")
    except Exception as e:
        print(f"❌ ERROR inesperado: {str(e)}")

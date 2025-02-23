import PyPDF2
import os
import json
import requests

# --- CONFIGURAR LA API KEY DESDE SECRETS ---
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("⚠️ Advertencia: No se encontró la API Key de OpenAI. Solo se mostrarán fragmentos relevantes sin generar respuestas con IA.")
    client = None
else:
    try:
        import openai
        client = openai
        client.api_key = api_key
    except ModuleNotFoundError:
        print("⚠️ Advertencia: El módulo 'openai' no está disponible. Solo se mostrarán fragmentos relevantes sin generar respuestas con IA.")
        client = None

# --- CONFIGURAR ARCHIVO DE MEMORIA ---
knowledge_file = "knowledge.json"

def download_pdfs_from_github():
    GITHUB_REPO = "https://raw.githubusercontent.com/lcandelarub/fotovoltaica-ai/main/documents/"
    if not os.path.exists("documents"):
        os.makedirs("documents")
    try:
        response = requests.get('https://api.github.com/repos/lcandelarub/fotovoltaica-ai/contents/documents')
        if response.status_code != 200:
            print(f"⚠️ Advertencia: No se pudo acceder a la API de GitHub. Código de estado: {response.status_code}")
            return
        files = response.json()
        pdf_files = [file['name'] for file in files if 'name' in file and file['name'].endswith('.pdf')]
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"⚠️ Error al obtener la lista de archivos desde GitHub: {e}")
        return
    
    for pdf in pdf_files:
        url = GITHUB_REPO + pdf
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join("documents", pdf), "wb") as f:
                    f.write(response.content)
            else:
                print(f"⚠️ Advertencia: No se pudo descargar {pdf} desde GitHub.")
        except requests.RequestException as e:
            print(f"⚠️ Error al descargar {pdf}: {e}")

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
    if not os.path.exists("documents"):
        print("⚠️ Advertencia: La carpeta 'documents' no existe. No se pueden procesar archivos.")
        return
    
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

# --- EJEMPLO DE CONSULTA AUTOMÁTICA ---
def auto_query(query):
    print(f"🔍 Consulta automática: {query}")
    with open(knowledge_file, "r") as f:
        knowledge = json.load(f)
    
    relevant_fragments = [text for key, text in knowledge.items() if query.lower() in text.lower()]
    combined_text = "\n\n".join(relevant_fragments[:5])  # Limitar a 5 fragmentos relevantes
    
    if client:
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
    else:
        print("⚠️ No se pudo generar una respuesta con IA. Solo se muestran los fragmentos más relevantes encontrados.")
        print(combined_text if combined_text else "⚠️ No se encontró información relevante en los documentos.")

# --- EJEMPLO DE USO ---
auto_query("¿Cómo funcionan los paneles solares?")

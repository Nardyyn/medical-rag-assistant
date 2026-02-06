import fitz  # PyMuPDF
import base64
import os

def process_pdf(uploaded_file):
    """Extrait le texte d'un fichier PDF."""
    try:
        # On lit le flux binaire du fichier
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text if text.strip() else "Le PDF semble vide ou est une image scannée."
    except Exception as e:
        return f"Erreur lors de la lecture du PDF : {e}"

def encode_image(uploaded_file):
    """Encode une image en base64 pour le modèle Vision."""
    try:
        return base64.b64encode(uploaded_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Erreur d'encodage image : {e}")
        return None

def save_to_data_folder(uploaded_file):
    """Sauvegarde temporaire dans le dossier data pour archivage ou cache."""
    if not os.path.exists("data"):
        os.makedirs("data")
    
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
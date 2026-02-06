import streamlit as st
import os
from utils import process_pdf, encode_image, save_to_data_folder
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIGURATION ET SÉCURITÉ ---
st.set_page_config(page_title="IA Médicale Pro", layout="wide", page_icon="🩺")

# Récupération des clés API
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# Initialisation du modèle VISION
# Ce modèle Llama 4 Scout est optimisé pour ne pas refuser l'analyse d'image
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", groq_api_key=GROQ_API_KEY)
search = TavilySearchResults(max_results=3)

# --- 2. GESTION DE LA MÉMOIRE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "image_data" not in st.session_state:
    st.session_state.image_data = None

st.title("🩺 Assistant Médical Expert Multimodal")
st.write("Analyse de PDF, Vision d'ordonnances et Recherche Web en temps réel.")

# --- 3. BARRE LATÉRALE : GESTION DES FICHIERS ---
with st.sidebar:
    st.header("📂 Documents & Analyses")
    uploaded_file = st.file_uploader(
        "Scanner un PDF ou une image", 
        type=['pdf', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        st.success(f"Fichier chargé : {uploaded_file.name}")
        save_to_data_folder(uploaded_file)
        
        # Traitement PDF
        if uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            with st.spinner("Extraction du texte PDF..."):
                st.session_state.pdf_context = process_pdf(uploaded_file)
            st.info("✅ Texte du PDF mémorisé.")
            
        # Traitement Image
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            with st.spinner("Analyse de l'image..."):
                # On rembobine le fichier avant encodage
                uploaded_file.seek(0)
                st.session_state.image_data = encode_image(uploaded_file)
            st.info("✅ Image prête pour analyse visuelle.")
            st.image(uploaded_file, caption="Aperçu de l'image", use_container_width=True)
    else:
        # Reset si aucun fichier
        st.session_state.pdf_context = ""
        st.session_state.image_data = None

# --- 4. INTERFACE DE CHAT ---
# Affichage de l'historique
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Entrée utilisateur
user_query = st.chat_input("Posez votre question (ex: 'Est-ce que ce médicament sur la photo est compatible avec mes allergies ?')")

if user_query:
    # Affichage du message utilisateur
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Réponse de l'assistant
    with st.chat_message("assistant"):
        with st.spinner("Analyse globale en cours..."):
            
            # 1. Recherche Web
            web_results = search.run(user_query)
            
            # 2. Construction du Prompt textuel CORRIGÉ
            prompt_text = f"""
            Tu es un assistant médical expert doté de capacités de VISION AVANCÉES. 
            
            CONSIGNES IMPORTANTES :
            - Si une image est jointe, tu PEUX et tu DOIS l'analyser. Ne dis jamais que tu ne vois pas l'image.
            - Utilise les pixels de la photo pour identifier les noms de médicaments, les dosages et les formes.
            - Croise ces informations avec le document PDF (allergies, historique) et les recherches WEB.
            - Nous sommes en février 2026.
            
            DONNÉES :
            - CONTENU DU PDF : {st.session_state.pdf_context if st.session_state.pdf_context else "Aucun PDF fourni."}
            - INFOS WEB : {web_results}
            - QUESTION DU PATIENT : {user_query}
            
            Réponse professionnelle et directe :
            """

            # 3. Préparation du message Multimodal
            content_list = [{"type": "text", "text": prompt_text}]
            
            if st.session_state.image_data:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.image_data}"}
                })

            # 4. Appel au modèle Groq
            response = llm.invoke([HumanMessage(content=content_list)])
            
            st.markdown(response.content)
            st.session_state.chat_history.append(AIMessage(content=response.content))

# Bouton de reset
if st.sidebar.button("🗑️ Effacer la discussion"):
    st.session_state.chat_history = []
    st.session_state.pdf_context = ""
    st.session_state.image_data = None
    st.rerun()
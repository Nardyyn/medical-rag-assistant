import streamlit as st
import os
from utils import process_pdf, encode_image, save_to_data_folder
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from fpdf import FPDF

# --- 1. CONFIGURATION ET SÉCURITÉ ---
st.set_page_config(
    page_title="IA Médicale Pro", 
    layout="wide", 
    page_icon="🩺"
)

# Récupération des clés API (Streamlit Secrets en ligne / .streamlit/secrets.toml en local)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]

# Initialisation du modèle Llama 4 Scout (Optimisé Vision & Raisonnement)
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)
search = TavilySearchResults(max_results=3)

# --- 2. GESTION DE LA MÉMOIRE & ÉTAT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_context" not in st.session_state:
    st.session_state.pdf_context = ""
if "image_data" not in st.session_state:
    st.session_state.image_data = None

# Fonction pour générer le rapport PDF
def generate_pdf_report(history):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Compte-rendu de Consultation IA - 2026", ln=True, align="C")
    pdf.ln(10)
    
    for msg in history:
        role = "Patient" if isinstance(msg, HumanMessage) else "Assistant Medical"
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 10, f"{role}:", ln=True)
        pdf.set_font("Arial", "", 11)
        # Nettoyage pour éviter les erreurs d'encodage PDF standard
        clean_text = msg.content.encode('latin-1', 'ignore').decode('latin-1')
        pdf.multi_cell(0, 10, clean_text)
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# --- 3. INTERFACE PRINCIPALE ---
st.title("🩺 Assistant Médical Expert Multimodal")
st.write("Analyse intelligente combinant vos documents, vos photos et la recherche médicale.")

# Guide d'utilisation
with st.expander("ℹ️ Comment utiliser cet assistant ?"):
    st.write("""
    1. **Téléchargez** votre historique médical (PDF) dans la barre latérale.
    2. **Prenez une photo** de la boîte de médicament ou de l'ordonnance.
    3. **Posez vos questions** (ex: "Ce médicament est-il compatible avec mon allergie ?").
    """)
    st.info("💡 Sur mobile, l'importation d'image peut ouvrir directement votre appareil photo.")

# --- 4. BARRE LATÉRALE (FICHIERS & ACTIONS) ---
with st.sidebar:
    st.header("📂 Documents & Analyses")
    uploaded_file = st.file_uploader(
        "Scanner un PDF ou une image", 
        type=['pdf', 'png', 'jpg', 'jpeg']
    )

    if uploaded_file is not None:
        save_to_data_folder(uploaded_file)
        
        if uploaded_file.type == "application/pdf":
            uploaded_file.seek(0)
            with st.spinner("Extraction du PDF..."):
                st.session_state.pdf_context = process_pdf(uploaded_file)
            st.success("✅ Historique PDF mémorisé.")
            
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            uploaded_file.seek(0)
            with st.spinner("Traitement de l'image..."):
                st.session_state.image_data = encode_image(uploaded_file)
            st.image(uploaded_file, caption="Aperçu de l'image", use_container_width=True)
            st.success("✅ Image prête pour analyse.")

    st.divider()
    
    # Boutons d'action
    if st.session_state.chat_history:
        pdf_bytes = generate_pdf_report(st.session_state.chat_history)
        st.download_button(
            label="📄 Télécharger le compte-rendu PDF",
            data=pdf_bytes,
            file_name="consultation_medicale.pdf",
            mime="application/pdf"
        )
        
    if st.button("🗑️ Effacer la discussion"):
        st.session_state.chat_history = []
        st.session_state.pdf_context = ""
        st.session_state.image_data = None
        st.rerun()

    st.warning("⚠️ **Note** : Cet outil est informatif et ne remplace pas un médecin.")

# --- 5. ZONE DE CHAT ---
# Affichage de l'historique
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Entrée utilisateur
user_query = st.chat_input("Votre question médicale...")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion en cours..."):
            
            # 1. Recherche Web en temps réel
            web_results = search.run(user_query)
            
            # 2. Construction du Prompt Système Avancé
            prompt_text = f"""
            Tu es un assistant médical expert doté de capacités de VISION AVANCÉES. 
            
            DIRECTIVES :
            - Si une image est fournie, ANALYSE-LA. Ne dis JAMAIS que tu ne vois pas l'image.
            - Identifie précisément les médicaments, dosages et formes galéniques sur la photo.
            - Croise ces données avec l'historique PDF fourni.
            - Cite tes sources web si nécessaire.
            - Nous sommes en Février 2026.

            CONTEXTE PDF : {st.session_state.pdf_context if st.session_state.pdf_context else "Aucun document PDF n'a été fourni."}
            RÉSULTATS WEB : {web_results}
            QUESTION PATIENT : {user_query}
            """

            # 3. Préparation du message multimodal
            content_list = [{"type": "text", "text": prompt_text}]
            
            if st.session_state.image_data:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{st.session_state.image_data}"}
                })

            # 4. Exécution
            response = llm.invoke([HumanMessage(content=content_list)])
            
            # Affichage de la réponse
            st.markdown(response.content)
            
            # Affichage des sources web cliquables
            if web_results:
                with st.expander("🌐 Sources médicales consultées"):
                    for res in web_results:
                        title = res.get('title', 'Lien source')
                        url = res.get('url', '#')
                        st.markdown(f"- [{title}]({url})")
            
            st.session_state.chat_history.append(AIMessage(content=response.content))
            st.rerun()
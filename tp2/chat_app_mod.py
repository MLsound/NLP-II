import streamlit as st
import sys
import os

# Asegurar que el path raíz esté en sys.path para imports absolutos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config.settings import PDF_FILE_PATH
from src.ingestion import load_pdf_content, process_text_to_docs
from src.vectorstore import setup_vectorstore
from src.rag_engine import get_qa_chain
from utils.streamlit import initialize_session_state, configure_streamlit_page, display_sidebar_menu

# 1. Configuración UI
configure_streamlit_page()
display_sidebar_menu()
initialize_session_state()

model_name = "Gemini 2.5 Flash-Lite"
st.title(f"🤖 Chatbot with {model_name}")
st.subheader("¡Ask a question!")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="feedback_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/5_Chat_with_user_feedback.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

# 2. Lógica de Ingesta (Se ejecuta al cargar)
# Nota: Idealmente esto iría en cache para no re-procesar en cada reload
try:
    with st.spinner("Cargando y procesando documentos..."):
        print("🔄 Iniciando ingesta de documentos...")
        raw_text = load_pdf_content(PDF_FILE_PATH)
        docs = process_text_to_docs(raw_text, PDF_FILE_PATH)
        print(f"Procesados {len(docs)} documentos.")
        # 3. Inicializar Vectorstore e Indexar
        # Pasamos reindex=True para asegurar que los docs se suban (puedes ajustar esta lógica)
        print("⏳ Configurando el vectorstore...")
        vectorstore = setup_vectorstore(documents=docs, reindex=True)
        print("✅ Vectorstore configurado.")
        # 4. Obtener la Cadena (Factory)
        print("⏳ Obteniendo la cadena de preguntas y respuestas...")
        qa_chain = get_qa_chain(vectorstore)
        print("✅ Cadena de preguntas y respuestas lista.")

except Exception as e:
    st.error(f"❌ Error crítico al iniciar la aplicación: {e}")
    st.stop()

# 5. Mostrar historial
print("⏳ Mostrando historial de conversación...")
for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 6. Chat Loop
print("🔄 Iniciando el loop de chat...")
if user_input := st.chat_input("Ask a question..."):
    if not user_input.strip():
        st.warning("Please enter a valid question.")
        st.stop()
    
    st.chat_message("user").markdown(user_input)
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Pensando..."):
        try:
            response = qa_chain.invoke({"input": user_input})
            answer = response["answer"]
            
            st.chat_message("assistant").markdown(answer)
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"❌ Error generando respuesta: {e}")
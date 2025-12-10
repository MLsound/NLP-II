import time
import logging
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from config.settings import LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME

def get_embeddings():
    """Fábrica de Embeddings."""
    if LLM_PROVIDER == 'openai':
        return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    elif LLM_PROVIDER == 'gemini':
        return GoogleGenerativeAIEmbeddings(
            google_api_key=GOOGLE_API_KEY, 
            model="models/text-embedding-004"
        )
    else:
        raise ValueError(f"Proveedor desconocido: {LLM_PROVIDER}")

def setup_vectorstore(documents=None, reindex=False):
    embeddings = get_embeddings()
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Aseguramos que el índice existe (opcional, depende de tu plan de Pinecone)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

    # Lógica de indexado
    if documents and reindex:
        logging.info(f"Iniciando carga de {len(documents)} documentos a Pinecone...")
        
        # Generar IDs únicos basados en el nombre y orden
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        BATCH_SIZE = 50
        DELAY_SECONDS = 3.5

        for i in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]
            
            print(f"-> Subiendo batch {i//BATCH_SIZE + 1}...")
            
            try:
                vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                if (i + BATCH_SIZE) < len(documents):
                    time.sleep(DELAY_SECONDS)
            except Exception as e:
                print(f"Error en batch {i}: {e}")
                # Dependiendo de la severidad, podrías hacer break o continue
                
    return vectorstore
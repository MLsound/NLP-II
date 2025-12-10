import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import ChatPromptTemplate
from config.settings import LLM_PROVIDER, OPENAI_API_KEY, GOOGLE_API_KEY

def get_llm():
    if LLM_PROVIDER == 'openai':
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model='gpt-4o-mini',
            temperature=0.5 # Bajamos un poco temp para RAG
        )
    elif LLM_PROVIDER == 'gemini':
        return ChatGoogleGenerativeAI(
            google_api_key=GOOGLE_API_KEY,
            model="gemini-2.5-flash-lite", # Modelo recomendado para chat rápido
            temperature=0.7
        )
    else:
        raise ValueError(f"Proveedor LLM no soportado: {LLM_PROVIDER}")

def load_system_prompt(filepath="prompt.md"):
    if not os.path.exists(filepath):
        return "Eres un asistente útil. Responde basándote en el contexto."
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def get_qa_chain(vectorstore):
    llm = get_llm()
    system_prompt_text = load_system_prompt()
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template
    )

    retriever = vectorstore.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 3}
    )

    qa_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=document_chain
    )

    return qa_chain
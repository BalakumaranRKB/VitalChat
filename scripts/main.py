import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_ollama import OllamaEmbeddings
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from elevenlabs import ElevenLabs
import base64
import io

app = FastAPI()

chat_history = {}

# Serve static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the HTML file
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.post("/query")
async def process_query(query: Query):
    # Your existing RAG setup
    embed = OllamaEmbeddings(model="mistral")
    weaviate_client = weaviate.connect_to_local()
    
    db = WeaviateVectorStore(
        client=weaviate_client,
        index_name="PodCastStore",
        text_key="text",
        embedding=embed
    )

    template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOllama(
        model="mistral",
        temperature=0,
        num_predict = 256
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    rag_chain = (
        {"context": lambda x: retriever.invoke, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result = rag_chain.invoke(query.text)
    client = ElevenLabs(api_key="sk_c15e4425199e6d92bd33163ea3882b0b4b7c96f043508734", )        
    try:
        # For newer versions of the ElevenLabs API
        audio_output = client.text_to_speech.convert_as_stream(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="mp3_44100_128",
            text=result,
            model_id="eleven_multilingual_v2"
        )
        
        # Handle generator object by collecting chunks
        audio_bytes = b''
        for chunk in audio_output:
            if chunk:
                audio_bytes += chunk
                
        base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        base64_audio = None  # No audio if there's an error
    
    # Store in chat history
    if not hasattr(app, 'chat_history'):
        app.chat_history = {}
    app.chat_history[query.text] = result
    
    
    
    chat_history[query.text] = result
    return {
        "response": result,
        "audio": base64_audio,
        "history": [{"question": q, "answer": a} for q, a in app.chat_history.items()]
    }
#"history": [{"question": q, "answer": a} for q, a in chat_history.items()]

    


# have a dictionary maintain question and answer/past histriy
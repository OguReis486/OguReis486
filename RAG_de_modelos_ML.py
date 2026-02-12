import os
import json
import warnings
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_weaviate import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory


# __ Para usar a OpenAI __
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain_community.vectorstores import Weaviate


warnings.filterwarnings("ignore", category=UserWarning)



# Iniciando minha intenção era usar o Gemini, mas ele tem dado problema de compatibilidade, então optei por 
# usar o modelo da HuggingFace para gerar os embeddings, pra evitar problemas de compatibilidade e ainda sim 
# conseguir testar um RAG usando o Weaviate.

load_dotenv(override=True)
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]


client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key)
)

print("Client is avaliable: ",client.is_ready())

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Gemini dando problema de compatibilidade
#embeddings = GoogleGenerativeAIEmbeddings(
#    model="text-embedding-004"
#)

vectorstore = WeaviateVectorStore(
    client=client,
    index_name="RAG_de_modelos_ML",
    text_key="content",
    embedding=embeddings
)

with open("data.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

texts = []
metadatas = []

for doc in documents:
    texts.append(doc["content"])
    metadatas.append({
        "title": doc["title"],
        "source": "json_modelos_ml"
    })

vectorstore.add_texts(
    texts=texts,
    metadatas=metadatas
)

print(f"{len(texts)} documentos inseridos")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

while True:
    question = input("\nPergunta (ou 'sair'): ")
    if question.lower() == "sair":
        break

    response = qa_chain.invoke({"question": question})
    print("\nResposta:\n", response["answer"])


client.close()

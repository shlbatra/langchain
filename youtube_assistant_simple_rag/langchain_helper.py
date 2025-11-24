from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_classic.agents import initialize_agent, AgentType, load_tools
from langchain_classic.document_loaders import YoutubeLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


load_dotenv

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, # individual doc size
            chunk_overlap = 100 # over lap between docs
    )
    docs = text_splitter.split_documents(transcript) # split docs to 1000 words -> to include part of token sequence length. Only send relevant part based on vector search using FAISS

    db = FAISS.from_documents(docs, embeddings) # smart search engine
    # 1. Each document gets converted to a vector (like [0.1, 0.8, 0.3, ...]) based on OpenAI Embeddings ex, [0.1234, -0.5678, 0.9012, 0.3456, ..., 0.7890]  # 1536 numbers with Similar documents get similar vectors
    # 2. FAISS stores these vectors in an optimized index for ultra-fast search
    # 3. When you search, your query also becomes a vector
    # 4. FAISS finds the most similar document vectors quickly

    return db

def get_response_from_query(db, query, k=4):
    # tet-davinci model can handle 4097 tokens
    
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="text-davinci-003")

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant that that can answer questions about youtube videos based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """
    )

    chain = prompt | llm

    response = chain.invoke({'question': query, 'docs': docs_page_content})

    response = response.replace("\n", "")

    return response
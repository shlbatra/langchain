from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_classic.agents import initialize_agent, AgentType
from langchain_community.document_loaders import YoutubeLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage


load_dotenv()

embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, # individual doc size
            chunk_overlap = 100 # over lap between docs
    )
    docs = text_splitter.split_documents(transcript) # split docs to 1000 words -> to include part of token sequence length. Only send relevant part based on vector search using FAISS

    vector_db = FAISS.from_documents(docs, embeddings) # smart search engine
    # 1. Each document gets converted to a vector (like [0.1, 0.8, 0.3, ...]) based on OpenAI Embeddings ex, [0.1234, -0.5678, 0.9012, 0.3456, ..., 0.7890]  # 1536 numbers with Similar documents get similar vectors
    # 2. FAISS stores these vectors in an optimized index for ultra-fast search
    # 3. When you search, your query also becomes a vector
    # 4. FAISS finds the most similar document vectors quickly

    return vector_db

def get_response_from_query(vector_db, query, k=4):
    
    # Legacy method 
    # docs = vector_db.similarity_search(query, k=k)  
    # docs_page_content = " ".join([d.page_content for d in docs])
    # llm = ChatOpenAI(model="gpt-3.5-turbo")
    # chain = prompt | llm
    # response = chain.invoke({'question': query, 'docs': docs_page_content})
    # response = response.content.replace("\n", "")

    retriever = vector_db.as_retriever(search_kwargs={'k': k})
    
    @tool("youtube_context", return_direct=False)
    def search_youtube_transcript(search_query: str) -> str:
        """Search youtube transcript for information"""
        docs = retriever.invoke(search_query)
        return " ".join([doc.page_content for doc in docs])

    system_prompt = """
    You are a helpful assistant that can answer questions about youtube videos based on the video's transcript.
    You can call youtube_context tool to retrieve context, and then answer succinctly. Maybe you have to use it multiple times before answering.
    Only use the factual information from the transcript to answer the question.
    If you feel like you don't have enough information to answer the question, say "I don't know".
    Your answers should be verbose and detailed.
    """

    agent = create_agent(
        model='gpt-4o-mini',  # Fixed model name
        tools=[search_youtube_transcript],  # Use custom tool
        system_prompt=system_prompt
    )

    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    return result["messages"][-1].content

if __name__=="__main__":
    yt_url="https://www.youtube.com/watch?v=zjkBMFhNj_g"
    query="What is jailbreak and give examples based on video" 

    vector_db = create_vector_db_from_youtube_url(yt_url)
    response = get_response_from_query(vector_db, query)
    print(response)
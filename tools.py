from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain_community.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain_community.tools import YouTubeSearchTool
from langchain_community.vectorstores import Pinecone
# from CRAG import library_rag


dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path)


@tool("youtube_search", return_direct=True)
def searchYoutube(input: str) -> str:
    """Useful when you the user needs youtube video recommendations regarding python or programming or tutorials about something python related.
    DO NOT USE FOR ANY OTHER KIND OF RECOMMENDATIONS"""
    tool = YouTubeSearchTool()
    # Makes only 2 recommendations
    return tool.run(input + ",2")


@tool("google_search", return_direct=True)
def searchGoogle(input: str) -> str:
    """Useful when you need to search the web for information. You need to input the query"""
    search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])
    return search.run(input)


# This tool is only for testing purposes
# @tool("lower_case", return_direct=True)
# def toLowerCase(input: str) -> str:
#     """Returns the input as lower case"""
#     return input.lower()


@tool("langchain_rag", return_direct=True)
def langchain_rag(query: str, chat_history: List[Dict[str, Any]] = []):
    """Useful when you need to answer questions regarding anything or everything about langchain python library.
     You can also use this tool if you are asked about RAG (retrieval augemnted generation) and vector stores. 
     You need to input the query as a parameter, as well as the chat history as an array."""
    embeddings = VertexAIEmbeddings(project="arctic-acolyte-414610")

    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name="langchain-test-index",
    )
    print("docsearch: ", docsearch)

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 40,
        "verbose": "true",
    }

    chat = ChatVertexAI(**parameters)
    print("Kanye")
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    print("qa", qa)
    response = qa({"question": query, "chat_history": chat_history})
    print(response)
    return response["answer"]


@tool("latest_news_rag", return_direct=True)
def news_rag(query: str, chat_history: List[Dict[str, Any]] = []):
    """Useful when you need to answer questions about latest technology, python or generative AI news. You need to input the query
    as a parameter, as well as the chat history as an array."""
    embeddings = VertexAIEmbeddings(project="arctic-acolyte-414610")
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name="python-news",
    )
    print("docsearch: ", docsearch)

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 40,
        "verbose": "true",
    }

    chat = ChatVertexAI(**parameters)
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    print("qa", qa)
    response = qa({"question": query, "chat_history": chat_history})
    print(response)
    return response["answer"]


# tools = [toLowerCase, searchGoogle, library_rag, searchYoutube]
tools = [langchain_rag, news_rag, searchGoogle, searchYoutube]

# print(news_rag("Whats new about OpenAI's Assistants APIs"))

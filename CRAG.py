from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain_community.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain_community.tools import YouTubeSearchTool
from langchain_community.vectorstores import Pinecone
from typing import Dict, TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
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
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain import hub

# from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import MongoDBChatMessageHistory, ConversationSummaryBufferMemory
import os
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.output_parsers.openai_tools import PydanticToolsParser
import pprint
from langgraph.graph import END, StateGraph
from google.cloud import aiplatform
from langchain.chains import LLMChain


# langchain function will take state as parameter
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"
aiplatform.init(project="arctic-acolyte-414610")

uri = os.getenv("MONGODB_CONNECTION_STRING")
message_history = MongoDBChatMessageHistory(
        connection_string=uri, session_id="5", collection_name="Chats"
    )

inputs = {
    "keys": {
        "question": "what are different langchain agents?",
        "chat_history": message_history.messages
    }
}

def crag(inputs):
    """Useful when you need to answer questions regarding anything or everything about langchain python library. You need to input the query
    as a parameter, as well as the chat history as an array."""
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retriever)  # retrieve
    workflow.add_node("grade_documents", grading_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search", googlesearch)  # web search

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    # Compile
    app = workflow.compile()
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

    # Final generation
    pprint.pprint(value["keys"]["generation"])
    return value["keys"]["generation"]


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]


def retriever(state):
    """Retrieve documents from vector store"""
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

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    chat_history = state_dict["chat_history"]
    response = qa({"question": question, "chat_history": chat_history})
    print("hereee")
    print("documents: ",response["answer"])
    return {
        "keys": {
            "documents": response["answer"],
            "question": question,
            "chat_history": chat_history,
        }
    }


def generate(state):
    """Generate an answer based on the retrieved documents."""
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]

    print("question: ", question)
    print("documents: ", documents)
    print("chat_history: ", chat_history)

    # Prompt
    #removed hwar el history ashan el attention.
    prompt_template = """
        Given the following documents {documents} what is the answer to the following question: {question}
        """
    prompt_template = PromptTemplate(
        input_variables=["documents", "chat_history", "question"],
        template=prompt_template,
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt_template | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke(
        {"documents": documents, "chat_history": chat_history, "question": question}
    )
   
    return {
        "keys": {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "generation": generation,
        }
    }


def grading_documents(state):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]

    # LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0, streaming=True
    )

    # Tool
    # grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    # llm_with_tool = model.bind(
    #     tools=[grade_tool_oai],
    #     tool_choice={"type": "function", "function": {"name": "grade"}},
    # )

    # Parser
    # parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        Here is the chat history: {chat_history} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        If it has ZERO relevence then grade it as irrelevant.  \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question", "chat_history"],
    )
    chain = LLMChain(llm=model, prompt=prompt)
    

    # Chain
    # chain = prompt | llm_with_tool | parser_tool
    print("here relevance")
    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    score = chain.invoke({"question": question, "context": documents, "chat_history": chat_history})
    print("score: ", score)    
    if score['text'] == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(documents)
    else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            

    print("here 3")

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
            "chat_history": chat_history,
        }
    }


def transform_query(state):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n 
        Look at the input and try to reason about the underlying sematic intent / meaning. \n 
        Here is the initial question:
        \n ------- \n
        {question} 
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0, streaming=True
    )

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print("better_question: ", better_question)

    return {"keys":{
        "documents": documents,
        "question": better_question,
        "chat_history": chat_history,
    }}


def googlesearch(state):
    """Web search based on the re-phrased question"""
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]
    print("here5")

    search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])
    docs = search.run({"query": question})
    documents = docs
    print("docs: ", docs)

    return {"keys":{"documents": documents, "question": question, "chat_history": chat_history}}


def decide_to_generate(state):
    """Decide whether to generate an answer or go to web search"""
    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


response = crag(inputs)
message_history.add_user_message(inputs["keys"]["question"])
message_history.add_ai_message(response)

from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, TypedDict
from langchain_community.chat_models import ChatVertexAI
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.vertexai import VertexAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import MongoDBChatMessageHistory, ConversationSummaryBufferMemory
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.output_parsers.openai_tools import PydanticToolsParser
import pprint
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from google.cloud import aiplatform
from langchain.chains import LLMChain
from langchain_pinecone import PineconeVectorStore


# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "arctic-acolyte-414610-c6dcb23dd443.json"
aiplatform.init(project="arctic-acolyte-414610")

# uri = os.getenv("MONGODB_CONNECTION_STRING")
# message_history = MongoDBChatMessageHistory(
#     connection_string=uri, session_id="47", collection_name="Chats"
# )

# inputs = {
#     "keys": {
#         "question": "how can i import a function in python?",
#         "chat_history": message_history.messages,
#     }
# }


def crag(session_id,user_input):
    """Useful when you need to answer questions regarding anything or everything about langchain python library. You need to input the query
    as a parameter, as well as the chat history as an array."""
    uri = os.getenv("MONGODB_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
    connection_string=uri, session_id=session_id, collection_name="Chats"
)
    inputs = {
    "keys": {
        "question": user_input,
        "chat_history": message_history.messages,
        "session_id": session_id,
    }
}
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("classify_question", classify_question)  # classify question
    workflow.add_node("retrieve", retriever)  # retrieve
    workflow.add_node("grade_documents", grading_documents)  # grade documents
    workflow.add_node("generate", generate)  # generate
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search", googlesearch)  # web search
    workflow.add_node("handle_general", handle_general)  # handle general questions

    # Build graph
    workflow.set_entry_point("classify_question")
    workflow.add_conditional_edges(
        "classify_question",
        decide_question_type,
        {
            "programming": "retrieve",
            "general": "handle_general",
        },
    )
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
    workflow.add_edge("handle_general", END)

    # Compile
    app = workflow.compile()
    for output in app.stream(inputs):
        for key, value in output.items():
            # Node
            pprint.pprint(f"Node '{key}':")
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


def classify_question(state):
    """Classify the question as programming-related or general."""
    print("---CLASSIFY QUESTION---")
    state_dict = state["keys"]
    question = state_dict["question"]
    print("question: ", question)

    # Create a prompt template for classification
    prompt = PromptTemplate(
        template="""Classify the following question as 'programming' or 'general':
        Question: {question}
        """,
        input_variables=["question"],
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0)

    # Chain
    classification_chain = prompt | llm | StrOutputParser()
    classification = classification_chain.invoke({"question": question}).strip().lower()

    return {
        "keys": {
            "question": question,
            "classification": classification,
            "chat_history": state_dict["chat_history"],
            "session_id": state_dict["session_id"],
        }
    }


def decide_question_type(state):
    """Decide the type of question and route accordingly."""
    state_dict = state["keys"]
    classification = state_dict["classification"]

    if classification == "programming":
        print("---QUESTION TYPE: PROGRAMMING---")
        return "programming"
    else:
        print("---QUESTION TYPE: GENERAL---")
        return "general"


def retriever(state):
    """Retrieve documents from vector store"""
    embeddings = VertexAIEmbeddings(project="arctic-acolyte-414610")

    docsearch = PineconeVectorStore.from_existing_index(
        embedding=embeddings,
        index_name="langchain-test-index",
    )

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
    print("retreiver debugging 1")
    print("documents retrieved: ", response["answer"])
    return {
        "keys": {
            "documents": response["answer"],
            "question": question,
            "chat_history": chat_history,
            "session_id": state_dict["session_id"],
        }
    }


def generate(state):
    """Generate an answer based on the retrieved documents."""
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]
    session_id = state_dict["session_id"]

    print("question: ", question)
    print("documents: ", documents)
    print("chat_history: ", chat_history)

    # Prompt
    prompt_template = """
        Given the following documents {documents} and {chat_history} what is the answer to the following question: {question}
        If it is a greeting, reply to it!
        """
    prompt_template = PromptTemplate(
        input_variables=["documents", "chat_history", "question"],
        template=prompt_template,
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0)

    # Post-processing
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt_template | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke(
        {"documents": documents, "chat_history": chat_history, "question": question}
    )
    uri = os.getenv("MONGODB_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
    connection_string=uri, session_id=session_id, collection_name="Chats"
    )
    message_history.add_user_message(question)
    message_history.add_ai_message(generation)

    return {
        "keys": {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "session_id": session_id,
            "generation": generation,
        }
    }


def grading_documents(state):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    print("documents:", documents)
    chat_history = state_dict["chat_history"]

    # LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0, streaming=True
    )

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document and chat history to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        Here is the chat history: {chat_history} \n
        If the document/chat history contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        If it has ZERO relevence then grade it as irrelevant.  \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question", "chat_history"],
    )
    chain = LLMChain(llm=model, prompt=prompt)

    print("here relevance")
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    score = chain.invoke(
        {"question": question, "context": documents, "chat_history": chat_history}
    )
    print("score: ", score)
    if score["text"] == "yes":
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
            "session_id": state_dict["session_id"],
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
        Formulate an improved question with no punctuation or grammar mistakes: """,
        input_variables=["question"],
    )

    # Grader
    model = ChatGoogleGenerativeAI(
        model="gemini-pro", verbose=True, temperature=0, streaming=True
    )

    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})
    print("better_question: ", better_question)

    return {
        "keys": {
            "documents": documents,
            "question": better_question,
            "session_id": state_dict["session_id"],
            "chat_history": chat_history,
        }
    }


def googlesearch(state):
    """Web search based on the re-phrased question"""
    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    chat_history = state_dict["chat_history"]

    search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])
    docs = search.run({question})
    documents = docs
    print("docs: ", docs)

    return {
        "keys": {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "session_id": state_dict["session_id"],
        }
    }


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


def handle_general(state):
    """Handle general questions like greetings or weather."""
    print("---HANDLE GENERAL---")
    state_dict = state["keys"]
    question = state_dict["question"]
    chat_history = state_dict["chat_history"]
    session_id = state_dict["session_id"]

    # Prompt
    prompt_template = """
        You are a python general assistant and your name is Pyerre.\
        If the user asks about anything malicious, harmful or vile, do not help him, otherwise respond normally, 
        if the context is python or programming related, or the user is just greeting you. Try to be as friendly 
        and helpful to the user as much as possible. You have access to tools that can help you answer questions 
        related to different python frameworks and libraries, 
        use them if necessary.
        Respond to the following question: {question}.
        Check the chat history for more context: {chat_history}.
        """
    prompt = PromptTemplate(
        input_variables=[
            "chat_history",
            "question",
        ],
        template=prompt_template,
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0)

    # Chain
    general_chain = prompt | llm | StrOutputParser()

    # Run
    response = general_chain.invoke(
        {"question": question, "chat_history": chat_history}
    )
    uri = os.getenv("MONGODB_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
    connection_string=uri, session_id=session_id, collection_name="Chats"
    )
    message_history.add_user_message(question)
    message_history.add_ai_message(response)

    return {
        "keys": {
            "question": question,
            "chat_history": chat_history,
            "generation": response,
            "session_id": state_dict["session_id"],
        }
    }


# crag("996","what is lcel in lnagchain python?")


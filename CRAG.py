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


# langchain function will take state as parameter
@tool("library_rag", return_direct=True)
def library_rag(inputs):
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
        A dictionary where each key is a string.
    """

    def __init__(self, state_dict: Dict[str, Any]):
        self.state_dict = state_dict

def retriever(inputs: Dict[str, Any]):
    """Retrieve documents from vector store"""
    embeddings = VertexAIEmbeddings(project='arctic-acolyte-414610')
    
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name="langchain-test-index",
    )
    print("docsearch: ",docsearch)
        
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 40,
        "verbose":'true'
    }
    
    chat = ChatVertexAI(
        **parameters
    )
    
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True, verbose=True
     )
    print("---RETRIEVE---")
    query = inputs['question']
    chat_history = inputs['chat_history']
    response = qa({"question": query, "chat_history": chat_history})
    return {"documents": response['answer'], "question": query, "chat_history": chat_history}

def generate(inputs: Dict[str, Any]):
    """Generate an answer based on the retrieved documents."""
    print("---GENERATE---")
    question = inputs["question"]
    documents = inputs["documents"]
    chat_history = inputs["chat_history"]

    # Prompt
    prompt_template = """
        Given the following documents {documents} and the following chat histroy {chat_history}, what is the answer to the following question: {question}
        """
    prompt_template = PromptTemplate(
        input_variables=["documents", "chat_history", "question"],
        template=prompt_template,
    )

    # LLM
    llm = ChatGoogleGenerativeAI(model='gemini-pro', verbose=True, temperature=0)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt_template | llm | StrOutputParser()

    # Run
    generation = rag_chain.invoke({"documents": documents, "chat_history":chat_history, "question": question})
    return {"documents": documents, "question": question, "generation": generation, "chat_history": chat_history}

def grading_documents(inputs):
    """Determines whether the retrieved documents are relevant to the question."""
    print("---CHECK RELEVANCE---")
    question = inputs["question"]
    documents = inputs["documents"]
    chat_history = inputs["chat_history"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatGoogleGenerativeAI(model='gemini-pro', verbose=True, temperature=0,streaming=True)

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[grade_tool_oai],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        Here is the chat history: {chat_history} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question", "chat_history"],
    )

    # Chain
    chain = prompt | llm_with_tool | parser_tool

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    return {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
            "chat_history": chat_history,
        }
def transform_query(inputs):
    """Transform the query to produce a better question."""
    print("---TRANSFORM QUERY---")
    question = inputs["question"]
    documents = inputs["documents"]
    chat_history = inputs["chat_history"]

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
    model = ChatGoogleGenerativeAI(model='gemini-pro', verbose=True, temperature=0,streaming=True)

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"documents": documents, "question": better_question, "chat_history": chat_history}

def googlesearch(inputs):
    """Web search based on the re-phrased question"""
    print("---WEB SEARCH---")
    question = inputs["question"]
    documents = inputs["documents"]
    chat_history = inputs["chat_history"]

    search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])
    docs = search.run({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question, "chat_history": chat_history}

def decide_to_generate(inputs):
    """Decide whether to generate an answer or go to web search"""
    print("---DECIDE TO GENERATE---")
    question = inputs["question"]
    filtered_documents = inputs["documents"]
    search = inputs["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    pass
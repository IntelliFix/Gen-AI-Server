from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from tools import tools
from langchain import hub

# from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import MongoDBChatMessageHistory, ConversationSummaryBufferMemory
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv()


def chatbot(session_id, user_input):
    """Takes in the session id and the user's message as its parameters. The session id
    is used to keep track of the conversation's history and context"""
    # Old system message
    # """You are a chatbot working as a python assistant. When you are asked
    #                       a question, you should check its scope. If the question is about python programming,
    #                       you should answer it with the best of your knowledge. If the question is anything
    #                       else, you should answer "Sorry, I am a pyhton Assistant only!" """

    gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro", verbose=True, temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"), convert_system_message_to_human=True)
    # open_source_llm = ChatOllama(model="llama2",verbose=True ,temperature=0)
    # gpt_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    # groq_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")

    uri = os.getenv("MONGODB_CONNECTION_STRING")
    message_history = MongoDBChatMessageHistory(
        connection_string=uri, session_id=session_id, collection_name="Chats"
    )

    # prompt = hub.pull("hwchase17/structured-chat-agent")

    # Can be found at: https://smith.langchain.com/hub/abdelmegeed/chat_agent?organizationId=bcea20b8-f288-5cb3-b935-d73c584ef50f
    prompt = hub.pull("abdelmegeed/chat_agent")
    print(f"Prompt: {prompt}")

    chat_agent = create_structured_chat_agent(
        llm=gemini_llm, tools=tools, prompt=prompt
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=chat_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )
    # adapted_input = {"keys": {"question": user_input}}

    print(message_history.messages)
    response = agent_executor.invoke(
        # {"input": adapted_input, "chat_history": message_history.messages}
        {"input": input, "chat_history": message_history.messages}
    )
    message_history.add_user_message(user_input)
    if "text" in response["output"]:
        message_history.add_ai_message(response["output"]["text"])
        return response["output"]["text"]
    else:
        message_history.add_ai_message(response["output"])
        return response["output"]
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import tools
from langchain import hub
# from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import MongoDBChatMessageHistory, ConversationSummaryBufferMemory
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path)
def chatbot(session_id,user_input):
    """Takes in the session id and the user's message as its parameters. The session id
                is used to keep track of the conversation's history and context"""
    # Old system message
    # """You are a chatbot working as a python assistant. When you are asked 
    #                       a question, you should check its scope. If the question is about python programming,
    #                       you should answer it with the best of your knowledge. If the question is anything
    #                       else, you should answer "Sorry, I am a pyhton Assistant only!" """
    
    gemini_llm = ChatGoogleGenerativeAI(model='gemini-pro', verbose=True, temperature=0, convert_system_message_to_human=True)
    # gpt_llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    uri = os.getenv('MONGODB_CONNECTION_STRING')
    message_history = MongoDBChatMessageHistory(
        connection_string=uri, session_id=session_id, collection_name= "Chats"
    )
    
    
    prompt = hub.pull("abdelmegeed/chat_agent")
    print(f"Prompt: {prompt}")

    
    chat_agent = create_structured_chat_agent(llm=gemini_llm, tools=tools, prompt=prompt)
        
    agent_executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent, 
            tools=tools, 
            verbose=True, 
            handle_parsing_errors=True,
            return_intermediate_steps=True,
    )
    print(message_history.messages)
    response = agent_executor.invoke(
                {
                    "input": f"{user_input}",
                    "chat_history": message_history.messages
                }
            )
    message_history.add_user_message(user_input)
    if 'text' in response["output"]:
        message_history.add_ai_message(response['output']['text'])
        return response['output']['text']
    else:
        message_history.add_ai_message(response['output'])
        return response['output']
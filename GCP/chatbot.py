import os
import dotenv
from langchain.chat_models import ChatVertexAI
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from pymongo.mongo_client import MongoClient
from langchain.memory import MongoDBChatMessageHistory

def adjusting_prompt(user_message):
    prompt = f""" You are a chatbot working as a python assistant. When you are asked 
                          a question, you should check its scope. If the question is about python programming,
                          you should answer it with the best of your knowledge. If the question is anything
                          else, you should answer "Sorry, I am a pyhton Assistant only!"
    question: {user_message}
    """
    return prompt

dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../gen-lang-client-0381077545-08b61ee2a65e.json"

uri = "mongodb+srv://karim:karim@cluster0.mqdt2q9.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

chat = ChatVertexAI(model_name="chat-bison",temperature=0)

message_history = MongoDBChatMessageHistory(
    connection_string=uri, session_id="test-session", collection_name= "Fathy"
)
# put session and collection name accordingly

messages = [SystemMessage(content="""You are a chatbot working as a python assistant. When you are asked 
                          a question, you should check its scope. If the question is about python programming,
                          you should answer it with the best of your knowledge. If the question is anything
                          else, you should answer "Sorry, I am a pyhton Assistant only!" """),]
print("AI: How can I help you?")

while True:
    human_message = input("You: ")
    prompt = adjusting_prompt(human_message)
    message_history.add_user_message(human_message)
    messages.append(HumanMessage(content=prompt))
    response = chat(messages)
    message_history.add_ai_message(response.content)
    print("AI: ", response.content)
    messages.append(response)



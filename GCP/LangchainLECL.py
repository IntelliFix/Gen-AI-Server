from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGooglePalm
from langchain.chat_models import ChatVertexAI
from langchain.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../future-oasis-396818-f8f0f89a62f0.json"

    
class OutputResponse(BaseModel):
    "Outputs the comment and corrected code as a JSON object"
    comment: str = Field(description="Comment about the incorrect code and what was wrong")
    corrected_code: str = Field(description="The corrected python code")
    
    def to_dict(self):
        return {"comment":self.comment,"corrected_code":self.corrected_code}
    
    
def LECLchat(message):
    parser = PydanticOutputParser(pydantic_object=OutputResponse)
    # make sure to check VrtexAI other models (not chat models) including code-bison and code-gecko
    model = ChatVertexAI(temperature=0,top_k=40,model_name='chat-bison')
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a python code fixer, and I am going to send you an agent executor chain, which contains \
        corrected python code, and comments about what was wrong in the code. I need you to extract this information from the message and \
        reply only with a JSON object which contains the comment, and the corrected code. The JSON object should have 2 keys only: comment \
        and corrected_code. REPLY WITH NOTHING BUT THE JSON OBJECT."),
    ("user", "{input}") ])
    
    tagging_chain = prompt | model | parser
    
    # Sometimes the parser fails to parse the response properly
    response = tagging_chain.invoke({"input": f"{message}"})
    
    return response.to_dict()
from pydantic import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from dotenv import load_dotenv
import os
import re

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../arctic-acolyte-414610-c6dcb23dd443.json"

# Output Parser    
class OutputResponse(BaseModel):
    "Outputs the comment and corrected code as a JSON object"
    comment: str = Field(
        description="Comment about the incorrect code and what was wrong"
    )
    corrected_code: str = Field(description="The corrected python code")

    def to_dict(self):
        return {"comment":self.comment,"corrected_code":self.corrected_code}
    
    
# Extract usefule info from the agent executor chain
def LECLchat(message):
    parser = PydanticOutputParser(pydantic_object=OutputResponse)
    # model = ChatVertexAI(temperature=0,top_k=40,model_name='chat-bison')
    # prompt = ChatPromptTemplate.from_messages([
    # ("system", "You are a python code fixer, and I am going to send you an agent executor chain, which contains \
    #     corrected python code, and comments about what was wrong in the code. I need you to extract this information from the message and \
    #     reply only with a JSON object which contains the comment, and the corrected code. The JSON object should have 2 keys only: comment \
    #     and corrected_code. REPLY WITH NOTHING BUT THE JSON OBJECT."),
    # ("user", "{input}") ])
    model = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    # model = ChatGoogleGenerativeAI(temperature=0, top_k=40, model='gemini-pro', convert_system_message_to_human=True)
    # The instruction prompt is identical to the commented prompt above
    prompt = hub.pull("abdelmegeed/code-fixer")
    
    try:
        tagging_chain = prompt | model | parser
        response = tagging_chain.invoke({"input": f"{message}"})
        return response.to_dict()
    # We enter the except block if the langchan output parser fails, hence we use regular expressions to avoid errors
    except Exception as e:
        print("Exception:", e)
        response = prompt | model
        response = tagging_chain.invoke({"input": f"{message}"})
        comment_pattern = r'"comment"\s*:\s*"([^"]*)"'
        corrected_code_pattern = r'"comment"\s*:\s*"([^"]*)"'
        comment_matches = re.search(comment_pattern, str(response))
        code_matches = re.search(corrected_code_pattern, str(response))
        
        return {"comment":comment_matches[0].split(':')[1].strip().strip('"'),
                "corrected_code":code_matches[0].split(':')[1].strip().strip('"')}
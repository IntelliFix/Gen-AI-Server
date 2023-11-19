import os
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import PythonREPLTool
from langchain.tools import PythonAstREPLTool
from langchain.utilities import PythonREPL
from langchain.llms import vertexai
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatVertexAI
from GCP.JSONParser import standardOutputParser
from GCP.PaLM import PaLMChat


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../future-oasis-396818-f8f0f89a62f0.json"
def pythonAgent(input_code,comment):
    
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 40
    }
    

    agent_executor = create_python_agent(
        llm=ChatVertexAI(**parameters),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={"handle_parsing_errors": True, "max_iterations":4,"return_intermediate_steps":True,"return_output":False},
    )
    
    augmented_code = input_code + " # " + comment
    x = agent_executor(f"""Ignore missing dependencies. What is wrong with the following code: ```{augmented_code}```?\
        Answer with the corrected code and your comments on what was wrong""")
    response = PaLMChat(x)
    # PALMs response object has many attributes, make sure to check them.
    # print(x)
    print("The whole response", response)
    return response.candidates[0]
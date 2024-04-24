import os
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
# from langchain.tools import PythonAstREPLTool
from langchain_community.utilities import PythonREPL
from langchain_community.llms import vertexai
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatVertexAI
from GCP.JSONParser import standardOutputParser
from GCP.PaLM import PaLMChat
from GCP.LangchainLECL import LECLchat

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    "../arctic-acolyte-414610-c6dcb23dd443.json"
)


def pythonAgent(input_code, comment):

    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 0,
        "top_p": 0.8,
        "top_k": 40,
    }

    agent_executor = create_python_agent(
        # llm=ChatVertexAI(**parameters),
        # Using Llama3 80B using Groq's inference API
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192"),
        # Using Llama 7B responded in 23 mins for DFS question (running locally)
        # llm= ChatOllama(model='llama2', temperature=0, top_k=40, top_p=0.8),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent_executor_kwargs={
            "handle_parsing_errors": True,
            "max_iterations": 4,
            "return_intermediate_steps": True,
            "return_output": False,
        },
    )

    augmented_code = input_code + " # " + comment
    agent_executor_chain = agent_executor(f"""If a function name is provided, try to deduce what the function does from it. Ignore missing dependencies. \
        What is wrong with the following code: ```{augmented_code}```?\
        Answer with the corrected code and your comments on what was wrong""")
    print("Response from the agent executor:")
    # print(agent_executor_chain)
    response = LECLchat(agent_executor_chain)
    return response

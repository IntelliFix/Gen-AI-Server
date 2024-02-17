from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from dotenv import load_dotenv



dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path)
@tool("google_search", return_direct=True)
def searchGoogle(input:str) -> str:
    """Useful when you need to search the web for a specific python library"""
    search = GoogleSerperAPIWrapper(serper_api_key=os.environ["SERPAPI_API_KEY"])
    return search.run(input)

@tool("lower_case", return_direct=True)
def toLowerCase(input:str) -> str:
    """Returns the input as lower case"""
    return input.lower()

tools = [toLowerCase, searchGoogle]
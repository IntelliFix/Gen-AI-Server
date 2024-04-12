from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
import re
import json


def langchainOutputParser(input):
    class Output(BaseModel):
        code: str = Field(description="python code")
        comments: List[str] = Field(
            description="list of comments about the python code"
        )

    parser = PydanticOutputParser(pydantic_object=Output)

    return parser.parse(input)


def standardOutputParser(input):
    code_pattern = r"```python(.*?)```"
    # comment_pattern = r"(```(.*?)```).*" # Not working
    code_matches = re.findall(code_pattern, input, re.DOTALL)
    comment_matches = input.replace("".join(code_matches), "").strip()
    if comment_matches[0] == "`":
        comment_matches = re.sub(r"^```(python)?```\n", "", comment_matches)
    # comment_matches = re.findall(comment_pattern, input, re.DOTALL)
    result = {"code": "".join(code_matches), "comment": comment_matches}
    # parsed_data = json.loads(json.dumps(result))
    # code = parsed_data['code']
    # print({"code":code})
    return result


# standardOutputParser("""
#                      ```python
# def bubble_sort(arr):
#     n = len(arr)
#     for i in range(n):
#         for j in range(0, n - i - 1):
#             if arr[j] > arr[j + 1]:
#                 arr[j], arr[j + 1] = arr[j + 1], arr[j]
#     return arr
# ```
# The error is that the line `arr[j], arr[j] = arr[j + 1], arr[j]` is incorrect. It should be `arr[j], arr[j + 1] = arr[j + 1], arr[j]`.
#                      """)

import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import ChatModel
import os
import dotenv


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="../future-oasis-396818-f8f0f89a62f0.json"

parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison")
model2 = ChatModel.from_pretrained('chat-bison@001')
# response = model.predict(
#     """Hello""",
#     **parameters
# )

chat = model2.start_chat(context="You are a python code fixer. I am going to input python code and I need you\
    to correct this code if it has any mistakes, and refactor anything that needs to be refactored especially variable\
    and function names. Reply only with a JSON object that has 2 keys, corrected_code and comment. The comment field has\
    the comments about the corrections or changes you made. The comment should be elaborative and contain the code changed, before and\
    after the change. If you have not made any changes and the code is already fine respond with 'Your code seems to look good!\
    I haven't made any changes.'. THE COMMENT SHOULD BE ONE THING ONLY!")





response = chat.send_message(
    """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j] = arr[j + 1], arr[j]
    return arr
    """, **parameters
    )


print(f"Response from Model: {response.text}")
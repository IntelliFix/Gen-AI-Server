from flask import request
from flask import jsonify
from yaml_loader import blueprint_yaml
from GCP.JSONParser import standardOutputParser
from injection import prompt_injection
from GCP.REPLAgent import pythonAgent
from GCP.CodeClassifier import classify_code
from langchain.load.dump import dumps
from chatbot import chatbot
from CRAG import crag
import json
import os
import asyncio

main_bp, blueprints = blueprint_yaml("blueprints", "main_bp")


@blueprints.route(main_bp["test-route"], methods=["POST"])
def test_API():
    try:
        request_data = request.get_json()

        question = request_data.get("question")

        return {"Echo": question}

    except Exception as e:
        return [], 500


@blueprints.route(main_bp["code-fixer-route"], methods=["POST"])
async def code_fixer():
    try:
        request_data = request.get_json()

        code = request_data.get("code")
        comment = request_data.get("comment")
        code_classification = classify_code(code)
        if code_classification != 'python':
            return {"corrected_code":"","comment":"I'm sorry :( , but I can only help you with python code." }
        llm_response = pythonAgent(code, comment)

        return llm_response

    except Exception as e:
        return {"error": str(e)}, 500


@blueprints.route(main_bp["chatbot-route"], methods=["POST"])
def chat():
    try:
        request_data = request.get_json()

        session_id = request_data.get("session_id")
        message = request_data.get("message")
        # print("Chatbot")
        response = crag(session_id=session_id, user_input=message)
        # response = chatbot(session_id=session_id, user_input=message)
        return {"output": response}

    except Exception as e:
        print(e)
        return {"exception": str(e)}, 500


@blueprints.route(main_bp["prompt-injection-route"], methods=["POST"])
def injection():
    try:
        request_data = request.get_json()

        message = request_data.get("message")

        response = prompt_injection(message)
        return response

    except Exception as e:
        print(e)
        return {"exception": str(e)}, 500

# @blueprints.route(main_bp["crag-route"], methods=["POST"])
# def chat_crag():
#     try:
#         request_data = request.get_json()

#         session_id = request_data.get("session_id")
#         message = request_data.get("message")
#         print("Chatbot")
#         response = crag(session_id=session_id, user_input=message)
#         return {"output": response}

#     except Exception as e:
#         print(e)
#         return {"exception": str(e)}, 500
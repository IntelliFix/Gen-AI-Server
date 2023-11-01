from flask import request
from flask import jsonify
from yaml_loader import blueprint_yaml
from GCP.JSONParser import standardOutputParser
from GCP.REPLAgent import pythonAgent
import os
import asyncio

main_bp, blueprints = blueprint_yaml("blueprints","main_bp")

@blueprints.route(main_bp["test-route"], methods=["POST"])
def test_API():
    try:
        request_data = request.get_json()

        question = request_data.get("question")

        return {"Echo":question}

    except Exception as e:
        return [], 500
    
@blueprints.route(main_bp["code-fixer-route"], methods=["POST"])
async def code_fixer():
    try:
        request_data = request.get_json()

        code = request_data.get("code")
        
        llm_response = await asyncio.to_thread(pythonAgent, code)
        print(llm_response)

        json_response = standardOutputParser(llm_response)
        print(json_response)
        
        return jsonify({"code":json_response["code"], "comment":json_response["comment"]})

    except Exception as e:
        return {"error": str(e)},500

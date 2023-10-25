from flask import request
from yaml_loader import blueprint_yaml

main_bp, blueprints = blueprint_yaml("blueprints","main_bp")

@blueprints.route(main_bp["test-route"], methods=["POST"])
def test_API():
    try:
        request_data = request.get_json()

        text = request_data.get("question")

        return {"Echo":text}

    except Exception as e:
        return [], 500

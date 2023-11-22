import sys
import os
from blueprints.main_blueprints import blueprints
from dotenv import load_dotenv
from flask import Flask
from google.cloud import aiplatform

load_dotenv()
app = Flask(__name__)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="future-oasis-396818-f8f0f89a62f0.json"
aiplatform.init(project='future-oasis-396818')
# sys.path.append("../..")


app.register_blueprint(blueprints)

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=4000)
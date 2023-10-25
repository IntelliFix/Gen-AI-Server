import sys

from blueprints.main_blueprints import blueprints
from dotenv import load_dotenv
from flask import Flask

load_dotenv()
app = Flask(__name__)

sys.path.append("../..")

app.register_blueprint(blueprints)

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=4000)
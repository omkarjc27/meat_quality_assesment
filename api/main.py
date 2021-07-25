from flask import Flask,request,render_template
from flask_restful import Resource,Api
from flask_cors import CORS
from predict import *

app = Flask(__name__)
CORS(app)
api = Api(app)

api.add_resource(Predict,'/predict/')

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
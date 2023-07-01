from flask import Flask, request
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self):
        return {"data": "Hello World"}
    def post(self):
        return {"data": "Hello Post"}
    
class HelloWorldName(Resource):
    def get(self, name):
        return {"data": name}

class HelloWorldPost(Resource):
    def post(self):
        print(request.form["additional"])
        return request.form["as"]

put_args = reqparse.RequestParser()
put_args.add_argument("name", type=str, help="Name is needed", required=True)
put_args.add_argument("num", type=str, help="Num is needed")

class HelloWorldPut(Resource):
    def put(self):
        args = put_args.pare_args()
        return {"args": args}
    def delete(self):
        return 1

api.add_resource(HelloWorld, "/helloworld")
api.add_resource(HelloWorldName, "/helloworld/<string:name>")
api.add_resource(HelloWorldPost, "/helloworld_post")

if __name__ == "__main__":
    app.run(debug=True, port=5000)
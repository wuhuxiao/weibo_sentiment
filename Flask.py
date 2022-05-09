import sys
from flask import Flask
from flask_cors import CORS

from cnn_model_test import queryComment
from cnn_train import trainModel

app = Flask(__name__, static_url_path='')
from flask import request

CORS(app, supports_credentials=True)


# socket
# from flask_socketio import SocketIO, emit

class Logger(object):
    def __init__(self, logFile="Default.log"):
        self.terminal = sys.stdout
        # self.emit = emit
        self.log = open(logFile, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # self.emit('terminal',message)
    def flush(self):
        pass
#
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
# socketio = SocketIO(app)
# @socketio.on('my event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']})
#
# @socketio.on('my broadcast event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']}, broadcast=True)
#
# @socketio.on('connect', namespace='/test')
# def test_connect():
#     emit('my response', {'data': 'Connected'})
#
# @socketio.on('disconnect', namespace='/test')
# def test_disconnect():
#     print('Client disconnected')

@app.route("/")
def hello():
    return "Hello World!"


# trainCNNModel?MaxLength=128&WordVectorType=word2vec&EmbeddingSize=300&BatchSize=32&Epochs=5
@app.route("/trainCNNModel")
def trainCNNModel():
    MaxLength = request.args.get('MaxLength')
    WordVectorType = request.args.get('WordVectorType')
    EmbeddingSize = request.args.get('EmbeddingSize')
    BatchSize = request.args.get('BatchSize')
    Epochs = request.args.get('Epochs')
    try:
        trainModel(MaxLength, WordVectorType, EmbeddingSize, BatchSize, Epochs)
        return '训练完成'
    except Exception as e:
        return str(e)


@app.route("/getCommentSenti")
def getCommentSenti():
    comment = request.args.get('message')
    lable = queryComment(comment)
    if lable == 1:
        return '积极'
    else:
        return '消极'


if __name__ == "__main__":
    # sys.stdout = Logger("./log/web_train.txt")
    app.run('0.0.0.0', 8080, True, )
    # socketio.run(app)

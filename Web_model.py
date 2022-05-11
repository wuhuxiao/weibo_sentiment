# 停止线程
import ctypes
import inspect
import sys
import threading

from flask import Flask
from flask import request
from flask_cors import CORS
# socket
from flask_socketio import SocketIO, emit

from cnn_model_test import queryComment
from cnn_train import trainModel


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    try:
        tid = ctypes.c_long(tid)
        if not inspect.isclass(exctype):
            exctype = type(exctype)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
        if res == 0:
            # pass
            raise ValueError("invalid thread id")
        elif res != 1:
            # """if it returns a number greater than one, you're in trouble,
            # and you should call it again with exc=NULL to revert the effect"""
            ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
    except Exception as err:
        print(err)


def stop_thread(thread):
    """终止线程"""
    _async_raise(thread.ident, SystemExit)


#
# class trainJob(threading.Thread):
#   def __init__(self, *args, **kwargs):
#     super(threading.Thread, self).__init__(self)
#     # 用于暂停线程的标识
#     self.__flag = threading.Event()
#     self.__flag.set()    # 设置为True
#     # 用于停止线程的标识
#     self.__running = threading.Event()
#     self.__running.set()   # 将running设置为True
#
#   def run(self):
#     while self.__running.isSet():
#       self.__flag.wait()   # 为True时立即返回, 为False时阻塞直到内部的标识位为True后返回
#       time.sleep(1)
#
#   def pause(self):
#     self.__flag.clear()   # 设置为False, 让线程阻塞
#
#   def resume(self):
#     self.__flag.set()  # 设置为True, 让线程停止阻塞
#
#   def stop(self):
#     self.__flag.set()    # 将线程从暂停状态恢复, 如果已经暂停的话
#     self.__running.clear()    # 设置为False

class Logger(object):
    def __init__(self, logFile="Default.log", emit=None):
        self.terminal = sys.stdout
        self.emit = emit
        self.log = open(logFile, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.emit('train cnn', {'data': message}, broadcast=True)

    def flush(self):
        pass


# web服务
app = Flask(__name__)
# 允许跨域
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*', threaded=True)
CORS(app, supports_credentials=True)
thread1 = None
cnn_trainning = False


# @socketio.on('my event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']})
#
# @socketio.on('my broadcast event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']}, broadcast=True)

@socketio.on('connect', namespace='/test')
def test_connect():
    emit('connect response', {'data': '后台连接成功'})


@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


@app.route("/")
def hello():
    global cnn_trainning
    cnn_trainning = False
    stop_thread(thread1)
    return "Hello World!"


def emit_wrap(f):
    def decorator(event, *args, **kwargs):
        return f(event, *args, **kwargs)

    return decorator


@socketio.on('train cnn', namespace='/test')
def socket_trainCNNModel(message):
    print(message['data'])
    data = message['data']
    global cnn_trainning
    global thread1
    MaxLength = eval(data[0])
    WordVectorType = data[1]
    EmbeddingSize = eval(data[2])
    BatchSize = eval(data[3])
    Epochs = eval(data[4])

    if cnn_trainning:
        emit('train cnn', {'data': '模型正在训练'})
    else:
        # return '训练完成'
        cnn_trainning = True
        emit('train cnn',
             {'data': '开始训练模型参数为MaxLength %s WordVectorType %s EmbeddingSize %s BatchSize %s Epochs %s' % (
                 MaxLength, WordVectorType, EmbeddingSize, BatchSize, Epochs)})
        # thread1 = socketio.start_background_task(trainModel, MaxLength, WordVectorType, EmbeddingSize, BatchSize,
        #                                          Epochs, socketio)
        thread1 = threading.Thread(target=trainModel,
                                   args=(MaxLength, WordVectorType, EmbeddingSize, BatchSize, Epochs, socketio))
        thread1.start()
        return '训练完成'


# trainCNNModel?MaxLength=128&WordVectorType=word2vec&EmbeddingSize=300&BatchSize=32&Epochs=5
# @app.route("/trainCNNModel")
# def trainCNNModel():
#     MaxLength = request.args.get('MaxLength')
#     WordVectorType = request.args.get('WordVectorType')
#     EmbeddingSize = request.args.get('EmbeddingSize')
#     BatchSize = request.args.get('BatchSize')
#     Epochs = request.args.get('Epochs')
#     try:
#         # return '训练完成'
#         emit('terminal', {'data': '开始训练模型参数为MaxLength %s WordVectorType %s EmbeddingSize %s BatchSize %s Epochs %s' % (
#         MaxLength, WordVectorType, EmbeddingSize, BatchSize, Epochs)})
#         # trainModel(MaxLength, WordVectorType, EmbeddingSize, BatchSize, Epochs)
#         return '训练完成'
#     except Exception as e:
#         return str(e)


@app.route("/getCommentSenti")
def getCommentSenti():
    comment = request.args.get('message')
    logits = queryComment(comment)
    label = logits.argmax()
    if label == 1:
        return '积极，置信度：%.2f' % (logits[0, 1])
    else:
        return '消极，置信度：%.2f' % (logits[0, 0])


if __name__ == "__main__":
    # app.run('0.0.0.0', 8080, True, )
    socketio.run(app, host='0.0.0.0', port=8080)

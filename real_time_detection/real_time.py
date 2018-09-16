import argparse

import sys
from pythonosc import dispatcher
from pythonosc import osc_server

import random
import time
import schedule
import threading
import timeit
from pythonosc import osc_message_builder
from pythonosc import udp_client
import cv2

import logging

from pythonosc.parsing import osc_types

import numpy as np
import tensorflow as tf
from sklearn import preprocessing

# path to save data file
path = '/Users/canxiuzhang/Documents/ABLE/experiment/dataCollection'


# Handle
message_left = [] # message left
def handle_left(address, accelX, accelY,accelZ, magX, magY,magZ,gyroX, gyroY, gyroZ,heel,toe):
    arg = [accelX, accelY, accelZ, magX, magY, magZ, gyroX, gyroY, gyroZ, heel, toe]
    print('receiving ' + str(arg) + 'from address' + str(address))
    if len(message_left) < 20:
        message_left.append(arg)
    else:
        message_left.pop(0)
        message_left.append(arg)
    #print('message_left len', len(message_left))

message_right = [] # message right
def handle_right(address, accelX, accelY,accelZ, magX, magY,magZ,gyroX, gyroY, gyroZ,heel,toe):
    arg = [accelX, accelY, accelZ, magX, magY, magZ, gyroX, gyroY, gyroZ, heel, toe]
    print('receiving ' + str(arg) + 'from address' + str(address))
    if len(message_right) < 20:
        message_right.append(arg)
    else:
        message_right.pop(0)
        message_right.append(arg)
    #print('message_right len', len(message_right))

# Client (supercollider)
parser_cli = argparse.ArgumentParser()
parser_cli.add_argument("--ip", default="192.168.0.154", help="The ip to send message")
parser_cli.add_argument("--port", type=int, default=10000, help="The port to send message")# unreal
args_cli = parser_cli.parse_args()
client = udp_client.SimpleUDPClient(args_cli.ip, args_cli.port)

# Server
parser_server = argparse.ArgumentParser()

# IP and port for listen
parser_server.add_argument("--ip", default="192.168.0.162", help="The ip to listen on")
parser_server.add_argument("--port", type=int, default=9001, help="The port to listen on")
args = parser_server.parse_args()
dispatcher = dispatcher.Dispatcher()

def action():
    if len(message_left) < 20 or len(message_right) < 20:
        threading.Timer(0.1, action).start()
        #print('action')
        pass
    else:
        # detection
        label = test_model(message_left, message_right)
        client.send_message("/label", label)
        print('label:', label)
        threading.Timer(1, action).start()  # test again after 1 second

def test_model(message_left, message_right):
    labels = ['Squat', 'Lunge', 'Walk', 'Stand', 'Default']
    message_left = np.stack(message_left, axis=0)  # (20,11)
    message_right = np.stack(message_right, axis=0)  # (20,11)
    X_test = np.concatenate((message_left, message_right), axis=1)  # (20, 22)
    X_test = X_test.astype(float)
    X_test = preprocessing.normalize(X_test, norm='l2')
    prob = sess.run(pred, feed_dict={x: X_test.reshape(-1, 20, 22)})
    prob = prob.reshape(-1,1)
    index = np.argmax(prob)
    print('squat: ' + str(prob[0]) + 'lunge:' + str(prob[1]) + 'walk:' + str(prob[2]) + 'stand:' + str(prob[3]))
    if prob[index] <= 0.5:
        label = labels[4]
    else:
        label = labels[index]
    return label

def sway():
    scale = 1.0
    if message_left == [] or message_right == []:
        threading.Timer(0.1, sway).start()
        pass
    else:
        # calculate COP
        le = message_left[-1]
        r = message_right[-1]
        cop = COP(le[9], le[10], r[9], r[10], scale)
        client.send_message("/cop/x", cop[0])
        print('send cop_x')
        client.send_message("/cop/y", cop[1])
        print('send cop_y')
        threading.Timer(0.1, sway).start()  # test again after 1 second

def COP(lh, lt, rh, rt, L):
    """
    :param lh: 1
    :param lt:
    :param rh:
    :param rt:
    :return: (x,y)
    """
    lh = float(lh)
    lt = float(lt)
    rh = float(rh)
    rt = float(rt)
    L = float(L)
    w = lh + lt + rh + rt # scale by 50
    cop = np.array([0.0,0.0])
    cop[0] = ((rt + rh) - (lt + lh)) * L / w
    cop[1] = ((lt + rt) - (lh + rh)) * L / w
    print("lh, lt, rh, rt", lh, lt, rh, rt)
    print("cop-x", cop[0])
    print("cop-y", cop[1])
    return cop

sess = tf.Session()
saver = tf.train.import_meta_graph('LSTM_model_clean.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

pred = graph.get_tensor_by_name('pred:0')
x = graph.get_tensor_by_name('x:0')

dispatcher.map("/left/all", handle_left)
dispatcher.map("/right/all", handle_right)

sway()
action()

server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()












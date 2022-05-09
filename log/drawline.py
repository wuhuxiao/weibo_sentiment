import matplotlib.pyplot as plt
import numpy as np
import pyecharts.options as opts
from pyecharts.charts import Line
with open('./log/cnn_train.txt', encoding='utf-8') as f:
    lines = f.readlines()
start = 10
epoch1_str = lines[10:start + 3000]
start = start + 3003
epoch2_str = lines[start:start + 3000]
start = start + 3003
epoch3_str = lines[start:start + 3000]
start = start + 3003
epoch4_str = lines[start:start + 3000]
start = start + 3003
epoch5_str = lines[start:start + 3000]
total_str = []
total_str.extend(epoch1_str)
total_str.extend(epoch2_str)
total_str.extend(epoch3_str)
total_str.extend(epoch4_str)
total_str.extend(epoch5_str)
loss_list = []
accuracy_list = []
for i,item in enumerate(total_str):
    loss = eval(item.split('\x08')[0].split('-')[2].split('loss:')[1])
    accuracy = eval(item.split('\x08')[0].split('-')[3].split('accuracy:')[1])
    loss_list.append(loss)
    accuracy_list.append(accuracy)
# 绘制训练 & 验证的准确率值
data_loss = loss_list[0:3000]

plt.plot(loss_list[0:3000])
plt.plot(accuracy_list[0:3000])
plt.title('First Epoch')
plt.xlabel('Steps')
plt.legend([ 'Train_loss','Train_acc'])
plt.show()

with open('./log/cnn_train_history.txt', encoding='utf-8') as f:
    lines = f.readlines()
history = eval(lines[0])

plt.plot(['1','2','3','4','5'],history['loss'])
plt.plot(['1','2','3','4','5'],history['val_loss'])
# plt.title('')
plt.xlabel('Epochs')
plt.legend([ 'Train_loss','Val_loss'])
plt.show()

plt.plot(['1','2','3','4','5'],history['accuracy'])
plt.plot(['1','2','3','4','5'],history['val_accuracy'])
# plt.title('')
plt.xlabel('Epochs')
plt.legend([ 'Train_acc','Val_acc'])
plt.show()
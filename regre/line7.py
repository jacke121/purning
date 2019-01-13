
import torch
from torch.autograd import Variable
import numpy as np


N = 5000
K = 7
filename = 'lograndtmp.txt'
x_data = []
y_data = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline()  # 整行读取数据
        if not lines:
            break
        E_tmp = [float(i) for i in lines.split(' ')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        x_data.append(E_tmp[:7])
        y_data.append([E_tmp[-1]])

x_data=np.asarray(x_data)
y_data=np.asarray(y_data)
#print(x_data.data.shape)
print(y_data.shape)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(7, 5)
        self.l2 = torch.nn.Linear(5, 3)
        self.l3 = torch.nn.Linear(3, 2)
        self.l4 = torch.nn.Linear(2, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        y_pred = self.l3(x)
        y_pred = self.l4(y_pred)
        return y_pred

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model = Model()

cirterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

hour_var = Variable(torch.Tensor([x_data[0]]))
print("(Before training)", model.forward(hour_var).data[0][0])

checkpoint_dir='model'
base_loss=1
# Training loop
import random

ran= [i for i in range(5000)]

for epoch in range(100000):

    random.shuffle(ran)


    for j in range(500):
        x=x_data[ran[j*10:(j+1)*10]]
        y=y_data[ran[j*10:(j+1)*10]]
        X_torch = torch.from_numpy(x).float()
        y_torch = torch.from_numpy(y).float()

        y_pred = model(X_torch)
        # y_pred,y_data不能写反(因为损失函数为交叉熵loss)
        loss = cirterion(y_pred, y_torch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 and j==499:
            print(epoch,  optimizer.param_groups[0]['lr'],float(loss))
        if float(loss)<base_loss:
            base_loss=float(loss)
            torch.save(model.state_dict(), '%s/%.4f_%04d.weights' % (checkpoint_dir,float(loss), epoch))

# After training
# hour_var = Variable(torch.Tensor([[-0.294118,0.487437,0.180328,-0.292929,0,0.00149028,-0.53117,-0.0333333]]))
# print("predict (after training)", model.forward(hour_var).data[0][0])
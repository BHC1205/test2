import torch
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

# print(torch.ones((2,3)))
# A=np.ones((3,2))
# # print(A[1])
# A[1,:]=2
# A=torch.tensor(A)
# B=torch.ones_like(A)
# print(id(A))
# # A=A+B
# # A+=B
# A[:]=A+B
# A=A.numpy()
# print((A))
# print(B)

# a=torch.tensor([2.2])
# print(type(a.item()))
# print(type(float(a)))

# import os
# os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# data_file = os.path.join('..', 'data', 'house_tiny.csv')
# with open(data_file, 'w') as f:
#     f.write('NumRooms,Alley,Price\n')  # 列名
#     f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
#     f.write('2,NA,106000\n')
#     f.write('4,NA,178100\n')
#     f.write('NA,NA,140000\n')
# import pandas as pd
# data=pd.read_csv(data_file)
# print(data)
# print(data.isna().sum())
# x=data.isna().sum()
# print(x.index[x.argmax()])

# inputs,outputs=data.iloc[:,:2],data.iloc[:,-1]
# # print(inputs,'\n',outputs)
# inputs=inputs.fillna(inputs.mean())
# inputs=pd.get_dummies(inputs,dummy_na=True)
# # print(inputs.values)
# X,y=torch.tensor(inputs.values),torch.tensor(outputs.values)
# print(X,y)

# data = {
#   "calories": [420, 380, 390],
#   "duration": [50, 40, 45]
# }

# 数据载入到 DataFrame 对象
# df = pd.DataFrame(data)
# print(df)
# # 返回第一行
# print(df.iloc[:,0])
# # 返回第二行
# print(df['duration'])

# x=torch.arange(40,dtype=torch.float32).reshape(2,5,4)
# print(x)
# xSum=x.sum(axis=0,keepdims=True)
# print(xSum)
# print(xSum.shape)
# print(x.cumsum(axis=0))
# print(x/xSum)
# A=torch.arange(12).reshape(4,3)
# print(A)
# print(A.sum(axis=0))
# print(A.cumsum(axis=0))

# x=torch.arange(4.0,requires_grad=True)
# x.requires_grad(True)
# y=torch.matmul(x,x)
# y.backward()
#
# # x.grad.zero_()  #_下划线表示在torch中写入前项
# # y=x.sum()
# # y.backward()
# #
# # x.grad.zero_()
# # y=x*x
# # # 等价y.backward(torch.ones(len(x)))
# # y.sum().backward()
# print(y,x.grad)

# x=torch.arange(0,20.0,0.1,requires_grad=True)
# y=torch.sin(x)
# y.sum().backward()
# # print(y,x.grad)
# # print(x.grad==torch.cos(x))

# plt.plot(x.detach(),y.detach(),label='sin x')
# plt.plot(x.detach(),x.grad,label='cos x')
# plt.legend()
# plt.show()

# x=torch.normal(0, 1, (500,1))
# x=x.numpy()
# x=np.random.normal(loc=0,scale=1.0,size=500)
# plt.hist(x)
# print(x)
# plt.show()
# from torch import nn
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
#     # forward 定义前向传播
#     def forward(self, x):
#         y = self.linear(x)
#         return y
#
# net = LinearNet(2)
# print(net) # 使用print可以打印出网络的结构

# a=torch.tensor([[1,2],[2,3]])
# print(a)
# print(a.sum(axis=1,keepdim=True))

# a=torch.arange(6).reshape(2,3)
# print(a)
# b=torch.zeros(a.shape)
# print(b)
# print(1==True)
# print(len(a))
# print(torch.cuda.device_count())
# print(torch.rand(50))
# a=(torch.rand(100))
# cnt=0
# for b in a:
#     if b>=0.5:
#         cnt+=1
#
# print(cnt)
# mask=(a>0.5)
# print(mask*2)

# a=(torch.randint(10000,(1,100)).float())
# print(type(((a-a.mean())/a.std())[0][0].numpy()))
# print(a)


# plt.scatter(features[:,0].detach(),labels.detach(),1)
# plt.show()

#torch 算子
# import yaml
# import torch
# model = torch.jit.load('model.ptl')
# ops = torch.jit.export_opnames(model)
# with open('model.yaml', 'w') as output:
#     yaml.dump(ops, output)

# a=torch.arange(12).reshape(2,3,2)
# print(a)
# print(a[:,None])
# print(d2l.sequence_mask(a,torch.tensor([0,1])))
# d2l.DotProductAttention()
# d2l.AdditiveAttention()
# nn.Embedding()

# ln = nn.LayerNorm(2)
# bn = nn.BatchNorm1d(2)
# X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# # 在训练模式下计算 `X` 的均值和方差
# print('layer norm:', ln(X), '\nbatch norm:', bn(X))

# print([1]*4)
# print(torch.arange(4).repeat(5,1))
d2l.EncoderDecoder
d2l.train_seq2seq
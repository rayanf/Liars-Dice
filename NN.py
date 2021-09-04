import numpy as np
import math


class NN:
    def __init__(self):
        self.param = 0


    def initialize_parametrs(self):
        self.parametrs = {}
        self.parametrs['W1'] = np.random.randn(8,12) * 0.001
        self.parametrs['b1'] = np.zeros((8,1))
        self.parametrs['W2'] = np.random.randn(8,8) * 0.001
        self.parametrs['b2'] = np.zeros((8,1))
        self.parametrs['W3'] = np.random.randn(4,8) * 0.001
        self.parametrs['b3'] = np.zeros((4,1))
        return self.parametrs


    def forward_prop(self,Ap,W,b):
        z = np.dot(W,Ap) + b

        forward_data = (Ap, W ,b)
        return  z,forward_data

    def forward_activation(self,Ap,W,b,activation):
        z, forward_data = self.forward_prop(Ap,W,b)

        if activation == 'relu':
            A = relu(z)
        elif activation == 'sigmoid':
            A = sigmoid(z)

        return A ,(forward_data,z)

    def forwardModel(self,X,parameters):
        datas = []
        A = X
        for i in range(2):
            Ap = A
            if i == 1:
                A, data = self.forward_activation(Ap, parameters['W'+str(i+1)], parameters['b'+str(i+1)], 'sigmoid')
                datas.append(data)
            else:
                A, data = self.forward_activation(Ap, parameters['W'+str(i+1)], parameters['b'+str(i+1)], 'relu')
                datas.append(data)

        yhat, data = self.forward_activation(Ap, parameters['W'+str(3)], parameters['b'+str(3)], 'relu')
        datas.append(data)
        return yhat,datas

    def cost(self,yhat,y):
        m = y.shape[1]
        cost = (yhat - y)**2
        cost = sum(sum(cost))/(4*m)
        return cost


    def back_prop(self,dz,data):
        Ap ,w ,b = data
        m = Ap.shape[1]

        dw = np.dot(dz,Ap.T)/m
        db = np.sum(dz,axis=1,keepdims=True)/m


        dAp = np.dot(w.T,dz)
        return dAp, dw, db


    def back_activation(self,dA,data,activation):
        forward_data ,z = data
        if activation == 'relu':
            dz = relu_backward(dA, z)
            dAp, dw, db = self.back_prop(dz, forward_data)
        else:
            dz = sigmoid_backward(dA, z)
            dAp, dw, db = self.back_prop(dz, forward_data)

        return dAp, dw ,db

    def backwardModel(self,yhat,y,datas):
        dyhat = 2*(yhat-y)
        grads = {}
        current_data = datas[-1]
        grads['dA2'], grads['dW3'], grads['db3'] = self.back_activation(dyhat,current_data,'relu')


        for l in reversed(range(2)):
            current_data = datas[l]
            if l == 1:
                dAp_temp, dW_temp, db_temp = self.back_activation(grads["dA"+ str(l+1)],current_data,'sigmoid')
            else:
                dAp_temp, dW_temp, db_temp = self.back_activation(grads["dA"+ str(l+1)],current_data,'relu')
            grads["dA" + str(l)] = dAp_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def parametrs_Update(self,parameters,grads,learning_rate):
        for l in range(3):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

        return parameters



    def run(self,X,Y):
        if not self.param:
            self.parametrs = self.initialize_parametrs()
            self.param = 1
        yhat, datas = self.forwardModel(X,self.parametrs)
        grades = self.backwardModel(yhat,Y,datas)
        cost = self.cost(yhat,Y)
        self.parametrs = self.parametrs_Update(self.parametrs,grades,0.05)
        return self.parametrs, cost



    def predict(self,X,Y):
        yhat, datas = self.forwardModel(X,self.parametrs)
        cost = self.cost(yhat,Y)
        yhat = yhat.T
        for i in range(yhat.shape[0]):
            for j in range(yhat.shape[1]):
                yhat[i][j] = yhat[i][j]
                ceil = math.ceil(yhat[i][j])
                floor = math.floor(yhat[i][j])
                yhat[i][j] = floor if abs(ceil - yhat[i][j])> abs(floor - yhat[i][j]) else ceil

        return cost,yhat



def relu(z):
    for i in range(len(z)):
        for j in range(len(z[i])):
            z[i][j] = max(0,z[i][j])

    return z

def relu_backward(dA,z):
    gprimz = np.zeros((z.shape[0],z.shape[1]))
    for i in range(len(z)):
        for j in range(len(z[i])):
            if j >= 0:
                gprimz[i][j] = 1
            else:
                gprimz[i][j] = 0

    return np.multiply(dA,gprimz)




def sigmoid(z):
    for i in range(len(z)):
        for j in range(len(z[i])):
            z[i][j] = max(0,z[i][j])

            # z[i][j] = 1 / (1 + math.exp(-z[i][j]))
            return 1 / (1 + math.exp(-z[i][j]))


def sigmoid_backward(dA,z):
     gprimz = np.zeros((z.shape[0],z.shape[1]))
     for i in range(len(z)):
         for j in range(len(z[i])):
            gprimz[i][j] = 1 / (1 + math.exp(-z[i][j])) * (1 - (1 / (1 + math.exp(-z[i][j]))))



     return np.multiply(dA,gprimz)

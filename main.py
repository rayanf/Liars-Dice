import numpy as np
import random
import strategic as Player
import NN

class Game:
    def __init__(self,numberOfplayers):
        self.numberOfplayers = numberOfplayers

    def playersMaker(self):
        self.players = []
        for p in range(self.numberOfplayers-1):
            p = Player.Player()
            self.players.append(p)

    def initial_dics(self,numberOfplayers):
        self.randdic = []
        self.realcount = [0,0,0,0]
        for i in range(numberOfplayers):
            p = []
            for i in range(5):
                p.append(random.randint(1,4))
            self.randdic.append(p)

            for i in range(4):
                self.realcount[i] += p.count(i+1)

    def play_one_roundd(self):
        self.initial_dics(self.numberOfplayers)
        self.result = []
        for p in range(len(self.players)):
            self.players[p].initcards(self.randdic[p].count(1),
                                    self.randdic[p].count(2),
                                    self.randdic[p].count(3),
                                    self.randdic[p].count(4),
                                    self.numberOfplayers*5)


            self.result.append(self.players[p].play())

        return self.result,self.realcount







def preprocess(myDicks,guess,truth,numberOfplayers):
    countNumGuess = [0,0,0,0]
    sumNumGuess   = [0,0,0,0]

    for i in range(numberOfplayers-1):
        keys = guess[i].keys()
        for k in keys:
            countNumGuess[k-1] += 1
            sumNumGuess[k-1] += guess[i][k]

    myDicks = [myDicks.count(1),myDicks.count(2),myDicks.count(3),myDicks.count(4)]
    exmple = myDicks + countNumGuess + sumNumGuess
    x = np.array([exmple])
    y = np.array([truth])

    return x.T , y


def get_data(iteration,numberOfplayers):
    for t in range(iteration):
        game = Game(numberOfplayers)
        game.playersMaker()
        guess,truth = game.play_one_roundd()
        mydics = game.randdic[-1]
        x,y = preprocess(mydics,guess,truth,numberOfplayers)

        if t == 0:
            X = np.array(x)
            Y = np.array(y)
        else:
            X = np.concatenate((X,x),axis=1)
            Y = np.concatenate((Y,y),axis=0)

    return X,Y




def train(iteration,numberOfplayers,epoch):
    X,Y = get_data(1000,5)
    for i in range(epoch):
        parametrs = network.run(X,Y.T)

        if not i % 10:
            print(i)

    return parametrs




def test():
    X,Y = get_data(1000,5)
    cost,yhat = network.predict(X,Y.T)

    print(Y)
    print('-----------------------------------------------------------')
    print(cost)
    print(yhat)



if __name__=='__main__':
    network = NN.NN()
    train(5000,8,200)
    test()

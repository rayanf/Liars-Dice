import math


class Player:
    def __init__(self):
            pass
        # self.most_common = lambda : self.numbers.index(max(self.numbers)) + 1
    def initcards(self,num1,num2,num3,num4,num_all):
        self.numbers = [num1,num2,num3,num4]
        self.num_all = num_all

        self.common = self.numbers.index(max(self.numbers)) + 1


    def guess(self):
        prob = self.num_all / 4
        ceil = math.ceil(prob)
        floor = math.floor(prob)
        prob = floor if abs(ceil - prob)> abs(floor - prob) else ceil

        return {self.common :prob + max(self.numbers)}

    def play(self):
        guess_ansewr = self.guess()
        return(guess_ansewr)


def play_one_round(cart_list,num_all):
    player = Player()
    player.initcards(cart_list.count(1),
                    cart_list.count(2),
                    cart_list.count(3),
                    cart_list.count(4),
                    num_all)

    try:
        player_guess = player.play()
        print(player_guess)

    except:
        print('something wrong please try again')
        l, num_all = get_input()
        play_one_round(l,num_all)





def get_input():
    l = input('list of my cart: ').split()
    num_all = int(input('number of all cart: '))
    l = list(map(int,l))

    return l,num_all

if __name__ == '__main__':
    l, num_all = get_input()

    play_one_round(l,num_all)

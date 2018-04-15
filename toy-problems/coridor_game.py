import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import os

class Coridor:
    # N - black hole size
    def __init__(self, length = 10):

        self.length = length
        self.states = np.arange(length)

        self.actions = np.array([-1, 1]) #left, up, right, down
        self.start = length//2
        self.goal = length
        self.state = self.start

        self.prepare_strategy()
        self.prepare_image()
        self.game_over = False

    def prepare_image(self):
        self.image = np.zeros((self.length,1))

        # for s in self.states:
        #     plt.text(s[0]-0.5,s[1],str(self.strategy[tuple(s)]))
        #     plt.arrow(s[0], s[1], 0.3*self.strategy[tuple(s)][0], 0.3*self.strategy[tuple(s)][1], head_width=0.1, head_length=0.2)

        # plt.imshow(self.image.T)#, cmap=plt.get_cmap('inferno'))
        # plt.colorbar()
        # # plt.waitforbuttonpress()
        # plt.show()

    #function applies action and returns a reward
    def do_action(self):
        self.state += int(self.strategy[self.state])
        if (self.state == self.goal):
            self.game_over = True

    def is_out_of_bounds(self, state):
        return not state in self.states

    def record_trajectories(self, n):
        n_traj = len(os.listdir('cor_trajectories'))

        for i in range(n_traj, n_traj+n):

            out_file = open('cor_trajectories/traj' + str(i) + '.csv','w+')
            self.refresh()
            states = ''
            sts = []
            while not self.game_over :
                states += str(self.state) + ', ' + str(self.strategy[self.state]) + '\n'
                self.do_action()
                sts.append(self.state)
            out_file.writelines(str(states))
            # im = self.image.copy()
            # for s in sts:
            #     im[tuple(s)] = 2.5
            # self.image[tuple(self.goal)] = 2
            # plt.imshow(im.T)
            # plt.show()
            out_file.close()


    def prepare_strategy(self):
        self.strategy = np.ones(self.length)

    def refresh(self):
        self.game_over = False
        self.state = np.random.choice(range(self.length-2))
        print(self.state)


if __name__ == '__main__':
    bh = Coridor(10)
    bh.record_trajectories(100)
    # k = np.loadtxt('trajectories/traj0.csv',dtype=int, delimiter=',')
    # print(k)
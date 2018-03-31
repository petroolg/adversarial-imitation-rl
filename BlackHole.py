import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
import os

class BlackHole:
    # N - black hole size
    def __init__(self, bh, border=3):

        self.bh = bh
        self.border = border
        self.N = bh+border*2
        x = np.arange(self.N)
        xv, yv = np.meshgrid(x, x)
        self.states = (np.dstack([xv,yv]).reshape([self.N**2,2]))

        self.actions = np.array([[-1,0],[0,1],[1,0],[0,-1]]) #left, up, right, down
        self.start = np.array([2, 0])
        # self.goal = [self.N-2, int(self.N/2)]
        self.goal = np.array([self.N//2,self.N-1])
        self.state = self.start
        self.black_hole = np.array([s for s in self.states if abs(s[0] - self.N//2) <= bh / 2 and abs(s[1] - self.N // 2) <= bh / 2])
        self.dang_region = np.array([s for s in self.states if abs(s[0] - self.N//2) <= bh/2+1 and abs(s[1] - self.N//2 ) <= bh/2+1])
        self.prepare_gradient()
        self.prepare_strategy()
        self.prepare_image()
        self.initQ()
        self.game_over = False

        self.power = 0.5

    def prepare_image(self):
        self.image = np.zeros([self.N,self.N])
        self.image[:, :] = 3

        self.image[list(zip(self.dang_region.T))] = 1
        self.image[list(zip(self.black_hole.T))] = 0
        self.dang_region = self.dang_region.tolist()

        #
        # for s in self.states:
        #     plt.text(s[0]-0.5,s[1],str(self.strategy[tuple(s)]))
        #     plt.arrow(s[0], s[1], 0.3*self.strategy[tuple(s)][0], 0.3*self.strategy[tuple(s)][1], head_width=0.1, head_length=0.2)

        self.image[tuple(self.goal)] = 2
        self.image[tuple(self.start)] = 2
        # plt.imshow(self.image.T)#, cmap=plt.get_cmap('inferno'))
        # plt.colorbar()
        # # plt.waitforbuttonpress()
        # plt.show()

    #function applies action and returns a reward
    def do_action(self):
        # chance of getting sucked by hole
        fort = np.random.rand() < 0.5
        self.state += self.strategy[tuple(np.array(self.state).astype(int))].astype(int)
        if fort:
            self.state = self.state + self.gradient[tuple(self.state.astype(int))].astype(int)
        if (self.state == self.goal).all():
            self.game_over = True
        if self.state.tolist() in self.black_hole.tolist():
            self.game_over = True
            return True
        return False

    def initQ(self):
        self.Q = {}
        for s in self.states:
            for a in self.actions:
                self.Q[(tuple(s),tuple(a))] = 0

    def is_out_of_bounds(self, state):
        return not np.array(state).tolist() in self.states.tolist()

    def is_in_dang_region(self,state):
        return np.array(state).tolist() in self.black_hole.tolist()

    def record_trajectories(self, n):
        n_traj = len(os.listdir('trajectories'))

        for i in range(n_traj, n_traj+n):

            out_file = open('trajectories/traj' + str(i) + '.csv','w+')
            self.refresh()
            states = ''
            sts = []
            while not self.game_over :
                states += str(np.array(self.state, dtype=int).tolist())[1:-1] + ', '+ str(self.strategy[tuple(np.array(self.state,dtype=int))].astype(int).tolist())[1:-1]+'\n'
                sts.append(self.state.copy())
                if self.do_action():
                    self.refresh()
                    states = ''
            out_file.writelines(str(states))
            # im = self.image.copy()
            # for s in sts:
            #     im[tuple(s)] = 2.5
            # self.image[tuple(self.goal)] = 2
            # plt.imshow(im.T)
            # plt.show()
            out_file.close()


    def prepare_strategy(self):
        self.strategy = -self.gradient
        self.strategy = np.zeros([self.N,self.N,2])
        for s in self.states:
            a = [0,0]
            if (abs(s[0] - self.goal[0]) >= abs(s[1] - self.goal[1]) or abs(s[1] - self.goal[1]) >= self.N//2 - self.bh) and (s[0] !=0 and s[0]!= self.N-1 or s[1] == self.N-1):
                if np.sign(s[0] - self.goal[0]) != 0:
                    a = [-np.sign(s[1] - self.goal[1]+self.N//2-self.bh//2-self.border//2+0.5)*np.sign(s[0] - self.goal[0]),0]
                elif s[1] - self.goal[1]+self.N//2-self.bh/2+1 > 0:
                    a = [0, -np.sign(s[1] - self.goal[1])]
                else:
                    a = [-1, 0]
            else:
                a = [0,1]
            self.strategy[tuple(s)] = a
        self.strategy[list(zip(self.dang_region.T))] = -self.gradient[list(zip(self.dang_region.T))]
        # self.strategy += np.array([[(self.goal[0]-s[0])**2,(self.goal[1]-s[1])**2] for s in self.states]).reshape([self.N,self.N,2])
        # self.strategy = np.sign(self.strategy)
        str_file = open('strategy.txt', 'w+')
        str_file.writelines('\n'.join([str((t[0].tolist(), t[1].tolist())) for t in zip(self.states, self.strategy.reshape((self.N*self.N,2),order='F'))]))
        str_file.close()

    def prepare_gradient(self):
        self.gradient = np.zeros([self.N, self.N, 2])
        for s in self.states:
            x = s[0] - self.N//2
            y = s[1] - self.N//2


            if y < x and y > -x:
                self.gradient[tuple(s)] = np.array([-1,0])
            if y > x and y < -x:
                self.gradient[tuple(s)] = np.array([1,0])


            if y >= x and y > -x or y > x and y >= -x:
                self.gradient[tuple(s)] = np.array([0,-1])
            if y <= x and y < -x or y < x and y <= -x:
                self.gradient[tuple(s)] = np.array([0, 1])



    def refresh(self):
        self.game_over = False
        # self.state = np.random.randint(0,self.N-1,[2],dtype=int)
        choices = list(set([tuple(s) for s in self.states]) - set([tuple(s) for s in self.black_hole]) - set([tuple(self.goal)]))
        self.state = np.array(choices[np.random.choice(len(choices))])
        print(self.state)


if __name__ == '__main__':
    bh = BlackHole(1, 2)
    bh.record_trajectories(300)
    # k = np.loadtxt('trajectories/traj0.csv',dtype=int, delimiter=',')
    # print(k)
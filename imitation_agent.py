import numpy as np
import tensorflow as tf
import os
import logging
from BlackHole import *

N = 5
GAMMA = 0.9

def prepare_sa(N):
    states = np.repeat(np.eye(N*N),4,axis=0)
    actions = np.tile(np.eye(4), N*N).T
    return np.hstack((states, actions)).astype(dtype=np.float32)

# def prepare_sa(N):
#     x = np.arange(N)
#     xv, yv = np.meshgrid(x, x)
#     states = (np.dstack([xv, yv]).reshape([N ** 2, 2]))
#     states = np.repeat(states,4,axis=0)
#     actions = np.array([[-1,0],[0,1],[1,0],[0,-1]])
#     return np.hstack((states, actions)).astype(dtype=np.float32)

def from_onehot_to_raw(states, N):
    states = np.atleast_2d(states)
    slen = states.shape[0]
    state_ind = np.array([pos.index(1) for pos in states[:,:-4].tolist()])[np.newaxis].T
    x = state_ind//N
    y = state_ind%N
    a_ind = np.array([act.index(1) for act in states[:,-4:].tolist()])*2-3
    act = np.zeros((slen,2))
    act[a_ind == -3, 1] = -1
    act[a_ind == 3, 1] = 1
    act[a_ind == 1, 0] = 1
    act[a_ind == -1, 0] = -1

    return np.hstack((x,y,act))

sa_pairs = prepare_sa(N)
sa_pairs_raw = from_onehot_to_raw(sa_pairs,N)

def one_hot(states, N):
    # print(states)
    states = np.atleast_2d(states)
    slen = states.shape[0]
    out = np.zeros([slen, N * N])
    st = (states[:,0]*N+states[:,1]).astype(int)
    out[range(slen),st] = 1.0
    actions = np.zeros((slen,4))
    k = states[:,-2:]
    actions[range(slen), ((k[:,1]*3+k[:,0]+3)/2).astype(int)] = 1
    out = np.hstack([out,actions])
    return out

class SGDRegressor_occupancy:
    def __init__(self, N, xd, n_pairs):
        lr = 10e-2
        lambda_ = 0.005

        # self.sa_pairs = prepare_sa(N)
        # self.n_pairs, xd = self.sa_pairs.shape

        self.sa_pairs = tf.placeholder(tf.float32, shape=[None,xd], name='sa_pairs')

        self.w0 = tf.Variable(tf.random_normal(shape=[xd, 30]), name='w0')
        self.b0 = tf.Variable(tf.random_normal(shape=[1, 30]), name='b0')
        self.w1 = tf.Variable(tf.random_normal(shape=[30, 10]), name='w1')
        self.b1 = tf.Variable(tf.random_normal(shape=[1,10]), name='b1')
        self.w2 = tf.Variable(tf.random_normal(shape=[10,1]), name='w2')


        self.expert_occ_measure = tf.placeholder(tf.float32, shape=(None), name='EOM')
        self.occ_measure = tf.placeholder(tf.float32, shape=(None), name='OM')
        self.occ_measure_for_Q = tf.placeholder(tf.float32, shape=(None,None), name='QOM')

        self.theta0 = tf.Variable(tf.random_normal(shape=[xd, 30]), name='theta0')
        self.btheta0 = tf.Variable(tf.random_normal(shape=[1, 30]), name='btheta0')
        self.theta1 = tf.Variable(tf.random_normal(shape=[30, 10]), name='theta1')
        self.btheta1 = tf.Variable(tf.random_normal(shape=[1, 10]), name='btheta1')
        self.theta2 = tf.Variable(tf.random_normal(shape=[10, 1]), name='theta2')

        # make prediction and cost of discriminator
        Dw0 = tf.nn.sigmoid(tf.matmul(self.sa_pairs, self.w0)+self.b0)
        Dw1 = tf.nn.sigmoid(tf.matmul(Dw0, self.w1) + self.b1)
        self.Dw = tf.nn.sigmoid(tf.matmul(Dw1, self.w2))
        self.cost_D = tf.matmul(self.occ_measure,tf.log(self.Dw)) + tf.matmul(self.expert_occ_measure, tf.log(1.0-self.Dw))

        pi0 = tf.nn.sigmoid(tf.matmul(self.sa_pairs, self.theta0) + self.btheta0)
        pi1 = tf.nn.sigmoid(tf.matmul(pi0, self.theta1) + self.btheta1)
        self.pi = tf.nn.sigmoid(tf.matmul(pi1, self.theta2))

        # prediction and cost of policy
        self.H = -tf.reduce_sum(tf.multiply(self.pi, tf.log(self.pi)))
        self.Q = tf.matmul(self.occ_measure_for_Q,tf.log(self.Dw))
        self.piQ = tf.multiply(tf.log(self.pi),self.Q)

        self.cost_pi = tf.matmul(self.occ_measure,self.piQ) - lambda_*self.H

        # ops we want to call later
        self.train_D = tf.train.AdamOptimizer(learning_rate=lr).minimize(-self.cost_D)
        self.train_pi = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost_pi, var_list=[self.theta0, self.theta1, self.theta2, self.btheta0,self.btheta1])


        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit_D(self, OM, eOM):
        self.session.run(self.train_D, feed_dict={self.occ_measure:np.atleast_2d(OM),self.expert_occ_measure:np.atleast_2d(eOM), self.sa_pairs: sa_pairs})

    def partial_fit_policy(self, OM, QOM):
        self.session.run(self.train_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})

    def predict_action_prob(self):
        # print('Predict action prob: ', state_action.shape)
        policy =  self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        return policy

    def comp_Dw(self):
        # print('Predict action prob: ', state_action.shape)
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa_pairs})

    def get_omega(self):
        w0 = self.session.run(self.w0)
        w1 = self.session.run(self.w1)
        w2 = self.session.run(self.w2)
        return w0, w1, w2

    def get_theta(self):
        theta0 = self.session.run(self.theta0)
        theta1 = self.session.run(self.theta1)
        theta2 = self.session.run(self.theta2)
        return theta0, theta1, theta2


    def comp_Dw_part(self, sa):
        # print('Predict action prob: ', state_action.shape)
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa})

    def comp_log_Dw(self):
        return self.session.run(tf.log(1.0 - self.Dw), feed_dict={self.sa_pairs: sa_pairs})

    def show_bastards(self, OM, QOM, eOM):
        # logging.debug('\n Occ measure:\n' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), OM.tolist()))))

        # logging.debug('\nQ\n '+ '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), self.session.run(self.Q, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs}).tolist()))))
        # logging.debug('\npiQ\n ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), self.session.run(self.piQ, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs}).tolist()))))
        logging.debug('costD ' + str(self.session.run(self.cost_D, feed_dict={self.occ_measure: np.atleast_2d(OM),self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs})))
        logging.debug('H ' + str(self.session.run(self.H, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM),
                                                                         self.occ_measure: np.atleast_2d(OM),
                                                                         self.expert_occ_measure: np.atleast_2d(eOM),
                                                                         self.sa_pairs: sa_pairs})))
        logging.debug('costpi ' + str(self.session.run(self.cost_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})))


# def occupancy_measure_exact(state_action, traj, policy):
#     prob = policy.predict(state_action)
#     max_d = max([len(t) for t in traj.shape[0]])
#     for i in range(max_d):
#

def predict_sa_prob(state_action,traj):
    n_taken, n_diff = 0, 0
    for trajectory in traj:
        for sa in trajectory:
            if (sa == state_action).all():
                n_taken += 1
            elif (sa[:-4] == state_action[:-4]).all():
                n_diff += 1
    if (n_taken+n_diff) == 0:
        return 0
    else:
        return n_taken/(n_taken+n_diff)

def occupancy_measure_approx_vector(traj):
    prob = np.zeros(N*N*4)
    if len(traj)==0:
        return prob
    max_d = max([np.array(t).shape[0] for t in traj])

    sa_x_traj = np.zeros((max_d, len(sa_pairs)))

    for t in traj:
        for i, sa in enumerate(t):
            sa_x_traj[i,sa_pairs.tolist().index(sa.tolist())] += 1
    prob = np.dot(np.array([GAMMA**i for i in range(max_d)]), (sa_x_traj.T/sa_x_traj.sum(axis=1)).T)
    # prob = np.dot(predict_sa_prob_vector(traj), prob)
    return prob

def occupancy_measure_approx(state_action, traj):
    prob = 0
    if len(traj)==0:
        return 0
    max_d = max([np.array(t).shape[0] for t in traj])
    for i in range(max_d):
        n_this, n_diff = 0,0
        for t in traj:
            if np.atleast_2d(t).shape[0] > i and (t[i] == state_action).all():
                n_this += 1
            else:
                n_diff += 1
        prob += GAMMA**i*n_this/(n_this+n_diff)
    prob *= predict_sa_prob(state_action, traj)
    return prob

def occ_measure(traj):
    om = []
    for sa in sa_pairs:
        om.append(occupancy_measure_approx(sa, traj))
    return np.array(om)

def crop_traj(sa, traj):
    new_traj = []
    for t in traj:
        if np.array(sa).tolist() in np.array(t).tolist():
            i = np.array(t).tolist().index(np.array(sa).tolist())
            new_traj.append(t[i:])
        # if np.array(sa).tolist() in np.array(t).tolist() and np.array(t).tolist().index(np.array(sa).tolist()) == 0:
        #     new_traj.append(t)
    return np.array(new_traj)


def occ_meaure_Q(traj):
    om = []
    for sa in sa_pairs:
        new_traj = crop_traj(sa, traj)
        oms=occupancy_measure_approx_vector(new_traj)
        om.append(oms)
        # show_om(oms,to_file=False)
    Q = np.array(om)
    # Q[Q<1e-5] = 1e-5
    # print(Q)
    return Q

def make_move(state, action, game:BlackHole):
    newstate = state + action
    # fort = np.random.rand() < 0.5

    # if game.is_in_dang_region(newstate):
    #     return state, True
    # if game.is_out_of_bounds(newstate):
    #     return state,False
    # if fort:
    #     newstate += game.gradient[tuple(newstate.astype(int))].astype(int)
    if game.is_in_dang_region(newstate):
        return state, True
    if game.is_out_of_bounds(newstate):
        return state,False
    return newstate, False

def sample_trajectories(game, model):
    trajectories=[]
    actions = np.eye(4)
    start= one_hot(np.hstack((game.start, [0, 0])),N)[0]
    fin = one_hot(np.hstack((game.goal, [0, 0])),N)[0]

    policy = model.predict_action_prob()

    while len(trajectories)<100:
        traj = []
        state = start.copy()
        game_over = False
        ts = 0
        while not (state[:-4] == fin[:-4]).all() and not game_over and ts < 30:
            probs = []
            for a in actions:
                p = policy[sa_pairs.tolist().index(np.hstack((state[:-4], a)).tolist()),0]
                probs.append(p)
            # print(probs)]
            # print(p, np.exp(probs), np.sum(np.exp(probs)))
            ind = np.random.choice(range(4), p=np.exp(probs) / np.sum(np.exp(probs)))

            action = actions[ind]
            old_state = np.hstack((state[:-4], action))
            traj.append(old_state.copy())
            sa = from_onehot_to_raw(old_state, N)[0]
            sa, game_over = make_move(sa[:2], sa[2:],game)
            state = np.hstack((sa,[0,0]))
            state = one_hot(state, N)[0]
            ts += 1
        trajectories.append(traj)
    return trajectories

def show_om(OM, to_file = False, filename = None):
    plt.clf()
    image = np.zeros((10*N, 10*N))
    x, y = np.meshgrid(np.arange(10*N), np.arange(10*N))
    for sa, om in zip(sa_pairs_raw, OM):
        if om > 0:
            if (sa[-2:] == [1,0]).all(): # right
                image[((y - 10 * sa[1] + 5 > x - 10 * sa[0]) & ((y - 10 * (sa[1]) - 15) < -(x - 10 * sa[0]))& (x >= 10 * (sa[0] + 1) - 3))] = om
            if (sa[-2:] == [-1,0]).all(): # left
                image[((y - 10 * sa[1] - 5 < x - 10 * sa[0]) & ((y - 10 * (sa[1]) - 5) > -(x - 10 * sa[0]))& (x <= 10 * sa[0] + 3))] = om
            if (sa[-2:] == [0,-1]).all(): # down
                image[((y - 10 * sa[1] + 5 > x - 10 * sa[0]) & ((y - 10 * (sa[1]) - 5) > -(x - 10 * sa[0])) & (y <= 10 * (sa[1]) + 3))] = om
            if (sa[-2:] == [0,1]).all(): # up
                image[((y - 10 * sa[1] - 5 < x - 10 * sa[0]) & ((y - 10 * (sa[1]) - 15) < -(x - 10 * sa[0])) & (y >= 10 * (sa[1] + 1) - 3))] = om
            # print(sa, om)
    plt.imshow(image)
    plt.colorbar()
    print(filename)
    if to_file:
        savepath = 'images/{}.png'.format(filename)
        plt.savefig(savepath)
        # print("File {} created.".format(savepath))
    else:
        plt.show()

def delete_imgs(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def measure_perf(expert_traj, model, agent_trajs):
    logging.debug('Mean D for expert_traj: ' +str(np.mean(model.comp_Dw_part(expert_traj[0]))))
    logging.debug('Mean D for agent_traj: ' + str(np.mean(model.comp_Dw_part(agent_trajs[0]))))

if __name__ == '__main__':
    expert_traj = []
    model = SGDRegressor_occupancy(N, 29, 0)
    game = BlackHole(1,2)

    for i, t in enumerate(os.listdir('trajectories')):
        raw_traj = np.loadtxt('trajectories/' + t, dtype=int, delimiter=',')
        onehot_traj = one_hot(raw_traj, N)
        expert_traj.append(onehot_traj)

    expert_traj = np.array(expert_traj)
    # eOM = occ_measure(expert_traj)
    eOM = occupancy_measure_approx_vector(expert_traj)
    show_om(eOM)

    delete_imgs('images/')
    logging.basicConfig(filename='images/black_hole.log', level=logging.DEBUG)

    for i in range(1000):
        print('{}/1000'.format(i))

        logging.debug('======Iteration #{}======'.format(i))
        # print(model.comp_Dw())

        agent_trajs = sample_trajectories(game,model)

        print('Dw: '+ '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), model.comp_Dw().tolist()))))
        # w0, w1, w2 =  model.get_omega()
        # print('w0 :', w0)
        # print('w1 :', w1)
        # print('w2 :', w2)
        # print('1 - Dw: ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), model.comp_log_Dw().tolist()))))

        # print(model.comp_Dw())
        # Update the Discriminator parameters from wi to wi+1 with the gradient
        OM = occupancy_measure_approx_vector(agent_trajs)
        model.partial_fit_D(OM, eOM)
        # print(model.comp_Dw())

        if i%1 == 0:
            show_om(OM, True, "img-{}".format(i))

        # Take a policy step from θi to θi+1, using the TRPO rule with cost function log(Dwi+1 (s, a)).
        QOM = occ_meaure_Q(agent_trajs)
        # model.show_bastards(OM, QOM)
        model.partial_fit_policy(OM, QOM)
        # model.show_bastards(OM, QOM)

        measure_perf(expert_traj, model, agent_trajs)
        model.show_bastards(OM, QOM, eOM)

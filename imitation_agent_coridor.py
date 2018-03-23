import numpy as np
import tensorflow as tf
import os
import logging
from coridor_game import *

N = 10
GAMMA = 0.9

graph_names = ['Mean D-score for agent', 'Mean D-score for expert', 'Policy cost', 'Discriminator cost']
graphs = [[],[],[],[]]

def prepare_sa(N):
    return np.eye(N*2)

def from_onehot_to_raw(states, N):
    states = np.atleast_2d(states)
    slen = states.shape[0]
    state_ind = np.array([pos.index(1) for pos in states.tolist()])[np.newaxis].T
    x = state_ind//2
    a = [[-1] if i==0 else [1] for i in state_ind%2]

    return np.hstack((x,a))

sa_pairs = prepare_sa(N)
sa_pairs_raw = from_onehot_to_raw(sa_pairs,N)

def one_hot(states, N):
    # print(states)
    states = np.atleast_2d(states)
    slen = states.shape[0]
    out = np.zeros([slen, N*2])
    st = (states[:,0]*2).astype(int)

    actions = np.array([0 if i == -1 else 1 for i in states[:,0]])[np.newaxis].T
    out[range(slen), st+actions] = 1.0
    return out

class SGDRegressor_occupancy:
    def __init__(self, N, xd):
        lr = 10e-2
        lambda_ = 0.0

        # self.sa_pairs = prepare_sa(N)
        # self.n_pairs, xd = self.sa_pairs.shape

        self.sa_pairs = tf.placeholder(tf.float32, shape=[None,xd], name='sa_pairs')

        self.w0 = tf.Variable(tf.random_normal(shape=[xd, 1], stddev=0.1), name='w0')
        self.b0 = tf.Variable(tf.random_normal(shape=[1], stddev=0.1), name='b0')

        self.expert_occ_measure = tf.placeholder(tf.float32, shape=(None), name='EOM')
        self.occ_measure = tf.placeholder(tf.float32, shape=(None), name='OM')
        self.occ_measure_for_Q = tf.placeholder(tf.float32, shape=(None,None), name='QOM')

        self.theta0 = tf.Variable(0.5*tf.ones(shape=[xd, 1]), name='theta0')

        # make prediction and cost of discriminator
        self.Dw = tf.nn.sigmoid(tf.matmul(self.sa_pairs,self.w0))
        # self.Dw = tf.constant(np.tile(np.array([[1.0],[0.0001]],dtype=np.float32).T,10).T)
        self.cost_D = tf.matmul(self.occ_measure,tf.log(self.Dw)) + tf.matmul(self.expert_occ_measure, tf.log(1.0-self.Dw))

        self.pi = self.theta0

        # prediction and cost of policy
        self.H = -tf.reduce_sum(tf.multiply(self.pi, tf.log(self.pi)))
        self.Q = tf.matmul(self.occ_measure_for_Q,tf.log(self.Dw))
        # self.Q = tf.log(self.Dw)
        self.piQ = tf.multiply(tf.log(self.pi),self.Q)

        self.cost_pi = tf.matmul(self.occ_measure,self.piQ) #- lambda_*self.H

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=lr)
        self.train_D = optimizer.minimize(-self.cost_D)
        self.train_pi = optimizer2.minimize(self.cost_pi, var_list=[self.theta0])

        self.grad_D = optimizer.compute_gradients(self.cost_D)
        self.grad_pi = optimizer2.compute_gradients(self.cost_pi, var_list=[self.theta0])

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit_D(self, OM, eOM):
        self.session.run(self.train_D, feed_dict={self.occ_measure:np.atleast_2d(OM),self.expert_occ_measure:np.atleast_2d(eOM), self.sa_pairs: sa_pairs})

    def partial_fit_policy(self, OM, QOM):
        policyb = self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        grads = self.session.run(self.grad_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})
        Q = self.session.run(self.Q, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.sa_pairs: sa_pairs})
        pi = self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        thetab = self.session.run(self.theta0)
        Dw = self.session.run(self.Dw, feed_dict={self.sa_pairs:sa_pairs})
        # for g in grads:
        #     for sg in g:
        #         if len(sg) > 1:
        #             plt.imshow(sg)
        #             plt.colorbar()
        #             plt.show()

        self.session.run(self.train_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})
        thetaa = self.session.run(self.theta0)
        policy = self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        print('Policy: ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), (policy-policyb).tolist(), policy.tolist()))))

    def predict_action_prob(self):
        # print('Predict action prob: ', state_action.shape)
        policy = self.session.run(self.pi)
        return policy

    def comp_Dw(self):
        # print('Predict action prob: ', state_action.shape)
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa_pairs})

    def comp_Dw_part(self, sa):
        # print('Predict action prob: ', state_action.shape)
        # sapairs = from_onehot_to_raw(sa,N)
        # out = np.array([int(a<0) for a in sapairs[:,1]])[np.newaxis].T
        # return out
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa})

    def get_omega(self):
        w0 = self.session.run(self.w0)
        return w0

    def get_theta(self):
        theta0 = self.session.run(self.theta0)
        return theta0

    def show_bastards(self, OM, QOM, eOM):

        policy_cost = self.session.run(self.cost_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})
        discr_cost = self.session.run(self.cost_D, feed_dict={self.occ_measure: np.atleast_2d(OM),self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs})
        entropy = self.session.run(self.H, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM),
                                                                         self.occ_measure: np.atleast_2d(OM),
                                                                         self.expert_occ_measure: np.atleast_2d(eOM),
                                                                         self.sa_pairs: sa_pairs})

        # logging.debug('\n Occ measure:\n' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), OM.tolist()))))

        # logging.debug('\nQ\n '+ '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), self.session.run(self.Q, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs}).tolist()))))
        # logging.debug('\npiQ\n ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), self.session.run(self.piQ, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.expert_occ_measure: np.atleast_2d(eOM), self.sa_pairs: sa_pairs}).tolist()))))
        logging.debug('costD ' + str(discr_cost))
        logging.debug('H ' + str(entropy))
        logging.debug('costpi ' + str(policy_cost))
        graphs[2].append(policy_cost[0,0])
        graphs[3].append(discr_cost[0,0])
        # graphs[4].append(entropy)

def occupancy_measure_approx_vector(traj):
    prob = np.zeros(N*2)
    if len(traj)==0:
        return prob
    max_d = max([np.array(t).shape[0] for t in traj])

    sa_x_traj = np.zeros((max_d, N))
    sa_a_traj = np.zeros((N, 2))
    for t in traj:
        for i, sa in enumerate(t):
            ind = sa.tolist().index(1)//2
            sa_x_traj[i,ind] += 1.0
            sa_a_traj[ind] += sa.tolist()[ind*2:ind*2+2]
    state_probs = np.nan_to_num(sa_x_traj.T / sa_x_traj.sum(axis=1))
    act_probs = np.nan_to_num(sa_a_traj.T / sa_a_traj.sum(axis=1))
    prob = np.dot(np.array([GAMMA**i for i in range(max_d)]), state_probs.T)
    # prob = np.dot(predict_sa_prob_vector(traj), prob)
    prob = np.repeat(prob, 2)*np.reshape(act_probs, (N*2),order='F')
    return prob

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

def make_move(state, action, game):
    newstate = state + action
    # fort = np.random.rand() < 0.5

    # if game.is_in_dang_region(newstate):
    #     return state, True
    # if game.is_out_of_bounds(newstate):
    #     return state,False
    # if fort:
    #     newstate += game.gradient[tuple(newstate.astype(int))].astype(int)
    if game.is_out_of_bounds(newstate):
        return state,False
    return newstate, False

def sample_trajectories(game, model):
    trajectories=[]
    actions = np.eye(4)
    fin = game.length

    policy = model.predict_action_prob()
    # print('Policy: ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), policy.tolist()))))
    while len(trajectories)<100:
        traj = []
        state_i = np.random.choice(range(game.length-2))
        state = np.zeros(N*2)
        state[state_i*2] = 1
        game_over = False
        ts = 0
        while not state.tolist().index(1)//2 == fin and not game_over and ts < 30:
            probs = []
            for i in range(2):
                p = policy[(state.tolist().index(1)//2)*2+i,0]
                probs.append(p)
            # print(probs)]
            # print(p, np.exp(probs), np.sum(np.exp(probs)))
            action = np.random.choice(range(2), p=np.exp(probs) / np.sum(np.exp(probs)))
            # ind = np.argmax(probs)
            old_state = np.zeros(N*2)
            old_state[(state.tolist().index(1)//2)*2+action] = 1.0
            traj.append(old_state.copy())
            sa = from_onehot_to_raw(old_state, N)[0]
            sa, game_over = make_move(sa[0], sa[1], game)
            state = np.array([sa,-1])
            state = one_hot(state, N)[0]
            ts += 1
        trajectories.append(traj)
    return trajectories

def show_om(OM, to_file = False, filename = None):
    plt.clf()
    image = np.zeros((10, 10*N))
    x, y = np.meshgrid(np.arange(10*N), np.arange(10))
    for sa, om in zip(sa_pairs_raw, OM):
        if om > 0:
            if sa[1] == 1: # right
                image[((y + 5 > x - 10 * sa[0]) & ((y - 15) < -(x - 10 * sa[0]))& (x >= 10 * (sa[0] + 1) - 3))] = om
            if sa[1] == -1: # left
                image[((y - 5 < x - 10 * sa[0]) & ((y - 5) > -(x - 10 * sa[0]))& (x <= 10 * sa[0] + 3))] = om
            # print(sa, om)
    plt.imshow(image)
    plt.colorbar()
    print(filename)
    if to_file:
        savepath = 'cor_images/{}.png'.format(filename)
        plt.savefig(savepath)
        # print("File {} created.".format(savepath))
        plt.close()
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
    ag, ex = [], []
    for i in range(5):
        ag.append(model.comp_Dw_part(agent_trajs[i]).ravel())
        ex.append(model.comp_Dw_part(expert_traj[0]))
    agent_score = np.mean(ag)
    expert_score = np.mean(ex)
    logging.debug('Mean D for expert_traj: ' + str(expert_score))
    logging.debug('Mean D for agent_traj: ' + str(agent_score))
    graphs[0].append(agent_score)
    graphs[1].append(expert_score)

def plot_graphs(path):
    try:
        if os.path.isfile(path):
            os.unlink(path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

    fig = plt.figure(figsize=(20,20))
    for i, graph_name  in enumerate(zip(graphs, graph_names)):
        plt.subplot(221+i)
        plt.plot(run_avg(graph_name[0]))
        plt.title(graph_name[1])
    plt.savefig(path)
    plt.close()

def run_avg(lst, bin=20):
    out = []
    l = len(lst)
    for i in range(l):
        out.append(np.mean(lst[max(0, i-bin):min(l-1, i+bin)]))
    return out

if __name__ == '__main__':
    expert_traj = []
    model = SGDRegressor_occupancy(N, N*2)
    game = Coridor(10)

    for i, t in enumerate(os.listdir('cor_trajectories')):
        raw_traj = np.loadtxt('cor_trajectories/' + t, dtype=int, delimiter=',')
        onehot_traj = one_hot(raw_traj, N)
        expert_traj.append(onehot_traj)

    expert_traj = np.array(expert_traj)
    # eOM = occ_measure(expert_traj)
    eOM = occupancy_measure_approx_vector(expert_traj)
    show_om(eOM)

    delete_imgs('cor_images/')
    logging.basicConfig(filename='cor_images/coridor.log', level=logging.DEBUG)

    for i in range(10000):
        print('{}/1000'.format(i))

        logging.debug('======Iteration #{}======'.format(i))
        # print(model.comp_Dw())

        agent_trajs = sample_trajectories(game,model)

        # print('Dw: '+ '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), model.comp_Dw().tolist()))))
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

        if i%10 == 0:
            show_om(OM, True, "img-{}".format(i))
            policy = model.session.run(model.pi, feed_dict={model.sa_pairs: sa_pairs})
            print('Policy: ' + '\n'.join(
            str(e) for e in list(zip(sa_pairs_raw.tolist(), policy.tolist()))))
            plot_graphs('cor_images/stat_graph.png')

        # Take a policy step from θi to θi+1, using the TRPO rule with cost function log(Dwi+1 (s, a)).
        QOM = occ_meaure_Q(agent_trajs)
        # model.show_bastards(OM, QOM)
        model.partial_fit_policy(OM, QOM)
        # model.show_bastards(OM, QOM)

        measure_perf(expert_traj, model, agent_trajs)
        model.show_bastards(OM, QOM, eOM)

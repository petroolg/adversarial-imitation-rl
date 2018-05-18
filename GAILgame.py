import tensorflow as tf
import numpy as np
import os
import shutil
import logging
from scipy import misc
import sklearn
from sklearn import preprocessing
from PythonClient.AirSimClient import *
import subprocess, signal

class CNN:

    def __init__(self, n_outputs, scope, time_steps=5):
        with tf.name_scope(scope):
            image_w = 128
            image_h = 72
            self.n_time_steps = time_steps

            print('\nBuilding the CNN.')
            # create inputs
            depth_map, segmentation, state_vector = [], [], []

            # set placeholders
            with tf.name_scope('input'):
                for i in range(self.n_time_steps):
                    depth_map.append(tf.placeholder(tf.float32, [None, image_h, image_w, 1], name='depth'+str(i)))
                    segmentation.append(tf.placeholder(tf.float32, [None, image_h, image_w, 3], name='segm'+str(i)))
                    state_vector.append(tf.placeholder(tf.float32, [None, 5], name='vector'+str(i)))

            #----------------------------------------------------------------------
            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                tf.summary.scalar('dropout_keep_probability', keep_prob)
            #----------------------------------------------------------------------
            # First conv+pool layer
            #----------------------------------------------------------------------
            specs_seg_1 = {'global_scope':'conv1',
                            'scope': 'segmentation',
                            'shape': [5, 5, 3, 32],
                            'pool_strides': [1, 2, 2, 1],
                            'pool_ksize' : [1, 2, 2, 1]}

            specs_depth_1 = {'global_scope':'conv1',
                             'scope': 'depth',
                             'shape': [5, 5, 1, 32],
                             'pool_strides': [1, 2, 2, 1],
                             'pool_ksize' : [1, 2, 2, 1]}

            conv_seg, conv_depth = [], []

            for i in range(self.n_time_steps):
                conv_seg.append(self.conv_layer(specs_seg_1, segmentation[i]))
                conv_depth.append(self.conv_layer(specs_depth_1, depth_map[i]))

            #----------------------------------------------------------------------
            # Second conv+pool layer
            #----------------------------------------------------------------------
            specs_seg_2 = {'global_scope':'conv2',
                            'scope': 'segmentation',
                            'shape': [5, 5, 32, 64],
                            'pool_strides': [1, 2, 2, 1],
                            'pool_ksize' : [1, 2, 2, 1]}

            specs_depth_2 = {'global_scope':'conv2',
                            'scope': 'depth',
                            'shape': [5, 5, 32, 64],
                            'pool_strides': [1, 2, 2, 1],
                            'pool_ksize' : [1, 2, 2, 1]}

            for i in range(self.n_time_steps):
                conv_seg[i] = self.conv_layer(specs_seg_2, conv_seg[i])
                conv_depth[i] = self.conv_layer(specs_depth_2, conv_depth[i])

            #----------------------------------------------------------------------
            # Reshape and oncatenate conv.layer outputs
            #----------------------------------------------------------------------

            for i in range(self.n_time_steps):
                conv_seg[i] = tf.reshape(conv_seg[i], [-1, 7 * 7 * 64])
                conv_depth[i] = tf.reshape(conv_depth[i], [-1, 7 * 7 * 64])

            seg_embedding_flat = tf.concat(conv_seg,axis=1)
            depth_embedding_flat = tf.concat(conv_depth,axis=1)
            state_flat = tf.reshape(state_vector, [-1, self.n_time_steps*5])
            embedding_flat = tf.concat([seg_embedding_flat, depth_embedding_flat, state_flat], axis=1)
            #----------------------------------------------------------------------
            # #1 fully connected layer
            #----------------------------------------------------------------------
            with tf.name_scope('fully_connected_1'):
                W_fc1 = tf.Variable(tf.truncated_normal([int(embedding_flat.shape[1]), 4096], stddev=0.1))
                b_fc1 = tf.Variable(tf.constant(0.1, shape=[4096]))
                h_fc1 = tf.nn.relu(tf.matmul(embedding_flat, W_fc1) + b_fc1)
                # Dropout
                #keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=1.0)

            # ----------------------------------------------------------------------
            # #2 fully connected layer
            # ----------------------------------------------------------------------
            with tf.name_scope('fully_connected_2'):
                W_fc2 = tf.Variable(tf.truncated_normal([4096, 1024], stddev=0.1))
                b_fc2 = tf.Variable(tf.constant(0.1, shape=[1024]))
                h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
                # Dropout
                h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob=1.0)
            #----------------------------------------------------------------------
            # Readout layer
            #----------------------------------------------------------------------
            with tf.name_scope('Readout_Layer'):
                W_fc3 = tf.Variable(tf.truncated_normal([1024, n_outputs], stddev=0.1))
                b_fc3 = tf.Variable(tf.constant(0.1, shape=[n_outputs]))
            # CNN output
            with tf.name_scope('Final_matmul'):
                self.y_conv = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
            #----------------------------------------------------------------------
            # Accuracy checks
            #----------------------------------------------------------------------
            # with tf.name_scope('accuracy'):
            #     with tf.name_scope('correct_prediction'):
            #         correct_prediction = tf.equal(tf.argmax(y_conv,1),
            #                                       tf.argmax(y_,1))
            #     with tf.name_scope('accuracy'):
            #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            # tf.summary.scalar('accuracy', accuracy)
            # print('CNN successfully built.')


    def conv_layer(self, specs, input):

        global_scope = specs.get('global_scope')
        scope = specs.get('scope')
        shape = specs.get('shape')
        pool_strides = specs.get('pool_strides')
        pool_ksize = specs.get('pool_ksize')

        with tf.name_scope(global_scope):
            with tf.name_scope('weights'+scope):
                W_conv1 = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(W_conv1)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(W_conv1 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(W_conv1))
                    tf.summary.scalar('min', tf.reduce_min(W_conv1))
                    tf.summary.histogram('histogram', W_conv1)

            with tf.name_scope('biases'+scope):
                b_conv1 = tf.Variable(tf.constant(0.1, shape=[shape[-1]]))
                with tf.name_scope('summaries'):
                    mean = tf.reduce_mean(b_conv1)
                    tf.summary.scalar('mean', mean)
                    with tf.name_scope('stddev'):
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(b_conv1 - mean)))
                    tf.summary.scalar('stddev', stddev)
                    tf.summary.scalar('max', tf.reduce_max(b_conv1))
                    tf.summary.scalar('min', tf.reduce_min(b_conv1))
                    tf.summary.histogram('histogram', b_conv1)
            with tf.name_scope('Wx_plus_b'+scope):
                preactivated1 = tf.nn.conv2d(input, W_conv1,
                                                  strides=[1,1,1,1],
                                                  padding='SAME') + b_conv1
                h_conv1 = tf.nn.relu(preactivated1)
                tf.summary.histogram('pre_activations', preactivated1)
                tf.summary.histogram('activations', h_conv1)
            with tf.name_scope('max_pool'+scope):
                h_pool =  tf.nn.max_pool(h_conv1,
                                          ksize=pool_ksize,
                                          strides=pool_strides,
                                          padding='SAME')
            # save output of conv layer to TensorBoard - first 16 filters
            with tf.name_scope('Image_output_'+global_scope+'_'+scope):
                image = h_conv1[0:1, :, :, 0:16]
                image = tf.transpose(image, perm=[3,1,2,0])
                tf.summary.image('Image_output_'+global_scope+'_'+scope, image)
        # save a visual representation of weights to TensorBoard
        with tf.name_scope('Visualise_weights_'+global_scope+'_'+scope):
            # We concatenate the filters into one image of row size 8 images
            W_a = W_conv1
            W_b = tf.split(W_a, shape[-1], 3)
            rows = []
            for i in range(int(shape[-1]/8)):
                x1 = i*8
                x2 = (i+1)*8
                row = tf.concat(W_b[x1:x2],0)
                rows.append(row)
            W_c = tf.concat(rows, 1)
            c_shape = W_c.get_shape().as_list()
            W_d = tf.reshape(W_c, [c_shape[2], c_shape[0], c_shape[1], 1])
            tf.summary.image("Visualize_kernels_"+global_scope+'_'+scope, W_d, 1024)
        return h_pool



class gailGame:
    def __init__(self):

        self.time_steps = 5

        self.discriminator = CNN(1, 'self.discriminator', time_steps=self.time_steps)
        # [nothing, gas, brake] x [straight, left, right]
        self.agent = CNN(9, 'self.agent', time_steps=self.time_steps)

        lr = 1e-3
        output_directory = 'AirSim_GAIG_logs'

        self.discriminator.occ_measure = tf.placeholder(tf.float32)
        self.discriminator.expert_occ_measure = tf.placeholder(tf.float32)
        self.agent.occ_measure = tf.placeholder(tf.float32)

        self.discriminator.Dw = self.discriminator.y_conv
        self.agent.pi = self.agent.y_conv

        self.discriminator.cost_D = tf.matmul(self.discriminator.occ_measure, tf.log(self.discriminator.Dw)) + \
                               tf.matmul(self.discriminator.expert_occ_measure, tf.log(1.0 - self.discriminator.Dw))

        self.agent.H = -tf.reduce_sum(tf.multiply(self.agent.pi, tf.log(self.agent.pi)))
        # self.Q = tf.matmul(self.occ_measure_for_Q,tf.log(self.Dw))
        self.agent.Q = tf.log(self.discriminator.Dw)
        self.agent.piQ = tf.multiply(tf.log(self.agent.pi), self.agent.Q)

        self.agent.cost_pi = tf.matmul(self.agent.occ_measure, self.agent.piQ)  # - lambda_*self.H

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=lr)
        self.discriminator.train_D = optimizer.minimize(-self.discriminator.cost_D)
        self.agent.train_pi = optimizer2.minimize(self.agent.cost_pi,var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='self.agent')) # TODO: var list!!!

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # ###START SESSSION AND COMMENCE TRAINING###
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # create session
        self.session = tf.Session()
        # initalise variables
        self.session.run(tf.global_variables_initializer())

        # Merge all the summaries and write them out to "mnist_logs"
        merged = tf.summary.merge_all()
        if not os.path.exists(output_directory):
            print('\nOutput directory does not exist - creating...')
            os.makedirs(output_directory)
            os.makedirs(output_directory + '/train')
            os.makedirs(output_directory + '/test')
            print('Output directory created.')
        else:
            print('\nOutput directory already exists - overwriting...')
            shutil.rmtree(output_directory, ignore_errors=True)
            os.makedirs(output_directory)
            os.makedirs(output_directory + '/train')
            os.makedirs(output_directory + '/test')
            print('Output directory overwitten.')
        # prepare log writers
        train_writer = tf.summary.FileWriter(output_directory + '/train', self.session.graph)
        test_writer = tf.summary.FileWriter(output_directory + '/test')
        roc_writer = tf.summary.FileWriter(output_directory)
        # prepare checkpoint writer
        saver = tf.train.Saver()
    
    def partial_fit_D(self, OM, eOM):
        self.session.run(self.discriminator.train_D, feed_dict={self.discriminator.occ_measure:np.atleast_2d(OM),self.discriminator.expert_occ_measure:np.atleast_2d(eOM)}) # TODO: right input

    def partial_fit_policy(self, OM, QOM):
        self.session.run(self.agent.traim_pi, feed_dict={self.agent.occ_measure_for_Q: np.atleast_2d(QOM), self.agent.occ_measure: np.atleast_2d(OM)}) # TODO: right input

    def predict_action_prob(self, data):
        inputs = ['depth' + str(i) for i in range(self.time_steps)] + ['segm' + str(i) for i in range(self.time_steps)] + ['vector' + str(i) for i in range(self.time_steps)]
        policy = self.session.run(self.agent.pi, feed_dict={i: d for i, d in zip(inputs, data)}) # TODO: right input
        return policy 

    def comp_Dw_part(self, sa):
        return self.session.run(self.discriminator.Dw, feed_dict={}) # TODO: right input

def measure_perf(expert_traj, model, agent_trajs):
    agent_score = np.mean(model.comp_Dw_part(agent_trajs[0]))
    expert_score = np.mean(model.comp_Dw_part(expert_traj[0]))
    logging.debug('Mean D for expert_traj: ' +str(expert_score))
    logging.debug('Mean D for agent_traj: ' + str(agent_score))
    graphs[0].append(agent_score)
    graphs[1].append(expert_score)

def create_traj(paths):
    # Timestamp  Speed (kmph)  Throttle  Steering  Brake  Gear  ImageName
    trajectories = []
    for path in paths:
        traj = []
        with open(path+'\\airsim_rec.txt', 'r') as file:
            file.readline() # get rid of header
            for i, line in enumerate(file):
                traj.append(line.split('\t'))
                imgs = traj[i][-1].split(';')
                traj[i].pop(-1)
                traj[i] = [float(x) for x in traj[i][:-1]]
                traj[i] = traj[i] + [img.strip() for img in imgs]
        trajectories.append(traj)
    print(traj)
    return np.array(trajectories)

def action_hash(action):
    # action: Throttle   Steering        Brake
    #         [0, 1]     int([-1, 1])    [0, 1]
    action = np.atleast_2d(action).astype(float).astype(int)

    hashes = action[:, 0] + action[:,2]*2 + (action[:,1]+1)*4

    return hashes

def img_hash(expert_trajectory_path, img_path):

    hashes = []

    for path in img_path:
        img = misc.imread(expert_trajectory_path+'\\images\\'+path)
        img = preprocessing.normalize(img[0])

        bits = "".join(map(lambda pixel: '1' if pixel < 0.5 else '0', img.ravel()))
        hexadecimal = int(bits, 2)
        hashes.append(hexadecimal)

    return hashes

def state_hash(path, state):
    # state: Speed (kmph)  Throttle  Steering  Brake  Gear  Image1 Image2

    action = np.atleast_2d(state)
    bins = np.linspace(-1, 1, 21)

    hashes = [hash(stsb) for stsb in zip(state[:, 0],
                                         state[:, 4],
                                         img_hash(path, state[:, 5]),
                                         img_hash(path,  state[:, 7]))]
    hashes = np.array(hashes) / 12
    return hashes


def sa_set_construct(expert_trajectory_paths, trajs):

    sa_dict = dict()
    sa_set = set()
    for traj, path in zip(trajs, expert_trajectory_paths):

        states = state_hash(path, traj)
        actions = action_hash(traj[:,2:5])

        for s, a in zip(states, actions):
            acts = sa_dict.get(s, set())
            acts = acts.union(set([a]))
            sa_dict[s] = acts

        sa_set = sa_set.union(set(states+actions))

    return list(sa_set), sa_dict.keys(), sa_dict.values()

def occupancy_measure(expert_trajectory_paths, trajs):

    # construct set of all state-action pairs
    # hash codes for s-a pair: [state hash/12 + action hash]

    sa_set, s_set, a_sets = sa_set_construct(expert_trajectory_paths, trajs)

    om = dict()
    max_d = max([len(t) for t in trajs])
    om.update(dict(zip(sa_set, np.zeros((len(sa_set), max_d)))))

    for t, path in zip(trajs, expert_trajectory_paths):
        # TODO: convert states to hash codes
        states = state_hash(path, t)
        actions = action_hash(t[:,2:5])

        vals = np.array([om.get(key) for key in states+actions])
        vals[range(len(t)), range(len(t))] += 1
        for sa, val in zip(states+actions, vals):
            om[sa] += val

    GAMMA = 0.9
    # TODO: compute actual probabilities
    out = dict()
    for state, actions in zip(s_set, a_sets):
        state_action_pairs = np.array([om.get(action+state) for action in actions])
        pi = np.sum(state_action_pairs,axis=1)/np.sum(state_action_pairs)
        P = np.nan_to_num(np.divide(state_action_pairs, np.sum(state_action_pairs, axis=0)),0)
        probs = [np.sum([s*GAMMA**i for i, s in enumerate(st)]) for st in P]
        out.update(dict(zip([action+state for action in actions], pi*probs)))
    print('\n'.join([str(k) for k in list(out.values())]))
    return out

def sample_trajectories(client, n_tries=30):

    # launch simulator
    os.system("C:/Users/Olga/Documents/CityEnviron/CityEnviron/CityEnviron.exe")

    client = CarClient()
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = CarControls()

    for i in range(n_tries):



        # TODO: throttle till full stop

    # restore to original state
    client.reset()
    client.enableApiControl(False)

    # kill simulator process
    p = subprocess.Popen(['tasklist'], stdout=subprocess.PIPE)

    out, err = p.communicate()

    for line in out.splitlines():
        if 'CityEnviron' in line:
            pid = int(line.split(None, 1)[0])
            os.kill(pid, signal.SIGKILL)
            break


if __name__== '__main__':

    # expert_trajectory_paths = ['C:\\Users\\Olga\\Documents\\AirSim\\2018-04-13-11-46-30']
    # # game = gailGame()
    # # print('Initialization completed')
    #
    # expert_traj = create_traj(expert_trajectory_paths)
    # # # eOM = occ_measure(expert_traj)
    # eOM = occupancy_measure(expert_trajectory_paths, expert_traj)

    # logging.basicConfig(filename='images/GAILgame.log', level=logging.DEBUG)

    # TODO: connect to simulator

    # T = 10e4
    # for i in range(T):
    #     print('{}/{}'.format(T,i))
    #     logging.debug('======Iteration #{}======'.format(i))
    #
    #     agent_trajs = sample_trajectories(model)
    #
    #     # Update the Discriminator parameters from wi to wi+1 with the gradient
    #     OM = occupancy_measure_approx_vector(agent_trajs)
    #     model.partial_fit_D(OM, eOM)
    #
    #     if i%10 == 0:
    #         plot_graphs('images/stat_graph.png')
    #
    #     # Take a policy step from θi to θi+1, using the TRPO rule with cost function log(Dwi+1 (s, a)).
    #     QOM = occ_meaure_Q(agent_trajs)
    #     model.partial_fit_policy(OM, QOM)
    #
    #     measure_performance(expert_traj, model, agent_trajs)
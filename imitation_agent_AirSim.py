import numpy as np
import tensorflow as tf
import os


class CNN:

    def __init__(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Dense Layer
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits Layer
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

class GAIlearning:
    def __init__(self, N, xd, n_pairs):
        lr = 10e-2
        lambda_ = 0.01

        # self.sa_pairs = prepare_sa(N)
        # self.n_pairs, xd = self.sa_pairs.shape

        self.sa_pairs = tf.placeholder(tf.float32, shape=[None,xd], name='sa_pairs')

        self.w0 = tf.Variable(tf.random_normal(shape=[xd, 10], stddev=0.1), name='w0')
        self.b0 = tf.Variable(tf.random_normal(shape=[1, 10], stddev=0.1), name='b0')
        # self.w1 = tf.Variable(tf.random_normal(shape=[30, 10]), name='w1')
        # self.b1 = tf.Variable(tf.random_normal(shape=[1,10]), name='b1')
        self.w2 = tf.Variable(tf.random_normal(shape=[10,1], stddev=0.1), name='w2')
        self.b2 = tf.Variable(tf.random_normal(shape=[1], stddev=0.1), name='b2')

        self.expert_occ_measure = tf.placeholder(tf.float32, shape=(None), name='EOM')
        self.occ_measure = tf.placeholder(tf.float32, shape=(None), name='OM')
        self.occ_measure_for_Q = tf.placeholder(tf.float32, shape=(None,None), name='QOM')

        self.theta0 = tf.Variable(tf.zeros(shape=[xd-4, 10]), name='theta0')
        self.btheta0 = tf.Variable(tf.zeros(shape=[1, 10]), name='btheta0')
        self.theta1 = tf.Variable(tf.zeros(shape=[10, 6]), name='theta1')
        self.btheta1 = tf.Variable(tf.zeros(shape=[1, 6]), name='btheta1')
        self.theta2 = tf.Variable(tf.zeros(shape=[6, 4]), name='theta2')
        self.btheta2 = tf.Variable(tf.zeros(shape=[4]), name='btheta2')

        # make prediction and cost of discriminator
        Dw0 = tf.nn.sigmoid(tf.matmul(self.sa_pairs, self.w0)+self.b0)
        # Dw1 = tf.nn.sigmoid(tf.matmul(Dw0, self.w1) + self.b1)
        self.Dw = tf.nn.sigmoid(tf.matmul(Dw0, self.w2) + self.b2)
       #  self.Dw = tf.constant(np.array([[1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  ,
       # 0.01, 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 0.01, 1.  , 1.  , 0.01,
       # 1.  , 1.  , 0.01, 1.  , 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  ,
       # 1.  , 1.  , 0.01, 1.  , 1.  , 0.01, 1.  , 1.  , 0.01, 1.  , 1.  ,
       # 0.01, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 1.  ,
       # 0.01, 1.  , 1.  , 1.  , 1.  , 1.  , 1.  , 0.01, 1.  , 0.01, 1.  ,
       # 1.  , 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  , 1.  , 0.01, 1.  ,
       # 0.01, 1.  , 1.  , 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  , 0.01,
       # 1.  , 1.  , 1.  , 0.01, 1.  , 1.  , 1.  , 0.01, 1.  , 0.01, 1.  ,
       # 1.  ]], dtype=np.float32).T)
        self.cost_D = tf.matmul(self.occ_measure,tf.log(self.Dw)) + tf.matmul(self.expert_occ_measure, tf.log(1.0-self.Dw))

        pi0 = tf.nn.sigmoid(self.theta0+self.btheta0)
        pi1 = tf.nn.sigmoid(tf.matmul(pi0, self.theta1) + self.btheta1)
        self.pi = tf.reshape(tf.nn.softmax(tf.matmul(pi1, self.theta2)+self.btheta2),(N*N*4,1))

        # prediction and cost of policy
        self.H = -tf.reduce_sum(tf.multiply(self.pi, tf.log(self.pi)))
        # self.Q = tf.matmul(self.occ_measure_for_Q,tf.log(self.Dw))
        self.Q = tf.log(self.Dw)
        self.piQ = tf.multiply(tf.log(self.pi),self.Q)

        self.cost_pi = tf.matmul(self.occ_measure,self.piQ) #- lambda_*self.H

        # ops we want to call later
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        optimizer2 = tf.train.AdamOptimizer(learning_rate=0.01)
        self.train_D = optimizer.minimize(-self.cost_D)
        self.train_pi = optimizer2.minimize(self.cost_pi, var_list=[self.theta0, self.btheta0,self.theta1, self.btheta1, self.theta2, self.btheta2])

        self.grad_D = optimizer.compute_gradients(self.cost_D)
        self.grad_pi = optimizer2.compute_gradients(self.cost_pi, var_list=[self.theta0])

        # start the session and initialize params
        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)

    def partial_fit_D(self, OM, eOM):
        self.session.run(self.train_D, feed_dict={self.occ_measure:np.atleast_2d(OM),self.expert_occ_measure:np.atleast_2d(eOM), self.sa_pairs: sa_pairs})

    def partial_fit_policy(self, OM, QOM):
        # policyb = self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        # grads = self.session.run(self.grad_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})
        # for g in grads:
        #     for sg in g:
        #         if len(sg) > 1:
        #             plt.imshow(sg)
        #             plt.colorbar()
        #             plt.show()
        self.session.run(self.train_pi, feed_dict={self.occ_measure_for_Q: np.atleast_2d(QOM), self.occ_measure: np.atleast_2d(OM), self.sa_pairs: sa_pairs})
        # policy = self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        # print('Policy: ' + '\n'.join(str(e) for e in list(zip(sa_pairs_raw.tolist(), (policy-policyb).tolist(), policy.tolist()))))

    def predict_action_prob(self):
        # print('Predict action prob: ', state_action.shape)
        policy =  self.session.run(self.pi, feed_dict={self.sa_pairs: sa_pairs})
        return policy

    def comp_Dw(self):
        # print('Predict action prob: ', state_action.shape)
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa_pairs})

    def get_omega(self):
        w0 = self.session.run(self.w0)
        # w1 = self.session.run(self.w1)
        w2 = self.session.run(self.w2)
        return w0, w2

    def get_theta(self):
        theta0 = self.session.run(self.theta0)
        # theta1 = self.session.run(self.theta1)
        theta2 = self.session.run(self.theta2)
        return theta0, theta2


    def comp_Dw_part(self, sa):
        # print('Predict action prob: ', state_action.shape)
        return self.session.run(self.Dw, feed_dict={self.sa_pairs: sa})

    def comp_log_Dw(self):
        return self.session.run(tf.log(1.0 - self.Dw), feed_dict={self.sa_pairs: sa_pairs})

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


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

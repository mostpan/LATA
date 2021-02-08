from time import  time
import tensorflow as tf
import numpy as np
import evaluate
import networkx as nx
import random
from tfdeterminism import patch
import augmentation
import correction


seed_value = 2020
tf.set_random_seed(seed_value)
patch()

class Model():
    def __init__(self, args):
        self.learning_rate = args.lr
        self.n_node = args.n_node
        self.batch_size = args.batch_size
        self.encoder_sizes = eval(args.encoder_sizes)
        self.decoder_sizes = [self.encoder_sizes[0], self.n_node]
        self.z_size = args.z_size
        self.alpha = args.alpha

        self.regularizer_scale = args.reg

        self.activation = tf.nn.tanh if args.activation=='tanh' else tf.nn.leaky_relu
        self.kernel_initializer = tf.initializers.glorot_uniform

        self.weight={}
        self.init_weight()
        self.build_model()


    def init_weight(self):
        self.weight['encoder_W1']=tf.Variable(tf.initializers.glorot_uniform()((self.n_node,self.encoder_sizes[0])))
        self.weight['encoder_W2']=tf.Variable(tf.initializers.glorot_uniform()((self.encoder_sizes[0],self.encoder_sizes[1])))
        self.weight['decoder_W1']=tf.Variable(tf.initializers.glorot_uniform()((self.encoder_sizes[1],self.decoder_sizes[0])))
        self.weight['decoder_W2']=tf.Variable(tf.initializers.glorot_uniform()((self.decoder_sizes[0],self.decoder_sizes[1])))
        self.weight['projection_W1']=tf.Variable(tf.initializers.glorot_uniform()((self.encoder_sizes[1],self.z_size)))

        self.weight['encoder_b1'] = tf.Variable(tf.initializers.glorot_uniform()((1, self.encoder_sizes[0])))
        self.weight['encoder_b2'] = tf.Variable(tf.initializers.glorot_uniform()((1, self.encoder_sizes[1])))
        self.weight['decoder_b1'] = tf.Variable(tf.initializers.glorot_uniform()((1, self.decoder_sizes[0])))
        self.weight['decoder_b2'] = tf.Variable(tf.initializers.glorot_uniform()((1, self.decoder_sizes[1])))
        self.weight['projection_b1'] = tf.Variable(tf.initializers.glorot_uniform()((1, self.z_size)))

        self.weight['a1'] = tf.Variable(0.0)

    def build_model(self):
        # Input
        self.x_label = tf.placeholder(tf.int32, shape=[None])
        self.x1 = tf.placeholder(tf.float32, shape=[None, self.n_node])
        self.x2 = tf.placeholder(tf.float32, shape=[None, self.n_node])
        self.beta = tf.placeholder(tf.float32)

        # Encoder
        x1=self.x1
        x2=self.x2
        x_label=tf.cast(self.x_label,dtype=tf.float32)
        pos_mask=self.x_label

        regularizer = tf.contrib.layers.l2_regularizer(scale=self.regularizer_scale)

        x1=self.activation(tf.matmul(x1,self.weight['encoder_W1'])+self.weight['encoder_b1'])
        x1=self.activation(tf.matmul(x1,self.weight['encoder_W2'])+self.weight['encoder_b2'])
        z1=self.activation(tf.matmul(x1, self.weight['projection_W1']) + self.weight['projection_b1'])
        x1 = self.activation(tf.matmul(x1, self.weight['decoder_W1']) + self.weight['decoder_b1'])
        x1 = tf.sigmoid(tf.matmul(x1, self.weight['decoder_W2']) + self.weight['decoder_b2'])

        x2 = self.activation(tf.matmul(x2, self.weight['encoder_W1']) + self.weight['encoder_b1'])
        x2 = self.activation(tf.matmul(x2, self.weight['encoder_W2']) + self.weight['encoder_b2'])
        z2 = self.activation(tf.matmul(x2, self.weight['projection_W1']) + self.weight['projection_b1'])
        x2 = self.activation(tf.matmul(x2, self.weight['decoder_W1']) + self.weight['decoder_b1'])
        x2 = tf.sigmoid(tf.matmul(x2, self.weight['decoder_W2']) + self.weight['decoder_b2'])

        # Optimizer
        x_out = tf.sigmoid(self.weight['a1']) * x1 + (1-tf.sigmoid(self.weight['a1'])) * x2
        self.out_A = tf.reshape(x_out, [-1])

        self.mse = tf.reshape(tf.square(x1-x2),[-1])
        self.ce_loss = -(x_label * tf.log(tf.clip_by_value(self.out_A, 1e-10, 1.0)) + (1 - x_label) * tf.log(tf.clip_by_value(1 - self.out_A, 1e-10, 1.0)))
        self.contrast_loss=self.contrastive_loss(tf.concat([z1,z2],axis=0))

        self.loss = (tf.reduce_mean(tf.boolean_mask(self.mse, pos_mask))+self.beta*tf.reduce_mean(tf.boolean_mask(self.ce_loss, pos_mask)) + tf.reduce_mean(tf.boolean_mask(self.ce_loss, 1 - pos_mask))) \
                    +self.contrast_loss*self.alpha \
                    +regularizer(self.weight['encoder_W1'])+regularizer(self.weight['encoder_W2'])+regularizer(self.weight['decoder_W1'])+regularizer(self.weight['decoder_W2'])+regularizer(self.weight['projection_W1'])

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.loss)

    def sim(self,x,y):
        distances = tf.matmul(x, y,transpose_b=True)
        return distances

    def contrastive_loss(self,hidden,
                             hidden_norm=True):
        LARGE_NUM=1e9
        if hidden_norm:
            hidden = tf.math.l2_normalize(hidden, -1)
        hidden1, hidden2 = tf.split(hidden, 2, 0)
        batch_size = tf.shape(hidden1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = self.sim(hidden1, hidden1)
        logits_bb = self.sim(hidden2, hidden2)
        logits_ab = self.sim(hidden1, hidden2)
        logits_ba = self.sim(hidden2, hidden1)

        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = logits_bb - masks * LARGE_NUM

        loss_a = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1))
        loss = (loss_a + loss_b)

        return loss



    def predict(self, g_o,args):
        adj_g_o = nx.adjacency_matrix(g_o).toarray()
        n_node = adj_g_o.shape[0]

        if not args.use_batch:
            feed_dict = {self.x1: adj_g_o, self.x2: adj_g_o}
            out_A = self.sess.run([self.out_A], feed_dict=feed_dict)[0]
        else:
            out_A=np.zeros(shape=(n_node,n_node),dtype=np.float)
            for bsi in range(0, n_node, self.batch_size):
                bei = min(bsi + self.batch_size, n_node)
                adj_x = adj_g_o[bsi:bei, :]
                feed_dict = {self.x1:adj_x,self.x2:adj_x}
                b_out_A = self.sess.run([self.out_A], feed_dict=feed_dict)[0]
                b_out_A=b_out_A.reshape((-1,n_node))
                out_A[bsi:bei,:]=b_out_A
            out_A=out_A.reshape((-1))

        for u, v in list(g_o.edges()):
            p = (out_A[n_node * (u - 1) + (v - 1)] + out_A[n_node * (v - 1) + (u - 1)]) / 2
            g_o.get_edge_data(u, v)['score'] = p


        final_metrics = {}
        evaluate_methods = {'AUC': evaluate.AUC}
        for name, method in evaluate_methods.items():
            ae_metric = method(g_o)
            final_metrics[name] = ae_metric

        return final_metrics

    def train(self, g_o,args):
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            self.sess = sess

            adj_g_o = nx.adjacency_matrix(g_o).toarray()
            x_label= np.zeros(len(adj_g_o) ** 2)
            x_label[adj_g_o.reshape(-1).nonzero()[-1]] = 1
            n_node = len(adj_g_o)
            edges = adj_g_o.nonzero()
            edges = [(edges[0][i], edges[1][i]) for i in range(len(edges[0]))]

            distr_a, distr_b = augmentation.get_E_distr(adj_g_o,augmentation.get_E_A), augmentation.get_E_distr(adj_g_o,augmentation.get_E_B)
            t = time()
            for i in range(1, 1 + args.pre_epochs+args.for_epochs):
                if i == 1 + args.pre_epochs:
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    x_label=correction.correct(n_node,edges,out_loss,x_label,args.cor_fraction)

                if (i-1)%args.verbose==0:
                    x1, x2 = augmentation.augment(adj_g_o,distr_a, distr_b,args.aug_fraction)
                if not args.use_batch:
                    feed_dict = {self.x1: x1, self.x2: x2, self.x_label: x_label,self.beta:max(args.beta,1-(1-args.beta)/args.pre_epochs*i)}
                    _, train_loss,out_loss= sess.run([self.opt_op, self.loss,self.ce_loss], feed_dict=feed_dict)
                else:
                    ids = list(range(n_node))
                    random.shuffle(ids)
                    train_loss,b_n=0,0
                    out_loss=np.zeros_like(x_label,dtype=np.float)
                    for bsi in range(0, n_node, self.batch_size):
                        bei = min(bsi + self.batch_size, n_node)
                        b_ids=np.array(ids[bsi:bei])
                        br_ids=((b_ids*n_node).reshape(-1,1)+np.array(range(n_node)).reshape(1,-1)).reshape(-1)
                        feed_dict = {self.x1: x1[b_ids], self.x2: x2[b_ids], self.x_label: x_label[br_ids],self.beta: max(args.beta, 1 - (1 - args.beta) / args.pre_epochs * i)}
                        _, t_train_loss, t_out_loss = sess.run([self.opt_op, self.loss, self.ce_loss], feed_dict=feed_dict)
                        train_loss+=t_train_loss
                        b_n+=1
                        out_loss[br_ids]=t_out_loss
                    train_loss/=b_n

                if i % args.verbose == 0:
                    t_test_metrics = self.predict(g_o,args)
                    if i<=args.pre_epochs:
                        info_str = "Epoch_A:%04d train_loss=%s test_metrics=%s time=%s" % (i, train_loss, t_test_metrics, (time() - t))
                        print(info_str)
                    else:
                        info_str = "Epoch_B:%04d train_loss=%s test_metrics=%s time=%s" % (i-args.pre_epochs, train_loss, t_test_metrics, (time() - t))
                        print(info_str)
                    t = time()

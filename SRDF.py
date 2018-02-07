# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:39:25 2018
@author: Luo Jiacheng, Hao Wuhan
Social-Regularized Deep Factors Model for Rating Prediction (Baseline)
"""

import tensorflow as tf
import numpy as np
from LoadData import LoadData
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from time import time
import argparse
import math
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm



#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run SRDF.")
    parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors.')
    parser.add_argument('--layers', nargs='?', default='[64,64]', help="Size of each perception layer.")
    parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--social_lambda', type=float, default=0, help='Regularizer for social relationship.')
    parser.add_argument('--keep_prob', nargs='?', default='[0.8,0.5]', help='Keep probability (i.e., 1-dropout_ratio) for each deep layer and the Interaction layer. 1: no dropout. Note that the last index is for the Bi-Interaction layer.')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer', help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--batch_norm', type=int, default=1, help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu', help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--verbose', type=int, default=1, help='Show the results per X epochs (0, 1 ... any positive integer)')
   
    
#    parser.add_argument('--early_stop', type=int, default=1, help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()

class SRDF(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_factor, layers, epoch, learning_rate, social_lambda,
                 keep_prob, optimizer_type, batch_norm, activation_function, verbose,  random_seed=2018):
        # bind params to class
        self.hidden_factor = hidden_factor # embedding size
        self.layers = layers               # perception_layers
#        self.features_U = 1               # Users/friends and items are represented by different embeddings so just only one feature
        self.social_lambda = social_lambda
        self.epoch = epoch
        self.random_seed = random_seed
        self.keep_prob = np.array(keep_prob)
#        self.no_dropout = np.array([1 for i in range(len(keep_prob))])
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.activation_function = activation_function
        self.train_phase =True
#        self.early_stop = early_stop
        # performance of each epoch
        self.train_rmse, self.test_rmse = [],  [] 
        
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_phase = tf.placeholder(tf.bool,name = "train_phase")
            self.lambda_ =  tf.placeholder(tf.float32, name = "social_lambda")
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None], name="dropout")
            self.user = tf.placeholder(tf.int32, shape=[None, 1], name="train_features")  # None * features_M
            self.item = tf.placeholder(tf.int32, shape=[None, 1], name="train_features")  # None * features_M
            self.friend = tf.placeholder(tf.int32, shape=[None, 1], name="train_features")  # None * features_M
            self.sim = tf.placeholder(tf.float32, shape=[None, 1], name="sim")
            self.y_true = tf.placeholder(tf.float32, shape=[None, 1], name="train_ytrue")  # None * 1
            # Variables.
            self.weights = self._initialize_weights()
            
            
    
            # Model.
            # _________ Embedding Layer _____________
            self.u_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user)  # None * 1*embedding_size
            self.i_emb = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item)  # None * 1*embedding_size
            self.f_emb = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.friend)  # None * 1*embedding_size

            self.u_emb = tf.reshape(self.u_emb, shape=[-1, self.hidden_factor])
            self.i_emb = tf.reshape(self.i_emb, shape=[-1, self.hidden_factor])
            self.f_emb = tf.reshape(self.f_emb, shape=[-1, self.hidden_factor])
            
            # _________ Interaction Layer _____________
            
            self.DF = tf.multiply(self.u_emb, self.i_emb)
            if self.batch_norm:
                self.DF = self.batch_norm_layer(self.DF, train_phase=self.train_phase, scope_bn='bn_CSMF')
            
            
            # ________ Perception Layers __________
            for i in range(0, len(self.layers)):
                self.DF = tf.add(tf.matmul(self.DF, self.weights['layer_%d' % i]), self.weights['bias_%d' % i])  # None * layer[i] * 1
                if self.batch_norm:
                    self.DF = self.batch_norm_layer(self.DF, train_phase=self.train_phase, scope_bn='bn_%d' % i)  # None * layer[i] * 1
                self.DF = self.activation_function(self.DF)
                self.DF = tf.nn.dropout(self.DF, self.dropout_keep[i])  # dropout at each Deep layer
            self.out = tf.add(tf.matmul(self.DF, self.weights['prediction']),self.weights['prediction_b'])
            #social regularization
            dev = tf.subtract(self.u_emb, self.f_emb)
            inner_product = tf.reduce_sum(tf.multiply(dev, dev), keep_dims=True)
            # Compute the loss.
            self.loss = tf.square(self.y_true - self.out) + self.lambda_ * inner_product * self.sim
#            print (self.loss)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            # init
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([max(LD.friends) + 1, self.hidden_factor], 0.0, 0.01), name='left_embeddings')  # features_U* K
        all_weights['item_embeddings'] = tf.Variable(
            tf.random_normal([max(LD.item) + 1,  self.hidden_factor], 0.0, 0.01), name='right_embeddings')  # features_I * K

        # deep layers
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)), dtype=np.float32)  # layers[-1] * 1
            all_weights['prediction_b'] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, 1)), dtype=np.float32)
        else:
            all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1
            all_weights['prediction_b'] = tf.Variable(np.ones((1, 1), dtype=np.float32)) 
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user:data[0],self.item:data[1],self.friend:data[2],self.sim:data[3],self.y_true:data[4]}
        feed_dict.update({ self.dropout_keep: self.keep_prob, self.train_phase: True,self.lambda_:self.social_lambda})
        
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss


    def train(self, train_batch, test_batch,Train_data,Test_data):  # fit a dataset
        # Check Init performance
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, test=%.4f [%.1f s]" % (init_train, init_test, time() - t2))

            
        for epoch in range(self.epoch):
            t1 = time()
            for i in range(total_batch):
                # Fit training
                train_data = next(train_batch)
                self.partial_fit(train_data)
            t2 = time()
            
            # output validation
            
            train_result = self.evaluate(Train_data)
            test_result = self.evaluate(Test_data)
            
            self.train_rmse.append(train_result)
            # self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\t train=%.4f, test=%.4f [%.1f s]" 
                       % ( epoch + 1, t2 - t1, train_result, test_result, time() - t2))

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data[-1])
        feed_dict = {self.user:data[0],self.item:data[1],self.friend:data[2],self.sim:data[3],self.y_true:data[4]}
        feed_dict.update({ self.dropout_keep: self.keep_prob, self.train_phase: True,self.lambda_:self.social_lambda})
        predictions = self.sess.run((self.out), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data[-1], (num_example,))
        predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
        predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
        RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
        return RMSE
    
if __name__ == '__main__':
    # Data loading
    LD = LoadData(0.8) #80% train
    batch_size=32
    train_batch = LD.get_batches(LD.train_user,LD.train_item,LD.train_rating,batch_size=batch_size)
    test_batch = LD.get_batches(LD.test_user,LD.test_item,LD.test_rating,batch_size=batch_size)
    
    Train_data = [LD.train_user,LD.train_item,LD.train_friends,LD.train_sim,LD.train_rating]
    Train_data = [np.reshape(x,[-1,1]) for x in Train_data]
    
    Test_data = [LD.test_user,LD.test_item,LD.test_friends,LD.test_sim,LD.test_rating]
    Test_data = [np.reshape(x,[-1,1]) for x in Test_data]
    
    total_batch = int(len(LD.train_user) / batch_size)
    args = parse_args()
   
#    if args.verbose > 0:
#        print("SRDF: hidden_factor=%d, layers=%s, epoch=%d,lr=%.4f,lambda=%.4f,dropout_keep=%s, optimizer=%s, batch_norm=%d, activation=%s" 
#              % (args.hidden_factor,args.layers,  args.epoch, args.lr, args.social_lambda,args.keep_prob, args.optimizer, args.batch_norm, args.activation))
    
#    # Training
#    t1 = time()
#    model = SRDF(args.hidden_factor, eval(args.layers),args.epoch, args.lr, args.social_lambda, eval(args.keep_prob), args.optimizer, args.batch_norm, activation_function, args.verbose)
    activation_function = tf.nn.relu
    model = SRDF(hidden_factor = 10,layers= [128,64],epoch = 12,learning_rate = 0.0001, 
                 social_lambda = 0.5, keep_prob = [1,1],
                 optimizer_type= 'GradientDescentOptimizer',batch_norm =True,
                 activation_function=activation_function,  verbose =1)
    model.train(train_batch, test_batch,Train_data,Test_data)
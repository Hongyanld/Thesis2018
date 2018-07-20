#! /usr/bin/env python3
# Copyright (C) 2018 Robert Werfelmann
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import keras_resnet
import keras_resnet.blocks
import keras_resnet.models
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Embedding, Dropout, TimeDistributed, Dense, Bidirectional, LSTM, Lambda, Concatenate, \
    MaxPooling1D, Conv1D, GlobalMaxPool1D, concatenate, GRU
from keras.optimizers import RMSprop

from main import word_embeddings_size, args, max_num_of_sentences, \
    max_num_of_tokens_per_sentence
from AttentionWithContext import AttentionWithContext





class RNN_Model(object):

    def __init__(self,config,is_training=True):

        self.keep_prob=config.keep_prob
        self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)

        num_step=config.num_step
        self.input_data=tf.placeholder(tf.int32,[None,num_step])
        self.target = tf.placeholder(tf.int64,[None])
        self.mask_x = tf.placeholder(tf.float32,[num_step,None])


        lstm_unit_size = args.lstm_unit_size
        learning_rate = 0.1
        dropout = 0.6



        class_num=config.class_num
        hidden_neural_size=128
        vocabulary_size=config.vocabulary_size
        embed_dim=config.embed_dim
        hidden_layer_num=1
        batch_size = 25
        num_step = 50
        self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
        self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

        #build LSTM network

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
        if self.keep_prob<1:
            lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell,output_keep_prob=self.keep_prob
            )

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*hidden_layer_num,state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size,dtype=tf.float32)

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
            inputs=tf.nn.embedding_lookup(embedding,self.input_data)

        if self.keep_prob<1:
            inputs = tf.nn.dropout(inputs,self.keep_prob)

        out_put=[]
        state=self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step>0: tf.get_variable_scope().reuse_variables()
                (cell_output,state)=cell(inputs[:,time_step,:],state)
                out_put.append(cell_output)

        #之前都跑的通

        out_put=out_put*self.mask_x[:,:,None]

        with tf.name_scope("mean_pooling_layer"):

            out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

        with tf.name_scope("Softmax_layer_and_output"):
            softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,class_num],dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32)
            self.logits = tf.matmul(out_put,softmax_w)+softmax_b

        with tf.name_scope("loss"):
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits+1e-10,self.target)
            self.cost = tf.reduce_mean(self.loss)

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(self.logits,1)
            correct_prediction = tf.equal(self.prediction,self.target)
            self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

        #add summary
        loss_summary = tf.scalar_summary("loss",self.cost)
        #add summary
        accuracy_summary=tf.scalar_summary("accuracy_summary",self.accuracy)

        if not is_training:
            return

        self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
        self.lr = tf.Variable(0.0,trainable=False)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                      config.max_grad_norm)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.merge_summary(grad_summaries)

        self.summary =tf.merge_summary([loss_summary,accuracy_summary,self.grad_summaries_merged])



        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))
        self.train_op=optimizer.apply_gradients(zip(grads, tvars))

        self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr,self.new_lr)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})


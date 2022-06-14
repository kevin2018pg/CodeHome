# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 20:53
# @Author  : west
# @File    : mmoe.py
# @Version : python 3.6
# @Desc    : mmoe + ple

import tensorflow as tf
from deepctr.feature_column import build_input_features, input_from_feature_columns
from deepctr.layers.core import PredictionLayer, DNN
from deepctr.layers.utils import combined_dnn_input, concat_func
# from tensorflow.python.keras.initializers import glorot_normal
from tensorflow.keras.initializers import glorot_normal
# from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Layer


class MMOELayer(Layer):
    """
    The Multi-gate Mixture-of-Experts layer in MMOE model
      Input shape
        - 2D tensor with shape: ``(batch_size,units)``.
      Output shape
        - A list with **num_tasks** elements, which is a 2D tensor with shape: ``(batch_size, output_dim)`` .
      Arguments
        - **num_tasks**: integer, the number of tasks, equal to the number of outputs.
        - **num_experts**: integer, the number of experts.
        - **output_dim**: integer, the dimension of each output of MMOELayer.
    References
      - [Jiaqi Ma, Zhe Zhao, Xinyang Yi, et al. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C]](https://dl.acm.org/doi/10.1145/3219819.3220007)
    """

    def __init__(self, num_tasks, num_experts, output_dim, seed=1024, **kwargs):
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.output_dim = output_dim
        self.seed = seed
        super(MMOELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.expert_kernel = self.add_weight(name='expert_kernel',
                                             shape=(input_dim, self.num_experts * self.output_dim), dtype=tf.float32,
                                             initializer=glorot_normal(seed=self.seed))
        self.gate_kernels = []
        for i in range(self.num_tasks):
            self.gate_kernels.append(self.add_weight(name='gate_weight_'.format(i),
                                                     shape=(input_dim, self.num_experts),
                                                     dtype=tf.float32,
                                                     initializer=glorot_normal(seed=self.seed)))
        super(MMOELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = []
        # tensordot:https://zhuanlan.zhihu.com/p/385099839
        expert_out = tf.tensordot(inputs, self.expert_kernel, axes=(-1, 0))
        expert_out = tf.reshape(expert_out, [-1, self.output_dim, self.num_experts])
        for i in range(self.num_tasks):
            gate_out = tf.tensordot(inputs, self.gate_kernels[i], axes=(-1, 0))
            gate_out = tf.nn.softmax(gate_out)
            gate_out = tf.tile(tf.expand_dims(gate_out, axis=1), [1, self.output_dim, 1])
            output = tf.reduce_sum(tf.multiply(expert_out, gate_out), axis=2)
            outputs.append(output)
        return outputs

    def get_config(self):
        config = {'num_tasks': self.num_tasks,
                  'num_experts': self.num_experts,
                  'output_dim': self.output_dim}
        base_config = super(MMOELayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.output_dim] * self.num_tasks


def MMOE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
         l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))

    """
    fibinet
    """

    """
    deepFM
    """

    """
    Origin
    """
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)
    mmoe_outs = MMOELayer(num_tasks, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for
                     mmoe_out in mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model


class PLELayer(Layer):
    def __init__(self, num_tasks, num_level, experts_num, experts_units, seed=1024, **kwargs):
        self.num_tasks = num_tasks
        self.num_level = num_level
        self.experts_num = experts_num
        self.experts_units = experts_units
        self.selector_num = 2
        self.seed = seed
        super(PLELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self.experts_weight_share = [
            self.add_weight(
                name='experts_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='experts_weight_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed))
        ]
        self.experts_bias_share = [
            self.add_weight(
                name='expert_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)),
            self.add_weight(
                name='expert_bias_share_2',
                dtype=tf.float32,
                shape=(self.experts_units, self.experts_num),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_weight_share = [
            self.add_weight(
                name='gate_weight_share_1',
                dtype=tf.float32,
                shape=(input_dim, self.experts_num * (self.num_tasks + 1)),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.gate_bias_share = [
            self.add_weight(
                name='gate_bias_share_1',
                dtype=tf.float32,
                shape=(self.experts_num * (self.num_tasks + 1),),
                initializer=glorot_normal(seed=self.seed)
            )
        ]
        self.experts_weight = [[], []]
        self.experts_bias = [[], []]
        self.gate_weight = [[], []]
        self.gate_bias = [[], []]

        for i in range(self.num_level):
            if 1 == i:
                input_dim = self.experts_units

            for j in range(self.num_tasks):
                # experts Task j
                self.experts_weight[i].append(self.add_weight(
                    name='experts_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim, self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.experts_bias[i].append(self.add_weight(
                    name='expert_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_units, self.experts_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                # gates Task j
                self.gate_weight[i].append(self.add_weight(
                    name='gate_weight_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(input_dim, self.experts_num * self.selector_num),
                    initializer=glorot_normal(seed=self.seed)
                ))
                self.gate_bias[i].append(self.add_weight(
                    name='gate_bias_task{}_{}'.format(j, i),
                    dtype=tf.float32,
                    shape=(self.experts_num * self.selector_num,),
                    initializer=glorot_normal(seed=self.seed)
                ))
        super(PLELayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gate_output_task_final = [inputs, inputs, inputs, inputs, inputs, inputs, inputs, ]
        gate_output_share_final = inputs
        for i in range(self.num_level):
            # experts shared outputs
            experts_output_share = tf.tensordot(gate_output_share_final, self.experts_weight_share[i], axes=1)
            experts_output_share = tf.add(experts_output_share, self.experts_bias_share[i])
            experts_output_share = tf.nn.relu(experts_output_share)
            experts_output_task_tmp = []
            for j in range(self.num_tasks):
                experts_output_task = tf.tensordot(gate_output_task_final[j], self.experts_weight[i][j], axes=1)
                experts_output_task = tf.add(experts_output_task, self.experts_bias[i][j])
                experts_output_task = tf.nn.relu(experts_output_task)
                experts_output_task_tmp.append(experts_output_task)
                gate_output_task = tf.matmul(gate_output_task_final[j], self.gate_weight[i][j])
                gate_output_task = tf.add(gate_output_task, self.gate_bias[i][j])
                gate_output_task = tf.nn.softmax(gate_output_task)
                gate_output_task = tf.multiply(concat_func([experts_output_task, experts_output_share], axis=2),
                                               tf.expand_dims(gate_output_task, axis=1))
                gate_output_task = tf.reduce_sum(gate_output_task, axis=2)
                gate_output_task = tf.reshape(gate_output_task, [-1, self.experts_units])
                gate_output_task_final[j] = gate_output_task

            if 0 == i:
                # gates shared outputs
                gate_output_shared = tf.matmul(gate_output_share_final, self.gate_weight_share[i])
                gate_output_shared = tf.add(gate_output_shared, self.gate_bias_share[i])
                gate_output_shared = tf.nn.softmax(gate_output_shared)
                gate_output_shared = tf.multiply(concat_func(experts_output_task_tmp + [experts_output_share], axis=2),
                                                 tf.expand_dims(gate_output_shared, axis=1))
                gate_output_shared = tf.reduce_sum(gate_output_shared, axis=2)
                gate_output_shared = tf.reshape(gate_output_shared, [-1, self.experts_units])
                gate_output_share_final = gate_output_shared
        return gate_output_task_final


def PLE(dnn_feature_columns, num_tasks, tasks, num_experts=4, expert_dim=8, dnn_hidden_units=(128, 128),
        l2_reg_embedding=1e-5, l2_reg_dnn=0, task_dnn_units=None, seed=1024, dnn_dropout=0, dnn_activation='relu'):
    if num_tasks <= 1:
        raise ValueError("num_tasks must be greater than 1")
    if len(tasks) != num_tasks:
        raise ValueError("num_tasks must be equal to the length of tasks")
    for task in tasks:
        if task not in ['binary', 'regression']:
            raise ValueError("task must be binary or regression, {} is illegal".format(task))
    features = build_input_features(dnn_feature_columns)
    inputs_list = list(features.values())
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)
    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_out = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(dnn_input)

    mmoe_outs = PLELayer(num_tasks, 2, num_experts, expert_dim)(dnn_out)
    if task_dnn_units != None:
        mmoe_outs = [DNN(task_dnn_units, dnn_activation, l2_reg_dnn, dnn_dropout, False, seed=seed)(mmoe_out) for
                     mmoe_out in mmoe_outs]

    task_outputs = []
    for mmoe_out, task in zip(mmoe_outs, tasks):
        logit = tf.keras.layers.Dense(1, use_bias=False, activation=None)(mmoe_out)
        output = PredictionLayer(task)(logit)
        task_outputs.append(output)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=task_outputs)
    return model

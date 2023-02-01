import tensorflow as tf
import numpy as np


class RuleGraphConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 out_channel,
                 num_features=81,
                 num_bond=22,
                 activation_fn=None,
                 combination_rules=[]
                 ):
        super(RuleGraphConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.num_bond = num_bond
        self.activation_fn = activation_fn
        self.combination_rules = combination_rules
        self.w_s = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features, out_channel]), shape=[num_features, out_channel], trainable=True, name='w_s')
        self.w_n = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features + num_bond, out_channel]),
                               shape=[num_features + num_bond, out_channel], trainable=True, name='w_n')

    def AtomDistance(self, x, y):
        return tf.sqrt(tf.reduce_sum(tf.square(x - y)))

    def addRule(self, rule, start_index, end_index=None):
        rules_dict = {
            "sum": tf.math.add,
            "multiply": tf.math.multiply,
            "distance": "distance",
            "divide": tf.math.divide,
            "subtract": tf.math.subtract,
        }
        if type(rule) == str:
            rule = rules_dict[rule]
        if end_index == None:
            self.combination_rules.append([[start_index], rule])
        else:
            self.combination_rules.append([[start_index, end_index], rule])

    def _call_single(self, inp):
        features = inp

        # self.counter1.assign(0)
        # self.self_conv_features.assign(np.array([0.0 for _ in range(self.out_channel)]).reshape([1, self.out_channel]))
        def self_weight_mul():  # i, self_conv_features):
            i = tf.constant(0)
            self_conv_features = tf.TensorArray(size=self.out_channel, dynamic_size=True, dtype=tf.float32)
            # condition
            c = lambda i, self_conv_features: tf.less(i, features.shape[0])

            # body
            def b(i, self_conv_features):
                self_conv_features = self_conv_features.write(i, tf.matmul(
                    tf.reshape(features[i][:self.num_features], [1, self.num_features]),
                    self.w_s
                ))
                return i + 1, self_conv_features

            return tf.while_loop(c, b, loop_vars=[i, self_conv_features])
            # shape_invariants=[self.counter1.get_shape(), tf.TensorShape([None, self.out_channel])])
        # self.self_conv_features = self.self_conv_features[1:]
        new_features = tf.reshape(self_weight_mul()[1].stack(), (features.get_shape()[0], self.out_channel))

        ## Neighbours part
        def neighbour_weight_mul():
            i = tf.constant(0)
            nei_conv_features = tf.TensorArray(size=features.get_shape()[0], dynamic_size=True, dtype=tf.float32)

            c = lambda i, nei_conv_features: tf.less(i, features.get_shape()[0])

            def b(i, nei_conv_features):
                new_ordered_features = tf.TensorArray(size=features.get_shape()[0], dynamic_size=True, dtype=tf.float32,
                                                      clear_after_read=False, infer_shape=False)
                distance = -1.
                # for v in [[adjacency_list[i][0], adjacency_list[i][1:23]], [adjacency_list[i][23], adjacency_list[i][24:]]]:
                self_features = tf.reshape(features[i][:self.num_features], [1, self.num_features])
                for v in range(2):
                    if tf.cast(features[i][self.num_features + v], tf.int32) == 0:
                        continue
                    for j, rule in enumerate(self.combination_rules):
                        if j == len(self.combination_rules) - 1 and len(rule[0]) == 1:
                            new_ordered_features.write(j, rule[1](self_features[0][rule[0][0]:],
                                                                  features[
                                                                      tf.cast(features[i][self.num_features + v], tf.int32)][
                                                                  rule[0][0]:]))
                        else:
                            if rule[1] == 'distance':
                                distance = self.AtomDistance(x=self_features[0][rule[0][0]:rule[0][1]],
                                                             y=features[tf.cast(features[i][self.num_features + v], tf.int32)][
                                                               rule[0][0]:rule[0][1]])
                                new_ordered_features.write(j, features[tf.cast(features[i][self.num_features + v], tf.int32)][
                                                              rule[0][0]:rule[0][1]])
                            else:
                                # print(tf.cast(adjacency_list[i][v*23], tf.int32))
                                new_ordered_features.write(j, rule[1](self_features[0][rule[0][0]:rule[0][1]],
                                                                      features[
                                                                          tf.cast(features[i][self.num_features + v], tf.int32)][
                                                                      rule[0][0]:rule[0][1]]))
                    new_ordered_features_tensor = tf.concat(
                        [new_ordered_features.read(k) for k in range(len(self.combination_rules))], axis=0)
                    if distance != -1.:
                        distance = distance if distance > 0 else 10e-3
                        new_ordered_features_tensor /= distance ** 2
                    nei_conv_features.write(i,
                                            tf.add(
                                                tf.matmul(
                                                    tf.reshape(
                                                        new_ordered_features_tensor, [1, self.num_features]),
                                                    self.w_n),
                                                new_features[i])
                                            )

                return i + 1, nei_conv_features

            return tf.while_loop(c, b, loop_vars=[i, nei_conv_features])

        neighbor_conv_features = neighbour_weight_mul()[1]
        # self.self_conv_features = self.self_conv_features[1:]
        neighbor_conv_features = tf.reshape(neighbor_conv_features.stack(), (features.get_shape()[0], self.out_channel))
        return neighbor_conv_features

    def call(self, inputs):
        output = []
        for inp in inputs:
            ans = self._call_single(inp)
            output.append(ans)
        return output


class ConvLayer(tf.keras.Model):
    def __init__(self, out_channel, num_features=20):
        super(ConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.w = tf.Variable(tf.initializers.glorot_uniform()
                             (shape=[num_features, out_channel]), shape=[num_features, out_channel], trainable=True, name='w_cl')

    def _call_single(self, inp):
        out = tf.zeros(shape=[1, self.out_channel], dtype=tf.float32)
        for feature in inp:
            feature = tf.reshape(feature, [1, -1])
            feature = tf.cast(feature, 'float')
            out += tf.reshape(tf.nn.sigmoid(tf.matmul(feature, self.w)), [-1])
        return tf.reshape(out, [-1])

    def call(self, inputs):
        output = []
        for inp in inputs:
            output.append(self._call_single(inp))
        return tf.reshape(output, [len(inputs), -1])  # np.array(output).reshape([len(inputs), -1])


class GraphConvLayer(tf.keras.layers.Layer):
    def __init__(self,
                 out_channel,
                 num_features=80,
                 activation_fn=tf.keras.activations.relu,
                 ):
        super(GraphConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.activation_fn = activation_fn
        self.w_s = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features, out_channel]), shape=[num_features, out_channel], trainable=True)
        self.w_n = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features, out_channel]),
                               shape=[num_features, out_channel], trainable=True)

    def _call_single(self, inp):
        features = inp[0]
        adjacency_list = inp[1]
        new_features = features.shape[0] * [None]
        for i, adj in enumerate(adjacency_list):
            self_features = tf.reshape(features[i], [1, self.num_features])
            self_conv_features = tf.matmul(self_features, self.w_s)
            new_features[i] = self_conv_features
            for neighbour in adj:
                neighbour_features = features[neighbour[0]]
                neighbour_bond = neighbour[1]
                neighbour_features = tf.reshape(neighbour_features, [1, self.num_features])
                new_features[i] += tf.matmul(neighbour_features, self.w_n)
                new_features[i] = new_features[i][0]
                if self.activation_fn != None:
                    new_features[i] = self.activation_fn(new_features[i])
        return ([tf.Variable(new_features, trainable=False), adjacency_list])

    def call(self, inputs):
        output = []
        for inp in inputs:
            output.append(self._call_single(inp))
        return output

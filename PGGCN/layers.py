import tensorflow as tf

class RuleGraphConvLayer(tf.keras.Model):
    def __init__(self,
                 out_channel,
                 num_features=81,
                 num_bond=22,
                 activation_fn=None,
                 combination_rules = []
                 ):
        super(RuleGraphConvLayer, self).__init__()
        self.out_channel = out_channel
        self.num_features = num_features
        self.num_bond = num_bond
        self.activation_fn = activation_fn
        self.combination_rules = combination_rules
        self.w_s = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features, out_channel]), shape=[num_features, out_channel], trainable=True)
        self.w_n = tf.Variable(initial_value=tf.initializers.glorot_uniform()
        (shape=[num_features + num_bond, out_channel]),
                               shape=[num_features + num_bond, out_channel], trainable=True)

    def AtomDistance(self, x, y):
        return tf.sqrt(tf.reduce_sum(tf.square(x - y)))

    def addRule(self, rule, start_index, end_index = None):
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
        features = inp[0]
        adjacency_list = inp[1]
        new_features = len(features) * [None]
        for i, adj in enumerate(adjacency_list):
            self_features = tf.reshape(features[i], [1, self.num_features])
            self_conv_features = tf.matmul(self_features, self.w_s)
            new_features[i] = self_conv_features
            for neighbour in adj:
                neighbour_features = features[neighbour[0]]
                neighbour_bond = neighbour[1]
                new_ordered_features = []
                distance = None
                for j, rule in enumerate(self.combination_rules):
                    rule_function = rule[1]
                    indices = rule[0]
                    if j == len(self.combination_rules) - 1 and len(indices) == 1:
                        new_ordered_features.append(rule_function(self_features[0][indices[0]:],
                                                                  neighbour_features[indices[0]:]))
                    else:
                        if rule_function == 'distance':
                            distance = self.AtomDistance(x=self_features[0][indices[0]:indices[1]],
                                                    y=neighbour_features[indices[0]:indices[1]])
                            new_ordered_features.append(neighbour_features[indices[0]:indices[1]])
                        else:
                            new_ordered_features.append(rule_function(self_features[0][indices[0]:indices[1]],
                                                                      neighbour_features[indices[0]:indices[1]]))
                new_ordered_features = tf.concat(new_ordered_features, axis=0)
                if distance != None:
                    new_ordered_features /= distance ** 2
                new_ordered_features = tf.concat([new_ordered_features, neighbour_bond], axis=0)
                new_ordered_features = tf.reshape(new_ordered_features, [1, self.num_features + self.num_bond])
                new_features[i] += tf.matmul(new_ordered_features, self.w_n)
                new_features[i] = new_features[i][0]
        return ([tf.Variable(new_features, trainable=False), adjacency_list, inp[2]])

    def call(self, inputs):
        output = []
        for inp in inputs:
            output.append(self._call_single(inp))
        return output


class GraphConvLayer(tf.keras.Model):
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
        return ([tf.Variable(new_features, trainable=False), adjacency_list, inp[2]])

    def call(self, inputs):
        output = []
        for inp in inputs:
            output.append(self._call_single(inp))
        return output
import tensorflow as tf

class NBTModel(object):
    """
    Decoding module to compute state and translation similarity
    """
    def __init__(self, **kwargs):
        # Slot-tracking model or Translation model
        self.slot_name = kwargs['slot_name']
        if " " in self.slot_name:
            self.slot_name = "_".join(self.slot_name.split(' '))
        with tf.variable_scope(self.slot_name, reuse=tf.AUTO_REUSE):
            #if parallel == "none":
            self.hidden_units_1 = 100
            self.longest_utterance_length = 40
            self.vector_dimension = kwargs['vector_dimension']
            self.label_count = kwargs['label_count']
            self.use_softmax = kwargs['use_softmax']
            self.keep_prob = tf.placeholder("float", name="keep_prob")
            self.primary = kwargs['primary']
            self.secondary = kwargs['secondary']
            self.utterance_representations_full = tf.placeholder(tf.float32, [None, 40, self.vector_dimension], name="utterance")
            self.utterance_representations_full_foreign = tf.placeholder(tf.float32, [None, 40, self.vector_dimension], name="utterance_foreign")
            self.phase = tf.placeholder(tf.bool, name='phase')

            utterance_representation = self.encode(self.utterance_representations_full, self.primary)
            utterance_representation_foreign = self.encode(self.utterance_representations_full_foreign, self.secondary)
            self.distance_utt = 0.5 * tf.reduce_sum(tf.square(utterance_representation - utterance_representation_foreign))
            print("=========================== Model declaration ===========================")

            if self.use_softmax:
                self.label_size = self.label_count + 1 # 1 is for NONE, dontcare is added to the ontology.
            else:
                self.label_size = self.label_count
            # these zre actual NN hyperparameters that we might want to tune at some point:
            print("Hidden layer size:", self.hidden_units_1, "Label Size:", self.label_size, "Use Softmax:", self.use_softmax)

            self.system_act_slots = tf.placeholder(tf.float32, shape=(None, self.vector_dimension))  # just slots, for requestables.
            self.system_act_confirm_slots = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name="system_act_confirm_slots")
            self.system_act_confirm_values = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='system_act_confirm_values')

            self.system_act_slots_foreign = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='system_act_slots_foreign')
            self.system_act_confirm_slots_foreign = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='system_act_confirm_slots_foreign')
            self.system_act_confirm_values_foreign = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='system_act_confirm_values_foreign')

            # Initial (distributional) vectors. Needed for L2 regularisation.
            self.W_slots = tf.placeholder(tf.float32, shape=(self.vector_dimension, ), name='W_slots')
            self.W_values = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='W_values')
            self.W_slots_foreign = tf.placeholder(tf.float32, shape=(self.vector_dimension, ), name='W_slots_foreign')
            self.W_values_foreign = tf.placeholder(tf.float32, shape=(None, self.vector_dimension), name='W_values_foreign')

            # output label, i.e. True / False, 1-hot encoded:
            self.y_ = tf.placeholder(tf.float32, [None, self.label_size], name='y_')
            self.alpha = kwargs.get('alpha', 1.0)
            self.y_past_state = tf.placeholder(tf.float32, [None, self.label_size], name='y_past_state')

            # dropout placeholder, 0.5 for training, 1.0 for validation/testing:
            gate = self.gating(self.primary, self.W_values, self.W_slots, self.system_act_slots, \
                               self.system_act_confirm_slots, self.system_act_confirm_values)
            gate_foreign = self.gating(self.secondary, self.W_values_foreign, self.W_slots_foreign, \
                                       self.system_act_slots_foreign, self.system_act_confirm_slots_foreign, \
                                       self.system_act_confirm_values_foreign)
            self.distance_gate = 0.5 * tf.reduce_sum(tf.square(gate - gate_foreign))
            self.transfer_loss = self.alpha * self.distance_gate + self.distance_utt

            self.y, self.cross_entropy = self.decode(utterance_representation, gate)
            self.y_foreign, _ = self.decode(utterance_representation_foreign, gate_foreign)

            self.predictions = tf.argmax(self.y, 1)  # will have ones where positive
            self.true_predictions = tf.argmax(self.y_, 1)
            correct_prediction = tf.cast(tf.equal(self.predictions, self.true_predictions), "float")
            self.accuracy = tf.reduce_mean(correct_prediction)

        with tf.variable_scope('optimization', reuse=tf.AUTO_REUSE):
            if self.use_softmax:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/{}-gate'.format(self.slot_name, self.secondary))
            var_list += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='{}/{}-encoder'.format(self.slot_name, self.secondary))
            self.train_transfer_step = self.optimizer.minimize(self.transfer_loss, var_list=var_list)
            self.train_step = self.optimizer.minimize(self.cross_entropy)

    def gating(self, language, W_values, W_slots, system_act_slots, system_act_confirm_slots, system_act_confirm_values):
        with tf.variable_scope('{}-gate'.format(language), reuse=tf.AUTO_REUSE):
            w_candidates = tf.get_variable(name="W-trans-cand", initializer=tf.random_normal([self.vector_dimension, self.vector_dimension]))
            b_candidates = tf.get_variable(name="b-trans-cand", initializer=tf.zeros([self.vector_dimension]))
            # multiply to get: [label_size, vector_dimension]
            candidates_transform = tf.nn.sigmoid(tf.matmul(W_values, w_candidates) + b_candidates)
            gate1 = candidates_transform[:self.label_count]
            gate1 = tf.expand_dims(gate1, 0)

            w_system_act_slot = tf.get_variable(name="W-system-act", initializer=tf.random_normal([self.vector_dimension, self.vector_dimension]))
            b_system_act_slot = tf.get_variable(name="b-system-act", initializer=tf.zeros([self.vector_dimension]))
            system_act_slots = tf.nn.sigmoid(tf.matmul(system_act_slots, w_system_act_slot) + b_system_act_slot)
            system_act_candidate_interaction = tf.multiply(W_slots, system_act_slots)
            gate2 = tf.reduce_mean(system_act_candidate_interaction, 1)
            gate2 = tf.expand_dims(tf.expand_dims(gate2, 1), 2)

            w_system_confirm_slot = tf.get_variable(name="W-system-confirm-slot", initializer=tf.random_normal([self.vector_dimension, self.vector_dimension]))
            b_system_confirm_slot = tf.get_variable(name="b-system-confirm-slot", initializer=tf.zeros([self.vector_dimension]))
            w_system_confirm_value = tf.get_variable(name="W-system-confirm-value", initializer=tf.random_normal([self.vector_dimension, self.vector_dimension]))
            b_system_confirm_value = tf.get_variable(name="b-system-confirm-value", initializer=tf.zeros([self.vector_dimension]))
            system_act_confirm_slots = tf.nn.sigmoid(tf.matmul(system_act_confirm_slots, w_system_confirm_slot) + b_system_confirm_slot)
            system_act_confirm_values = tf.nn.sigmoid(tf.matmul(system_act_confirm_values, w_system_confirm_value) + b_system_confirm_value)
            gate3 = []
            for value_idx in range(0, self.label_count):
                dot_product = tf.multiply(tf.reduce_mean(tf.multiply(W_slots, system_act_confirm_slots), 1), \
                                          tf.reduce_mean(tf.multiply(W_values[value_idx, :], system_act_confirm_values), 1))
                gate3.append(tf.cast(tf.equal(dot_product, tf.ones(tf.shape(dot_product))), "float32"))
            gate3 = tf.stack(gate3, 1)
            gate3 = tf.expand_dims(gate3, 2)

            gate = gate1 + gate2 + gate3
        if self.use_softmax:
            return gate
        else:
            return gate1

    def encode(self, utterance_representations_full, language):
        with tf.variable_scope('{}-encoder'.format(language), reuse=tf.AUTO_REUSE):
            filter_sizes = [1, 2, 3]
            num_filters = 300
            h_utterance_representation = tf.zeros([num_filters], tf.float32)
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                # Convolution Layer
                filter_shape = [filter_size, self.vector_dimension, 1, num_filters]
                W = tf.get_variable(name="W-{}".format(i), initializer=tf.truncated_normal(filter_shape, stddev=0.1))
                b = tf.get_variable(name="b-{}".format(i), initializer=tf.constant(0.1, shape=[num_filters]))
                conv = tf.nn.conv2d(
                    tf.expand_dims(utterance_representations_full, -1),
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.longest_utterance_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID')
                pooled_outputs.append(pooled)
                h_utterance_representation += tf.reshape(tf.concat(pooled, 3), [-1, num_filters])
            #h_utterance_representation = tf.contrib.layers.batch_norm(h_utterance_representation, center=False, scale=False, is_training=self.phase, scope='bn')
            h_utterance_representation = tf.nn.l2_normalize(h_utterance_representation, dim=1)
        return h_utterance_representation

    def decode(self, h_utterance_representation, gate):
        h = tf.multiply(gate, tf.expand_dims(h_utterance_representation, 1))
        h = tf.reshape(h, [-1, self.vector_dimension])
        h = tf.nn.dropout(h, self.keep_prob)
        with tf.variable_scope("output_model", reuse=tf.AUTO_REUSE):
            w2_softmax = tf.get_variable(name="W2-projection", initializer=tf.random_normal([self.vector_dimension, 1]))
            b2_softmax = tf.get_variable(name="b2-projection", initializer=tf.zeros([1]))
            if self.use_softmax:
                #h = tf.sigmoid(tf.matmul(h, w1_softmax) + b1_softmax)
                y_presoftmax = tf.matmul(h, w2_softmax) + b2_softmax
                y_presoftmax = tf.reshape(y_presoftmax, [-1, self.label_count])
                append_zeros_none = tf.zeros([tf.shape(y_presoftmax)[0], 1])
                y_presoftmax = tf.concat([y_presoftmax, append_zeros_none], 1)
            else:
                y_presoftmax = tf.matmul(h, w2_softmax) + b2_softmax
                y_presoftmax = tf.reshape(y_presoftmax, [-1, self.label_count])

        with tf.variable_scope('distribution_output', reuse=tf.AUTO_REUSE):
            if self.use_softmax:
                a_memory = tf.get_variable(name="a-memory", initializer=tf.random_normal([1, 1]))
                diag_memory = a_memory * tf.diag(tf.ones(self.label_size))
                b_memory = tf.get_variable(name="b-memory", initializer=tf.random_normal([1, 1]))
                non_diag_memory = tf.matrix_set_diag(b_memory * tf.ones([self.label_size, self.label_size]), tf.zeros(self.label_size))
                W_memory = diag_memory + non_diag_memory
                a_current = tf.get_variable(name="a-current", initializer=tf.random_normal([1, 1]))
                diag_current = a_current * tf.diag(tf.ones(self.label_size))
                b_current = tf.get_variable(name="b-current", initializer=tf.random_normal([1, 1]))
                non_diag_current = tf.matrix_set_diag(b_current * tf.ones([self.label_size, self.label_size]), tf.zeros(self.label_size))
                W_current = diag_current + non_diag_current
                y_combine = tf.matmul(self.y_past_state, W_memory) + tf.matmul(y_presoftmax, W_current)
                y = tf.nn.softmax(y_combine)  # + y_ss_update_contrib)
                cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_combine, labels=self.y_))
            else:
                y = tf.nn.sigmoid(y_presoftmax)  # for requestables, we just have turn-level binary decisions
                cross_entropy = tf.reduce_sum(tf.square(y - self.y_))
        return y, cross_entropy

    def train(self, sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys, batch_ys_prev,
              slot_vector, value_vector):
        [_, loss] = sess.run([self.train_step, self.cross_entropy],
                            feed_dict={self.utterance_representations_full: batch_xs_full, \
                                    self.system_act_slots: batch_sys_req, \
                                    self.system_act_confirm_slots: batch_sys_conf_slots, \
                                    self.system_act_confirm_values: batch_sys_conf_values, \
                                    self.W_slots: slot_vector, \
                                    self.W_values: value_vector, \
                                    self.y_: batch_ys, self.y_past_state: batch_ys_prev,
                                    self.phase: 1, \
                                    self.keep_prob: 0.5})
        return loss

    def train_transfer(self, sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values,
              slot_vector, value_vector, batch_xs_full_foreign, batch_sys_req_foreign, batch_sys_conf_slots_foreign,
              batch_sys_conf_values_foreign, slot_vector_foreign, value_vector_foreign):
        [_, loss] = sess.run([self.train_transfer_step, self.transfer_loss],
                             feed_dict={self.utterance_representations_full: batch_xs_full, \
                                        self.system_act_slots: batch_sys_req, \
                                        self.system_act_confirm_slots: batch_sys_conf_slots, \
                                        self.system_act_confirm_values: batch_sys_conf_values, \
                                        self.W_slots: slot_vector, \
                                        self.W_values: value_vector, \
                                        self.utterance_representations_full_foreign: batch_xs_full_foreign, \
                                        self.system_act_slots_foreign: batch_sys_req_foreign, \
                                        self.system_act_confirm_slots_foreign: batch_sys_conf_slots_foreign, \
                                        self.system_act_confirm_values_foreign: batch_sys_conf_values_foreign, \
                                        self.W_slots_foreign: slot_vector_foreign, \
                                        self.W_values_foreign: value_vector_foreign, \
                                        self.phase: 1, \
                                        self.keep_prob: 0.5})
        return loss

    def eval(self, sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys, batch_ys_prev,
             slot_vector, value_vector):
        [current_predictions, groundtruh, current_accuracy] = sess.run([self.predictions, self.true_predictions, self.accuracy],
                                                                       feed_dict={self.utterance_representations_full: batch_xs_full, \
                                                                                self.system_act_slots: batch_sys_req, \
                                                                                self.system_act_confirm_slots: batch_sys_conf_slots, \
                                                                                self.system_act_confirm_values: batch_sys_conf_values, \
                                                                                self.W_slots: slot_vector, \
                                                                                self.W_values: value_vector, \
                                                                                self.y_: batch_ys, self.y_past_state: batch_ys_prev, \
                                                                                self.phase: 0, \
                                                                                self.keep_prob: 1.0})
        return current_predictions, groundtruh, current_accuracy

    def predict(self, sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys_prev,
                slot_vector, value_vector):
        [current_y] = sess.run([self.y],
                               feed_dict={self.utterance_representations_full: batch_xs_full,
                                       self.W_slots: slot_vector, \
                                       self.W_values: value_vector, \
                                       self.system_act_slots: batch_sys_req,
                                       self.system_act_confirm_slots: batch_sys_conf_slots, \
                                       self.system_act_confirm_values: batch_sys_conf_values, \
                                       self.y_past_state: batch_ys_prev, \
                                       self.phase: 0, \
                                       self.keep_prob: 1.0})
        return current_y

    def predict_foreign(self, sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys_prev,
                        slot_vector, value_vector):
        [current_y] = sess.run([self.y_foreign],
                               feed_dict={self.utterance_representations_full_foreign: batch_xs_full,
                                       self.W_slots_foreign: slot_vector, \
                                       self.W_values_foreign: value_vector, \
                                       self.system_act_slots_foreign: batch_sys_req,
                                       self.system_act_confirm_slots_foreign: batch_sys_conf_slots, \
                                       self.system_act_confirm_values_foreign: batch_sys_conf_values, \
                                       self.y_past_state: batch_ys_prev, \
                                       self.phase: 0, \
                                       self.keep_prob: 1.0})
        return current_y
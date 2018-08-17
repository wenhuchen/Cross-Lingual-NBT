# -*- coding: utf-8 -*-

import string
import ConfigParser
import types
import json
from modelv2 import NBTModel
import codecs
import numpy
import os
from utils import load_woz_data
import tensorflow as tf
from utils import *
import time
import pickle
import random

class BeliefTracker(object):
    """
    Parent Belief Tracking class
    """
    def __init__(self, config_filepath):
        config = ConfigParser.RawConfigParser()
        self.config = config
        try:
            config.read(config_filepath)
        except:
            print("Couldn't read config file from", config_filepath, "... aborting.")
            return
        self.lp = {"english": u"en", "german": u"de", "italian": u"it"}
        self.dataset_name = config.get("model", "dataset_name")
        # Use value specific decoder
        value_specific_decoder = self.config.get("model", "value_specific_decoder")
        if value_specific_decoder in ["True", "true"]:
            self.value_specific_decoder = True
        else:
            self.value_specific_decoder = False
        print("value_specific_decoder", self.value_specific_decoder)
        # use belief state updating
        learn_belief_state_update = self.config.get("model", "learn_belief_state_update")
        if learn_belief_state_update in ["True", "true"]:
            self.learn_belief_state_update = True
        else:
            self.learn_belief_state_update = False
        print("learn_belief_state_update", self.learn_belief_state_update)
        # Get training details
        self.max_iteration = int(self.config.get("train", "max_iteration"))
        self.batch_size = int(self.config.get("train", "batch_size"))
        self.language = self.config.get("model", "language")
        self.id = self.config.get('model', 'id')
        self.restore_id = self.config.get('model', 'restore_id')
        restore = self.config.get("train", "restore")
        if restore in ["True", "true"]:
            self.restore = True
        else:
            self.restore = False
        self.language_suffix = self.lp[self.language]
        self.override_en_ontology = False
        # Create other additional word vectors
        _, utterances_train2 = load_woz_data("data/" + self.dataset_name + "/" + "tok_" + self.dataset_name + \
                                             "_train_" + self.language_suffix + ".json", self.language)
        _, utterances_val2 = load_woz_data("data/" + self.dataset_name + "/" + "tok_" + self.dataset_name + \
                                           "_validate_" + self.language_suffix + ".json", self.language)
        val_count = len(utterances_val2)
        self.utterances_train = utterances_train2 + utterances_val2[0:int(0.75 * val_count)]
        self.utterances_val = utterances_val2[int(0.75 * val_count):]
        self.target_model = None
        self.restore_model = None
        self.word_vectors = {}

        with open("word-vectors/ontology-mapping.json") as f:
            self.ontology_mapping = json.load(f)

    def init_model(self, params):
        # Model saver and session
        self.saver = tf.train.Saver(var_list=params)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # Reloading previous modelMultiNeuralBeliefTracker
        if tf.train.checkpoint_exists(self.restore_model) and self.restore:
            try:
                #self.saver.restore(self.sess, self.restore_model)
                self.optimistic_restore(self.restore_model)
                print("----------- Loading Model", self.restore_model, " ----------------")
            except Exception:
                print("failed to load model, start from scratch")

    def optimistic_restore(self, save_file):
        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

        with tf.variable_scope('', reuse=True):
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
        saver = tf.train.Saver(restore_vars)
        saver.restore(self.sess, save_file)

    def train(self):
        raise NotImplementedError

    def test(self, percentage=1.0, restore=True, printing=True):
        raise NotImplementedError

class CrossNeuralBeliefTracker(BeliefTracker):
    """
    Call to initialise the model with pre-trained parameters and given ontology.
    """
    def __init__(self, config_filepath):
        super(CrossNeuralBeliefTracker, self).__init__(config_filepath)

        if self.config.get('multilingual', 'only_dialogue') in ['true', 'True']:
            self.only_dialogue = True
        else:
            self.only_dialogue = False
        if self.config.get('multilingual', 'only_translate') in ['true', 'True']:
            self.only_translate = True
        else:
            self.only_translate = False
        self.foreign_language = self.config.get("model", "foreign_language")
        self.foreign_language_suffix = self.lp[self.foreign_language]
        self.alpha = float(self.config.get("train", "alpha"))
        self.tau = float(self.config.get("model", "tau"))
        self.restore_model = "models/transfer-model-{}-{}-{}".format(self.language_suffix, self.foreign_language_suffix, self.restore_id)
        self.target_model = "models/transfer-model-{}-{}-{}".format(self.language_suffix, self.foreign_language_suffix, self.id)
        self.word_vectors = read_embeeding('word-vectors/wiki.multi.{}.vec'.format(self.language_suffix))
        self.foreign_word_vectors = read_embeeding('word-vectors/wiki.multi.{}.vec'.format(self.foreign_language_suffix))
        bilingual_dict = 'word-vectors/{}-{}.dictionary.sorted.filtered'.format(self.language_suffix, self.foreign_language_suffix)
        if os.path.exists(bilingual_dict):
            self.bilingual_dictionary = bilingual_dictionary(bilingual_dict)
        self.word_vector_size = random.choice(self.word_vectors.values()).shape[0]

        with codecs.open("ontologies/ontology_dstc2_{}.json".format(self.language_suffix), 'r', 'utf8') as f:
            self.dialogue_ontology = json.load(f)["informable"]
        with codecs.open("ontologies/ontology_dstc2_{}.json".format(self.foreign_language_suffix), 'r', 'utf8') as f:
            self.foreign_dialogue_ontology = json.load(f)["informable"]

        name = 'data/trans/parallel.{}-{}.{}.txt'.format(self.language_suffix, self.foreign_language_suffix, self.language_suffix)
        if os.path.exists(name):
            with codecs.open(name, 'r', 'utf8') as f:
                self.translation_source = map(lambda x: x.strip(), f.readlines())
        name = 'data/trans/parallel.{}-{}.{}.txt'.format(self.language_suffix, self.foreign_language_suffix, self.foreign_language_suffix)
        if os.path.exists(name):
            with codecs.open(name, 'r', 'utf8') as f:
                self.translation_target = map(lambda x: x.strip(), f.readlines())

        self.foreign2primary = {}
        self.primary2foreign = {}
        for (en, de, it) in self.ontology_mapping:
            if self.foreign_language_suffix == "de":
                self.foreign2primary[de] = en
                self.primary2foreign[en] = de
            elif self.foreign_language_suffix == "it":
                self.foreign2primary[it] = en
                self.primary2foreign[en] = it

        src_embedding = self.create_ontology_embedding(self.dialogue_ontology, self.word_vectors)
        trg_embedding = self.create_ontology_embedding(self.foreign_dialogue_ontology, self.foreign_word_vectors)
        self.merge_vector(src_embedding, trg_embedding)

        # Create other additional word vectors
        self.test_dialogues, _ = load_woz_data("data/" + self.dataset_name + "/" + "tok_" +  self.dataset_name + \
                                               "_test_" + self.language_suffix + ".json", self.language)      
        self.test_foreign_dialogues, _ = load_woz_data("data/" + self.dataset_name + "/" + "tok_" +  self.dataset_name + \
                                               "_test_" + self.foreign_language_suffix + ".json", self.foreign_language)
        self.test_dialogues, _ = load_woz_data("data/" + self.dataset_name + "/" + "tok_" +  self.dataset_name + \
                                               "_validate_" + self.language_suffix + ".json", self.language)        
        self.valid_foreign_dialogues, _ = load_woz_data("data/" + self.dataset_name + "/" + "tok_" +  self.dataset_name + \
                                               "_validate_" + self.foreign_language_suffix + ".json", self.foreign_language)  
        # Neural Net Initialisation (keep variables packed so we can move them to either method):
        args = {"vector_dimension": self.word_vector_size, "parallel": "none", \
                "primary": self.language, "secondary": self.foreign_language, "alpha": self.alpha}

        self.models = {}
        for slot in self.dialogue_ontology:
            if slot == "request":
                self.models[slot] = NBTModel(label_count=len(self.dialogue_ontology[slot]), \
                                             use_softmax=False, slot_name=slot, **args)
            else:
                self.models[slot] = NBTModel(label_count=len(self.dialogue_ontology[slot]), \
                                            use_softmax=True, slot_name=slot, **args)
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="")
        self.init_model(self.params)

    def create_ontology_embedding(self, ontology, word_vectors):
        ontology_embedding = {}
        for ont in ontology:
            if ont in word_vectors:
                ontology_embedding[ont] = word_vectors[ont]
            elif " " in ont:
                constituents = ont.split(" ")
                if constituents[0] not in word_vectors:
                    word_vectors[constituents[0]] = xavier_vector(constituents[0])
                    print "failed {}".format(constituents[0])
                if constituents[1] not in word_vectors:
                    word_vectors[constituents[1]] = xavier_vector(constituents[1])
                    print "failed {}".format(constituents[1])
                ontology_embedding[ont] = word_vectors[constituents[0]] + word_vectors[constituents[1]]
                ontology_embedding[constituents[0]] = word_vectors[constituents[0]]
                ontology_embedding[constituents[1]] = word_vectors[constituents[1]]
            else:
                ontology_embedding[ont] = xavier_vector(ont)
                print "failed {}".format(ont)
            for term in ontology[ont]:
                if term in word_vectors:
                    ontology_embedding[term] = word_vectors[term]
                elif " " in term:
                    constituents = term.split(" ")
                    if constituents[0] not in word_vectors:
                        word_vectors[constituents[0]] = xavier_vector(constituents[0])
                        print "failed {}".format(constituents[0])
                    if constituents[1] not in word_vectors:
                        word_vectors[constituents[1]] = xavier_vector(constituents[1])
                        print "failed {}".format(constituents[1])
                    ontology_embedding[term] = word_vectors[constituents[0]] + word_vectors[constituents[1]]
                    ontology_embedding[constituents[0]] = word_vectors[constituents[0]]
                    ontology_embedding[constituents[1]] = word_vectors[constituents[1]]
                else:
                    ontology_embedding[term] = xavier_vector(term)
                    print "failed {}".format(term.encode('utf8'))
        return ontology_embedding

    def slot_name(self, name, language):
        return "{}_{}".format(language, name)

    def slot_value(self, slot):
        slot_vectors = self.word_vectors[slot]
        value_vectors = numpy.zeros((len(self.dialogue_ontology[slot]), self.word_vector_size), dtype="float32")
        for value_idx, value in enumerate(self.dialogue_ontology[slot]):
            value_vectors[value_idx, :] = self.word_vectors[value]
        return slot_vectors, value_vectors

    def foreign_slot_value(self, slot):
        slot_vectors = self.foreign_word_vectors[slot]
        value_vectors = numpy.zeros((len(self.foreign_dialogue_ontology[slot]), self.word_vector_size), dtype="float32")
        for value_idx, value in enumerate(self.foreign_dialogue_ontology[slot]):
            value_vectors[value_idx, :] = self.foreign_word_vectors[value]
        return slot_vectors, value_vectors

    def merge_vector(self, src_embedding, trg_embedding):
        os.system("mkdir -p models/")
        os.system("mkdir -p results/")

        if 'dont' in self.word_vectors and 'care' in self.word_vectors:
            src_embedding['dontcare'] = self.word_vectors['dont'] + self.word_vectors['care']
            src_embedding['es ist egal'] = self.word_vectors['dont'] + self.word_vectors['care']
            src_embedding['non importa'] = self.word_vectors['dont'] + self.word_vectors['care']
            trg_embedding['dontcare'] = self.word_vectors['dont'] + self.word_vectors['care']
            trg_embedding['es ist egal'] = self.word_vectors['dont'] + self.word_vectors['care']
            trg_embedding['non importa'] = self.word_vectors['dont'] + self.word_vectors['care']

        if 'request' in self.word_vectors:
            trg_embedding['request'] = self.word_vectors['request']

        self.word_vectors['UNK'] = xavier_vector("UNK")

        for k, v in src_embedding.iteritems():
            self.word_vectors[k] = v
        for k, v in trg_embedding.iteritems():
            self.foreign_word_vectors[k] = v

    def replace_dict(self, batch_xs_full, utterance):
        batch_xs_full_foreign = deepcopy(batch_xs_full)
        replace_history = []
        for i, utt in enumerate(utterance):
            utt = utt.split()
            max_len = len(utt)
            unnorm_prob = [math.exp(-self.tau * i) for i in range(max_len)]
            summed = sum(unnorm_prob)
            prob = [_ / summed for _ in unnorm_prob]
            num_sub = numpy.where(numpy.random.multinomial(1, prob) == 1)[0][0]
            sub_pos = numpy.random.choice(max_len, num_sub, replace=False)
            for pos in sub_pos:
                source_word = utt[pos]
                if source_word not in self.bilingual_dictionary:
                    continue
                mapped_target = self.bilingual_dictionary[source_word]
                picked_target = numpy.random.choice(mapped_target, 1)[0]
                replace_history.append((source_word, picked_target))
                if picked_target in self.foreign_word_vectors:
                    emb = self.foreign_word_vectors[picked_target]
                else:
                    emb = xavier_vector(picked_target)
                batch_xs_full_foreign[i][pos] = emb
        return batch_xs_full_foreign, replace_history


    def transfer_combine(self):
        system_act_features, system_act_features_foreign = extract_system_act_vectors(self.utterances_train,
                                                                                      self.word_vectors,
                                                                                      self.foreign_word_vectors,
                                                                                      self.primary2foreign)

    def transfer_dict(self):
        system_act_features, system_act_features_foreign = extract_system_act_vectors(self.utterances_train,
                                                                                      self.word_vectors,
                                                                                      self.foreign_word_vectors,
                                                                                      self.primary2foreign)
        train_feature_vectors = extract_feature_vectors(self.utterances_train, self.word_vectors, language=self.language)
        train_pos_examples, train_neg_examples = generate_data(self.utterances_train, self.dialogue_ontology)

        for iteration in range(self.max_iteration):
            rand = numpy.random.randint(0, len(self.dialogue_ontology))
            target_slot = self.dialogue_ontology.keys()[rand]
            slot_vector, value_vector = self.slot_value(target_slot)
            slot_vector_foreign, value_vector_foreign = self.foreign_slot_value(self.primary2foreign[target_slot])
            mx = self.models[target_slot]

            batch_system_act_features = []
            batch_system_act_features_foreign = []

            batch_data = generate_examples(target_slot, train_feature_vectors, self.dialogue_ontology, \
                                           train_pos_examples, train_neg_examples, self.batch_size, self.batch_size,
                                           origin_utterance=True)

            (batch_src_features, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys,
             batch_ys_prev, utterance, utterance_id) = batch_data

            for idx in utterance_id:
                batch_system_act_features.append(system_act_features[idx])
                batch_system_act_features_foreign.append(system_act_features_foreign[idx])

            batch_trg_features, replace_history = self.replace_dict(batch_src_features, utterance)

            transfer_loss = mx.train_transfer(self.sess, batch_xs_full=batch_src_features,
                                              batch_xs_full_foreign=batch_trg_features,
                                              batch_sys_req=[_[0] for _ in batch_system_act_features],
                                              batch_sys_req_foreign=[_[0] for _ in batch_system_act_features_foreign],
                                              batch_sys_conf_slots=[_[1] for _ in batch_system_act_features],
                                              batch_sys_conf_slots_foreign=[_[1] for _ in
                                                                            batch_system_act_features_foreign],
                                              batch_sys_conf_values=[_[2] for _ in batch_system_act_features],
                                              batch_sys_conf_values_foreign=[_[2] for _ in
                                                                             batch_system_act_features_foreign],
                                              slot_vector=slot_vector, slot_vector_foreign=slot_vector_foreign,
                                              value_vector=value_vector, value_vector_foreign=value_vector_foreign)
            if iteration % 1000 == 0 and iteration > 0:
                print("testing transfer langauge accuracy")
                req_acc, goal_acc = self.test_foreign(self.valid_foreign_dialogues, 1.0, False, False)
                print("Validation Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                req_acc, goal_acc = self.test_foreign(self.test_foreign_dialogues, 1.0, False, False)
                print("Testing Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                dialogue_loss = []
                trans_loss = []
                self.saver.save(self.sess, self.target_model)

    def transfer_corpus(self):
        system_act_features, system_act_features_foreign = extract_system_act_vectors(self.utterances_train, self.word_vectors,
                                                                                      self.foreign_word_vectors, self.primary2foreign)

        src_feature_vectors, trg_feature_vectors = extract_trans_feature_vectors(self.translation_source, self.translation_target,
                                                                                 self.word_vectors, self.foreign_word_vectors,
                                                                                 language=self.language,
                                                                                 foreign_language=self.foreign_language)
        for iteration in range(self.max_iteration):
            rand = numpy.random.randint(0, len(self.dialogue_ontology))
            target_slot = self.dialogue_ontology.keys()[rand]
            slot_vector, value_vector = self.slot_value(target_slot)
            slot_vector_foreign, value_vector_foreign = self.foreign_slot_value(self.primary2foreign[target_slot])
            mx = self.models[target_slot]

            rand = numpy.random.randint(0, len(system_act_features) - self.batch_size)
            batch_system_act_features = system_act_features[rand : rand + self.batch_size]
            batch_system_act_features_foreign = system_act_features_foreign[rand : rand + self.batch_size]

            batch_src_features = src_feature_vectors[rand: rand + self.batch_size]
            batch_trg_features = trg_feature_vectors[rand: rand + self.batch_size]

            transfer_loss = mx.train_transfer(self.sess, batch_xs_full=batch_src_features, batch_xs_full_foreign=batch_trg_features,
                                              batch_sys_req=[_[0] for _  in batch_system_act_features],
                                              batch_sys_req_foreign=[_[0] for _ in batch_system_act_features_foreign],
                                              batch_sys_conf_slots= [_[1] for _ in batch_system_act_features],
                                              batch_sys_conf_slots_foreign=[_[1] for _ in batch_system_act_features_foreign],
                                              batch_sys_conf_values=[_[2] for _ in batch_system_act_features],
                                              batch_sys_conf_values_foreign=[_[2] for _ in batch_system_act_features_foreign],
                                              slot_vector=slot_vector, slot_vector_foreign=slot_vector_foreign,
                                              value_vector=value_vector, value_vector_foreign=value_vector_foreign)
            if iteration % 2000 == 0 and iteration > 0:
                print("testing transfer langauge accuracy")
                req_acc, goal_acc = self.test_foreign(self.valid_foreign_dialogues, 1.0, False, False)                
                print("Validation Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                req_acc, goal_acc = self.test_foreign(self.test_foreign_dialogues, 1.0, False, False)
                print("Testing Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                dialogue_loss = []
                trans_loss = []
                self.saver.save(self.sess, self.target_model)
    def train(self):
        train_feature_vectors = extract_feature_vectors(self.utterances_train, self.word_vectors, language=self.language)
        valid_feature_vectors = extract_feature_vectors(self.utterances_val, self.word_vectors, language=self.language)

        train_pos_examples, train_neg_examples = generate_data(self.utterances_train, self.dialogue_ontology)
        val_pos_examples, val_neg_examples = generate_data(self.utterances_val, self.dialogue_ontology)

        start_time = time.time()
        dialogue_loss = []
        trans_loss = []
        for iteration in range(self.max_iteration):
            rand = numpy.random.randint(0, len(self.dialogue_ontology))
            target_slot = self.dialogue_ontology.keys()[rand]
            #target_slot = "request"
            batch_data = generate_examples(target_slot, train_feature_vectors, self.dialogue_ontology, \
                                           train_pos_examples, train_neg_examples, self.batch_size, 0)
            (batch_xs_full, batch_sys_req, batch_sys_conf_slots, batch_sys_conf_values, batch_ys, batch_ys_prev) = batch_data

            mx = self.models[target_slot]
            slot_vector, value_vector = self.slot_value(target_slot)
            loss = mx.train(self.sess, batch_xs_full, batch_sys_req, batch_sys_conf_slots, \
                            batch_sys_conf_values, batch_ys, batch_ys_prev, slot_vector, value_vector)
            dialogue_loss.append(loss)
            #print "Iteration:{} Transfer loss = {}".format(iteration, loss)
            #"""
            if iteration % 2000 == 0 and iteration > 0:
                #print("testing native language accuracy")
                #req_acc, goal_acc = self.test(0.5, False, False)
                #print("Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                print("testing transfer langauge accuracy")
                req_acc, goal_acc = self.test(1.0, False, False)
                print("Goal acc: ", round(goal_acc, 5), "Request acc: ", round(req_acc, 5))
                dialogue_loss = []
                trans_loss = []
                self.saver.save(self.sess, self.target_model)
            #"""

    def test(self, percentage=1.0, restore=True, printing=True):
        evaluated_dialogues = []
        dialogue_count = int(len(self.test_dialogues) * percentage)
        for idx in range(0, dialogue_count):
            if idx % 100 == 0 and printing:  # progress for test
                print("{}/{} done".format(idx, dialogue_count))

            prev_belief_states = {}
            belief_states = {}  # for each slot, a list of numpy arrays.

            for slot in self.dialogue_ontology:
                belief_states[slot] = {}
                if slot != "request":
                    value_count = len(self.dialogue_ontology[slot]) + 1
                    prev_belief_states[slot] = numpy.zeros((value_count,), dtype="float32")

            predictions_for_dialogue = []
            belief_states = []
            prev_bs = None

            for idx, trans_and_req_and_label_and_currlabel in enumerate(self.test_dialogues[idx]):
                belief_states.append({})
                current_bs = {}
                transcription_and_asr, req_slot, conf_slot, conf_value, label, prev_belief_state = trans_and_req_and_label_and_currlabel
                for slot in self.dialogue_ontology:
                    if idx == 0:
                        example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_belief_state)]
                    else:
                        example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_bs)]

                    potential_values = self.dialogue_ontology[slot]
                    if slot == "request":
                        value_count = len(potential_values)
                    else:
                        value_count = len(potential_values) + 1

                    # should be a list of features for each ngram supplied.
                    fv_tuples = extract_feature_vectors(example, self.word_vectors, language=self.language)

                    # accumulators
                    fv_full = []
                    fv_sys_req = []
                    fv_conf_slot = []
                    fv_conf_val = []
                    features_previous_state = []

                    for idx_hyp, extracted_fv in enumerate(fv_tuples):
                        prev_belief_state_vector = numpy.zeros((value_count,), dtype="float32")
                        if slot != "request":
                            prev_value = example[idx_hyp][4][slot]
                            if prev_value == "none" or prev_value not in self.dialogue_ontology[slot]:
                                prev_belief_state_vector[value_count - 1] = 1
                            else:
                                prev_belief_state_vector[self.dialogue_ontology[slot].index(prev_value)] = 1

                        features_previous_state.append(prev_belief_state_vector)
                        fv_full.append(extracted_fv[0])
                        fv_sys_req.append(extracted_fv[1])
                        fv_conf_slot.append(extracted_fv[2])
                        fv_conf_val.append(extracted_fv[3])

                    mx = self.models[slot]
                    slot_vector, value_vector = self.slot_value(slot)
                    distribution = mx.predict(self.sess, fv_full,fv_sys_req,
                                              fv_conf_slot, fv_conf_val, features_previous_state, slot_vector, value_vector)
                    state_distribution = distribution[0]
                    if slot in "request":
                        current_bs[slot] = print_belief_state_woz_requestables(self.dialogue_ontology[slot], \
                                                                               state_distribution, threshold=0.5)
                    else:
                        current_bs[slot] = print_belief_state_woz_informable(self.dialogue_ontology[slot], \
                                                                             state_distribution, threshold=0.01)

                prev_bs = deepcopy(current_bs)
                trans_plus_sys = "User: " + transcription_and_asr[0]
                # + req_slot, conf_slot, conf_value
                if req_slot[0] != "":
                    trans_plus_sys += "    System Request: " + str(req_slot)
                if conf_slot[0] != "":
                    trans_plus_sys += "    System Confirm: " + str(conf_slot) + " " + str(conf_value)
                predictions_for_dialogue.append({"ASR": trans_plus_sys, "True State": label, \
                                                 "Prediction": current_bs, "Previous State": prev_belief_state})

            evaluated_dialogues.append(predictions_for_dialogue)

        dialogue_count = len(evaluated_dialogues)
        indexed_dialogues = []

        for d_idx in range(0, dialogue_count):
            new_dialogue = {}
            new_dialogue["dialogue_idx"] = d_idx
            new_dialogue["dialogue"] = evaluated_dialogues[d_idx]
            indexed_dialogues.append(new_dialogue)

        with codecs.open("results/woz_tracking.json", "w", "utf8") as f:
            json.dump(indexed_dialogues, f, indent=4, ensure_ascii=False)

        req_acc, goal_acc = evaluate_woz(indexed_dialogues, self.dialogue_ontology, printing=printing)
        return req_acc, goal_acc

    def test_foreign(self, dialogues, percentage=1.0, restore=True, printing=True):
        evaluated_dialogues = []
        dialogue_count = int(len(dialogues) * percentage)
        for idx in range(0, dialogue_count):
            if idx % 100 == 0 and printing:  # progress for test
                print(idx, "/", dialogue_count, "done.")

            prev_belief_states = {}
            belief_states = {}  # for each slot, a list of numpy arrays.

            for slot in self.foreign_dialogue_ontology:
                belief_states[slot] = {}
                if slot != "request":
                    value_count = len(self.foreign_dialogue_ontology[slot]) + 1
                    prev_belief_states[slot] = numpy.zeros((value_count,), dtype="float32")

            predictions_for_dialogue = []
            belief_states = []
            prev_bs = None

            for idx, trans_and_req_and_label_and_currlabel in enumerate(dialogues[idx]):
                belief_states.append({})
                current_bs = {}
                transcription_and_asr, req_slot, conf_slot, conf_value, label, prev_belief_state = trans_and_req_and_label_and_currlabel
                for slot in self.foreign_dialogue_ontology:
                    if idx == 0:
                        example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_belief_state)]
                    else:
                        example = [(transcription_and_asr, req_slot, conf_slot, conf_value, prev_bs)]

                    potential_values = self.foreign_dialogue_ontology[slot]
                    if slot == "request":
                        value_count = len(potential_values)
                    else:
                        value_count = len(potential_values) + 1

                    # should be a list of features for each ngram supplied.
                    fv_tuples = extract_feature_vectors(example, self.foreign_word_vectors, language=self.foreign_language)

                    # accumulators
                    fv_full = []
                    fv_sys_req = []
                    fv_conf_slot = []
                    fv_conf_val = []
                    features_previous_state = []

                    for idx_hyp, extracted_fv in enumerate(fv_tuples):
                        prev_belief_state_vector = numpy.zeros((value_count,), dtype="float32")
                        if slot != "request":
                            prev_value = example[idx_hyp][4][slot]
                            if prev_value == "none" or prev_value not in self.foreign_dialogue_ontology[slot]:
                                prev_belief_state_vector[value_count - 1] = 1
                            else:
                                prev_belief_state_vector[self.foreign_dialogue_ontology[slot].index(prev_value)] = 1

                        features_previous_state.append(prev_belief_state_vector)
                        fv_full.append(extracted_fv[0])
                        fv_sys_req.append(extracted_fv[1])
                        fv_conf_slot.append(extracted_fv[2])
                        fv_conf_val.append(extracted_fv[3])

                    mx = self.models[self.foreign2primary[slot]]
                    slot_vector, value_vector = self.foreign_slot_value(slot)
                    distribution = mx.predict_foreign(self.sess, fv_full, fv_sys_req,
                                                      fv_conf_slot, fv_conf_val, features_previous_state, slot_vector, value_vector)
                    state_distribution = distribution[0]
                    if slot in "request":
                        current_bs[slot] = print_belief_state_woz_requestables(self.foreign_dialogue_ontology[slot], \
                                                                             state_distribution, threshold=0.5)
                    else:
                        current_bs[slot] = print_belief_state_woz_informable(self.foreign_dialogue_ontology[slot], \
                                                                             state_distribution, threshold=0.01)

                prev_bs = deepcopy(current_bs)
                trans_plus_sys = "User: " + transcription_and_asr[0]
                # + req_slot, conf_slot, conf_value
                if req_slot[0] != "":
                    trans_plus_sys += "    System Request: " + str(req_slot)
                if conf_slot[0] != "":
                    trans_plus_sys += "    System Confirm: " + str(conf_slot) + " " + str(conf_value)
                predictions_for_dialogue.append({"ASR": trans_plus_sys, "True State": label, "Prediction": current_bs,
                                                 "Previous State": prev_belief_state})

            evaluated_dialogues.append(predictions_for_dialogue)

        dialogue_count = len(evaluated_dialogues)
        indexed_dialogues = []

        for d_idx in range(0, dialogue_count):
            new_dialogue = {}
            new_dialogue["dialogue_idx"] = d_idx
            new_dialogue["dialogue"] = evaluated_dialogues[d_idx]
            indexed_dialogues.append(new_dialogue)

        with codecs.open("results/woz_tracking.json", "w", "utf8") as f:
            json.dump(indexed_dialogues, f, indent=4, ensure_ascii=False)

        req_acc, goal_acc = evaluate_woz(indexed_dialogues, self.foreign_dialogue_ontology, printing=printing)
        return req_acc, goal_acc

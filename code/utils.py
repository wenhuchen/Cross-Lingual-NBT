# -*- coding: utf-8 -*-

import numpy
import math
import codecs
import string
import json
from copy import deepcopy
import random
from nltk.stem.snowball import SnowballStemmer
import re
from collections import OrderedDict

def bilingual_dictionary(name):
    mapping = {}
    with codecs.open(name, 'r', 'utf8') as f:
        for line in f:
            line = line.strip()
            src = line.split()[0]
            trg = line.split()[1]
            if src not in mapping:
                mapping[src] = [trg]
            else:
                mapping[src].append(trg)
    return mapping

def read_bilingual_embedding(emb, language_suffix, foreign_language_suffix):
    """
    Read Bi-embeeding file
    """
    word_dictionary = OrderedDict()
    foreign_dictionary = OrderedDict()
    with codecs.open(emb, 'r', 'utf8') as f:
        for line in f:
            if len(line.split()) > 2:
                line = line.strip()
                line = line.split(" ", 1)
                if line[0].startswith(language_suffix + "_"):
                    word_dictionary[line[0][3:]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
                elif line[0].startswith(foreign_language_suffix + "_"):
                    foreign_dictionary[line[0][3:]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
                elif line[0].startswith("meta_"):
                    word_dictionary[line[0][5:]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
                    foreign_dictionary[line[0][5:]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
    return normalise_word_vectors(word_dictionary), normalise_word_vectors(foreign_dictionary)

def read_embeeding(emb, language_suffix="en"):
    """
    Read embeeding file
    """
    word_dictionary = OrderedDict()
    with codecs.open(emb, 'r', 'utf8') as f:
        for line in f:
            if len(line.split()) > 2:
                line = line.split(" ", 1)
                if line[0].startswith(language_suffix + "_"):
                    word_dictionary[line[0][3:]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
                else:
                    word_dictionary[line[0]] = numpy.fromstring(line[1], dtype="float32", sep=" ")
    return normalise_word_vectors(word_dictionary)

def xavier_vector(word, D=300):
    """
    Initialize vector
    """
    seed_value =abs(hash(word)) % (10 ** 8)
    numpy.random.seed(seed_value)
    neg_value = - math.sqrt(6)/math.sqrt(D)
    pos_value = math.sqrt(6)/math.sqrt(D)
    rsample = numpy.random.uniform(low=neg_value, high=pos_value, size=(D,))
    norm = numpy.linalg.norm(rsample)
    rsample_normed = rsample/norm
    return rsample_normed

def binary_mask(example, requestable_count):
    """
    Convert to one-hot vector
    """
    zeros = numpy.zeros((requestable_count,), dtype=numpy.float32)
    for x in example:
        zeros[x] = 1
    return zeros

def compare_request_lists(list_a, list_b):
    """
    Compare whether two lists are equal
    """
    if len(list_a) != len(list_b):
        return False
    list_a.sort()
    list_b.sort()
    for idx in range(0, len(list_a)):
        if list_a[idx] != list_b[idx]:
            return False
    return True

def delexicalise_utterance_values(utterance, target_slot, target_values):
    """
    Delexicalize the utterance value and tries to find out the matching in the utterance
    """
    if type(utterance) is list:
        utterance = " ".join(utterance)
    if target_slot == "request":
        value_count = len(target_values)
    else:
        value_count = len(target_values)+1
    delexicalised_vector = numpy.zeros((value_count,), dtype="float32")
    for idx, target_value in enumerate(target_values):
        if " " + target_value + " " in utterance:
            delexicalised_vector[idx] = 1.0
    return delexicalised_vector

def print_belief_state_woz_informable(curr_values, distribution, threshold):
    """
    Returns the top one if it is above threshold.
    """
    max_value = "none"
    max_score = 0.0
    total_value = 0.0

    for idx, value in enumerate(curr_values):

        total_value += distribution[idx]

        if distribution[idx] >= threshold:

            if distribution[idx] >= max_score:
                max_value = value
                max_score = distribution[idx]

    if max_score >= (1.0 - total_value):
        return max_value
    else:
        return "none"

def print_belief_state_woz_requestables(curr_values, distribution, threshold):
    """
    Returns the top one if it is above threshold.
    """
    requested_slots = []

    # now we just print to JSON file:
    for idx, value in enumerate(curr_values):

        if distribution[idx] >= threshold:
            requested_slots.append(value)

    return requested_slots

def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors

def print_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
    """
    Prints all the activated slot values for the provided predictions
    """
    predicted_values = []
    for idx, value in enumerate(slot_values):
        if distribution[idx] >= threshold:
            predicted_values += ((value, round(distribution[idx], 2)))  # , round(post_sf[idx], 2) ))
    print("Predictions for", str(target_slot + ":"), predicted_values)
    return predicted_values


def return_slot_predictions(distribution, slot_values, target_slot, threshold=0.05):
    """
    Prints all the activated slot values for the provided predictions
    """
    predicted_values = []
    for idx, value in enumerate(slot_values):
        if distribution[idx] >= threshold:
            predicted_values += ((value, round(distribution[idx], 2)))  # , round(post_sf[idx], 2) ))

    return predicted_values

def process_turn_hyp(transcription, language):
    """
    Returns the clean (i.e. handling interpunction signs) string for the given language.
    """
    exclude = set(string.punctuation)
    exclude.remove("'")

    transcription = ''.join(ch for ch in transcription if ch not in exclude)

    transcription = transcription.lower()
    transcription = transcription.replace(u"’", "'")
    transcription = transcription.replace(u"‘", "'")
    transcription = transcription.replace("don't", "dont")
    if language == "it" or language == "italian":# or language == "en" or language == "english":
        transcription = transcription.replace("'", " ")
    if language == "en" or language == "english":# or language == "en" or language == "english":
        transcription = transcription.replace("'", "")

    return transcription

def evaluate_woz(f, dialogue_ontology, printing=True):
    hit = {'goal': 0}
    miss = {'goal': 0}
    none_hit = {'goal': 0}
    none_miss = {'goal': 0}
    for slot in dialogue_ontology:
        hit[slot] = 0
        miss[slot] = 0
        none_hit[slot] = 0
        none_miss[slot] = 0
    for dialogue in f:
        for turn in dialogue['dialogue']:
            true_states = turn['True State']
            prediction_states = turn['Prediction']
            for slot in dialogue_ontology:
                if isinstance(true_states[slot], list):
                    if len(true_states[slot]) == 0:
                        if len(prediction_states[slot]) == 0:
                            none_hit['request'] += 1
                        else:
                            none_miss['request'] += 1
                        continue
                    flag = compare_request_lists(true_states[slot], prediction_states[slot])
                else:
                    if true_states[slot] == "none":
                        if prediction_states[slot] == "none":
                            none_hit['goal'] += 1
                        else:
                            none_miss['goal'] += 1
                        continue
                    flag = true_states[slot] == prediction_states[slot]
                    if not flag and slot == "food":
                        with open("/tmp/errors", "a") as f:
                            print >> f, turn['ASR']
                            print >> f, turn['True State']
                            print >> f, turn['Prediction']
                            print >> f, turn['Previous State']

                if flag:
                    hit[slot] += 1
                    if slot != "request":
                        hit['goal'] += 1
                else:
                    miss[slot] += 1
                    if slot != "request":
                        miss['goal'] += 1
    #acc = {'food': 0, 'area': 0, 'price range': 0}
    #for key in hit:
    #    if key != "request" and key != "goal":
    #       acc[key] = hit[key] / (hit[key] + miss[key] + 0.0)
    #       if printing:
    #            print("{} accuracy is {}".format(key, acc[key]))

    req_acc = hit['request'] / (hit['request'] + miss['request'] + 0.01)
    goal_acc = hit['goal'] / (hit['goal'] + miss['goal'] + 0.01)
    none_req_acc = none_hit['request'] / (none_hit['request'] + none_miss['request'] + 0.01)
    none_goal_acc = none_hit['goal'] / (none_hit['goal'] + none_miss['goal'] + 0.01)
    if printing:
        print "-----------------------------------------------------------"
        print("request accuracy is {}".format(req_acc))
        print("joint goal accuracy is {}".format(goal_acc))
        print("none request accuracy is {}".format(none_req_acc))
        print("none joint goal accuracy is {}".format(none_goal_acc))
        print "-----------------------------------------------------------"

    return req_acc, goal_acc


def load_woz_data(file_path, language, percentage=1.0, override_en_ontology=False):
    """
    This method loads WOZ dataset as a collection of utterances.

    Testing means load everything, no split.
    """
    with codecs.open(file_path, 'r', 'utf8') as f:
        woz_json = json.load(f)
    dialogues = []
    training_turns = []
    dialogue_count = len(woz_json)
    percentage = float(percentage)
    dialogue_count = int(percentage * float(dialogue_count))

    print("loading from file {} totally {} dialogues".format(file_path, dialogue_count))

    for idx in range(0, dialogue_count):
        current_dialogue = process_woz_dialogue(woz_json[idx]["dialogue"], language, override_en_ontology)
        dialogues.append(current_dialogue)
        for turn_idx, turn in enumerate(current_dialogue):
            current_label = []

            for req_slot in turn[4]["request"]:
                current_label.append(("request", req_slot))

            # this now includes requests:
            for inf_slot in turn[4]:
                if inf_slot != "request":
                    current_label.append((inf_slot, turn[4][inf_slot]))

            transcription_and_asr = turn[0]
            current_utterance = (transcription_and_asr, turn[1], turn[2], turn[3], current_label, turn[5])  # turn [5] is the past belief state

            training_turns.append(current_utterance)
    return (dialogues, training_turns)


def process_woz_dialogue(woz_dialogue, language, override_en_ontology):
    """
    Returns a list of (tuple, belief_state) for each turn in the dialogue.
    """
    # initial belief state
    # belief state to be given at each turn
    if language == "english" or language == "en" or override_en_ontology:
        null_bs = {}
        null_bs["food"] = "none"
        null_bs["price range"] = "none"
        null_bs["area"] = "none"
        null_bs["request"] = []
        informable_slots = ["food", "price range", "area"]
        pure_requestables = ["address", "phone", "postcode"]

    elif (language == "italian" or language == "it"):
        null_bs = {}
        null_bs["area"] = "none"
        null_bs["cibo"] = "none"
        null_bs["prezzo"] = "none"
        null_bs["request"] = []
        informable_slots = ["cibo", "prezzo", "area"]
        pure_requestables = ["codice postale", "telefono", "indirizzo"]

    elif (language == "german" or language == "de"):
        null_bs = {}
        null_bs["gegend"] = "none"
        null_bs["essen"] = "none"
        null_bs["preisklasse"] = "none"
        null_bs["request"] = []
        informable_slots = ["essen", "preisklasse", "gegend"]
        pure_requestables = ["postleitzahl", "telefon", "adresse"]
    else:
        null_bs = {}
        pure_requestables = None

    prev_belief_state = deepcopy(null_bs)
    dialogue_representation = []

    lp = {}
    lp["german"] = u"de_"
    lp["italian"] = u"it_"

    for idx, turn in enumerate(woz_dialogue):

        current_DA = turn["system_acts"]

        current_req = []
        current_conf_slot = []
        current_conf_value = []

        for each_da in current_DA:
            if each_da in informable_slots:
                current_req.append(each_da)
            elif each_da in pure_requestables:
                current_conf_slot.append("request")
                current_conf_value.append(each_da)
            else:
                if type(each_da) is list:
                    current_conf_slot.append(each_da[0])
                    current_conf_value.append(each_da[1])

        if not current_req:
            current_req = [""]

        if not current_conf_slot:
            current_conf_slot = [""]
            current_conf_value = [""]

        current_transcription = turn["transcript"]
        current_transcription = current_transcription

        read_asr = turn["asr"]

        current_asr = []

        for (hyp, score) in read_asr:
            current_hyp = hyp
            current_asr.append((current_hyp, score))

        exclude = set(string.punctuation)
        exclude.remove("'")

        current_transcription = ''.join(ch for ch in current_transcription if ch not in exclude)
        current_transcription = current_transcription.lower()

        current_labels = turn["turn_label"]

        current_bs = deepcopy(prev_belief_state)

        # print "=====", prev_belief_state
        if "request" in prev_belief_state:
            del prev_belief_state["request"]

        current_bs["request"] = []  # reset requestables at each turn

        for label in current_labels:
            (c_slot, c_value) = label

            if c_slot in informable_slots:
                current_bs[c_slot] = c_value

            elif c_slot == "request":
                current_bs["request"].append(c_value)

        curr_lab_dict = {}
        for x in current_labels:
            if x[0] != "request":
                curr_lab_dict[x[0]] = x[1]

        dialogue_representation.append(((current_transcription, current_asr), current_req, current_conf_slot,
                                        current_conf_value, deepcopy(current_bs), deepcopy(prev_belief_state)))

        prev_belief_state = deepcopy(current_bs)

    return dialogue_representation

def generate_examples(target_slot, feature_vectors, dialogue_ontology, positive_examples, negative_examples,
                      positive_count=None, negative_count=None, origin_utterance=False):
    """
    This method returns a minibatch of positive_count examples followed by negative_count examples.
    If these two are not set, it creates the full dataset (used for validation and test).
    It returns: (features_unigram, features_bigram, features_trigram, features_slot,
                 features_values, y_labels) - all we need to pass to train.
    """
    # total number of positive and negative examples.
    pos_example_count = len(positive_examples[target_slot])
    neg_example_count = len(negative_examples[target_slot])

    if positive_count is None:
        positive_indices = numpy.arange(0, pos_example_count)
        positive_count = pos_example_count
    else:
        positive_indices = numpy.random.choice(pos_example_count, positive_count)
    if negative_count is None:
        negative_indices = numpy.arange(0, neg_example_count)
        negative_count = neg_example_count
    else:
        negative_indices = numpy.random.choice(neg_example_count, negative_count)

    if target_slot != "request":
        label_count = len(dialogue_ontology[target_slot]) + 1  # NONE
    else:
        label_count = len(dialogue_ontology[target_slot])

    examples = []
    labels = []

    for idx in positive_indices:
        examples.append(positive_examples[target_slot][idx])
    for idx in negative_indices:
        examples.append(negative_examples[target_slot][idx])

    value_count = len(dialogue_ontology[target_slot])

    # each element of this array is (xs_unigram, xs_bigram, xs_trigram, fv_slot, fv_value):
    features_requested_slots = []
    features_confirm_slots = []
    features_confirm_values = []
    features_full = []
    features_previous_state = []
    utterance_full = []
    utterance_idx_full = []
    # now go through all examples (positive followed by negative).
    for idx_example, example in enumerate(examples):
        (utterance_idx, utterance, value_idx) = example
        utterance_fv = feature_vectors[utterance_idx]
        utterance_idx_full.append(utterance_idx)

        # prev belief state is in utterance[5]
        prev_belief_state = utterance[5]

        if idx_example < positive_count:
            if target_slot != "request":
                labels.append(value_idx)  # includes dontcare
            else:
                labels.append(binary_mask(value_idx, len(dialogue_ontology["request"])))
        else:
            if target_slot != "request":
                labels.append(value_count)  # NONE
            else:
                labels.append([])  # wont ever use this

        features_full.append(utterance_fv[0])
        utterance_full.append(utterance[0][0])
        features_requested_slots.append(utterance_fv[1])
        features_confirm_slots.append(utterance_fv[2])
        features_confirm_values.append(utterance_fv[3])
        prev_belief_state_vector = numpy.zeros((label_count,), dtype="float32")

        if target_slot != "request":
            prev_value = prev_belief_state[target_slot]
            if prev_value == "none" or prev_value not in dialogue_ontology[target_slot]:
                prev_belief_state_vector[label_count - 1] = 1
            else:
                prev_belief_state_vector[dialogue_ontology[target_slot].index(prev_value)] = 1

        features_previous_state.append(prev_belief_state_vector)

    features_full = numpy.array(features_full)
    features_requested_slots = numpy.array(features_requested_slots)
    features_confirm_slots = numpy.array(features_confirm_slots)
    features_confirm_values = numpy.array(features_confirm_values)
    features_previous_state = numpy.array(features_previous_state)

    y_labels = numpy.zeros((positive_count + negative_count, label_count), dtype="float32")
    for idx in range(0, positive_count):
        if target_slot != "request":
            y_labels[idx, labels[idx]] = 1
        else:
            y_labels[idx, :] = labels[idx]

    if target_slot == "request":
        y_labels[positive_count:, :] = 0
    if target_slot != "request":
        y_labels[positive_count:, label_count - 1] = 1

    if origin_utterance:
        return (features_full, features_requested_slots, features_confirm_slots, \
                features_confirm_values, y_labels, features_previous_state, utterance_full, utterance_idx_full)
    else:
        return (features_full, features_requested_slots, features_confirm_slots, \
                features_confirm_values, y_labels, features_previous_state)

def extract_system_act_vectors(utterances, word_vectors, foreign_word_vectors, primary2foreign):
    word_vector_size = random.choice(word_vectors.values()).shape[0]
    system_acts = []
    system_acts_foreign = []

    for idx, utterance in enumerate(utterances):
        requested_slots = utterances[idx][1]
        current_requested_vector = numpy.zeros((word_vector_size,), dtype="float32")
        current_foreign_requested_vector = numpy.zeros((word_vector_size,), dtype="float32")
        for requested_slot in requested_slots:
            if requested_slot != "":
                current_requested_vector += word_vectors[requested_slot]
                current_foreign_requested_vector += foreign_word_vectors[primary2foreign[requested_slot]]

        curr_confirm_slots = utterances[idx][2]
        curr_confirm_values = utterances[idx][3]

        current_conf_slot_vector = numpy.zeros((word_vector_size,), dtype="float32")
        current_conf_value_vector = numpy.zeros((word_vector_size,), dtype="float32")
        current_conf_slot_vector_foreign = numpy.zeros((word_vector_size,), dtype="float32")
        current_conf_value_vector_foreign = numpy.zeros((word_vector_size,), dtype="float32")
        confirmation_count = len(curr_confirm_slots)

        for sub_idx in range(0, confirmation_count):
            current_cslot = curr_confirm_slots[sub_idx]
            current_cvalue = curr_confirm_values[sub_idx]

            if current_cslot != "" and current_cvalue != "":
                current_conf_slot_vector += word_vectors[current_cslot]
                current_conf_value_vector += word_vectors[current_cvalue]

                current_conf_slot_vector_foreign += foreign_word_vectors[primary2foreign[current_cslot]]
                current_conf_value_vector_foreign += foreign_word_vectors[primary2foreign[current_cvalue]]

        system_acts.append((current_requested_vector, current_conf_slot_vector, current_conf_value_vector))
        system_acts_foreign.append((current_foreign_requested_vector, current_conf_slot_vector_foreign, current_conf_value_vector_foreign))

    return system_acts, system_acts_foreign

def extract_feature_vectors(utterances, word_vectors, longest_utterance_length=40, language="english"):
    """
    This method returns feature vectors for all dialogue utterances.
    It returns a tuple of lists, where each list consists of all feature vectors for ngrams of that length.
    This method doesn't care about the labels: other methods assign actual or fake labels later on.
    This can run on any size, including a single utterance.
    """
    stemmer = SnowballStemmer(language)
    word_vector_size = random.choice(word_vectors.values()).shape[0]

    utterance_count = len(utterances)
    ngram_feature_vectors = []

    # let index 6 denote full FV (for conv net):
    for j in range(0, utterance_count):
        ngram_feature_vectors.append(numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32"))

    requested_slot_vectors = []
    confirm_slots = []
    confirm_values = []

    success = 0
    fail = 0
    for idx, utterance in enumerate(utterances):
        full_asr = utterances[idx][0][0]  # just use ASR

        requested_slots = utterances[idx][1]
        current_requested_vector = numpy.zeros((word_vector_size,), dtype="float32")
        for requested_slot in requested_slots:
            if requested_slot != "":
                current_requested_vector += word_vectors[requested_slot]

        requested_slot_vectors.append(current_requested_vector)

        curr_confirm_slots = utterances[idx][2]
        curr_confirm_values = utterances[idx][3]

        current_conf_slot_vector = numpy.zeros((word_vector_size,), dtype="float32")
        current_conf_value_vector = numpy.zeros((word_vector_size,), dtype="float32")

        confirmation_count = len(curr_confirm_slots)

        for sub_idx in range(0, confirmation_count):
            current_cslot = curr_confirm_slots[sub_idx]
            current_cvalue = curr_confirm_values[sub_idx]

            if current_cslot != "" and current_cvalue != "":
                #if " " not in current_cslot:
                current_conf_slot_vector += word_vectors[current_cslot]
                #else:
                #    words_in_example = current_cslot.split()
                #    for cword in words_in_example:
                #        current_conf_slot_vector += word_vectors[unicode(cword)]

                #if " " not in current_cvalue:
                current_conf_value_vector += word_vectors[current_cvalue]
                #else:
                #    words_in_example = current_cvalue.split()
                #    for cword in words_in_example:
                #        current_conf_value_vector += word_vectors[unicode(cword)]

        confirm_slots.append(current_conf_slot_vector)
        confirm_values.append(current_conf_value_vector)

        full_fv = numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32")

        if full_asr != "":
            # print c_example
            words_utterance = full_asr
            for word_idx, word in enumerate(words_utterance.split()):
                if word_idx == longest_utterance_length:
                    break
                if word not in word_vectors:
                    root = stemmer.stem(word)
                    if root in word_vectors:
                        success += 1
                        word_vectors[word] = word_vectors[root]
                    else:
                        fail += 1
                        word_vectors[word] = xavier_vector(word)
                else:
                    success += 1
                full_fv[word_idx * word_vector_size: (word_idx + 1) * word_vector_size] = word_vectors[word]

        ngram_feature_vectors[idx] = numpy.reshape(full_fv, (longest_utterance_length, word_vector_size))

    #print "reading dialogue, success={}, fail={}".format(success, fail)
    list_of_features = []
    for idx in range(0, utterance_count):
        list_of_features.append((ngram_feature_vectors[idx],
                                 requested_slot_vectors[idx],
                                 confirm_slots[idx],
                                 confirm_values[idx],
                                 ))

    return list_of_features

def extract_trans_feature_vectors(source_sentence, target_sentence, word_vectors, foreign_word_vectors,
                                  longest_utterance_length=40, language="english", foreign_language="german"):
    stemmer = SnowballStemmer(language)
    foreign_stemmer = SnowballStemmer(foreign_language)
    sentence_count = len(source_sentence)
    ngram_src_vectors = []
    ngram_trg_vectors = []
    word_vector_size = random.choice(word_vectors.values()).shape[0]
    success = 0
    fail = 0
    # let index 6 denote full FV (for conv net):
    for j in range(0, sentence_count):
        ngram_src_vectors.append(numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32"))
        ngram_trg_vectors.append(numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32"))

    for idx, (s, t) in enumerate(zip(source_sentence, target_sentence)):
        full_fv = numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32")
        for word_idx, word in enumerate(s.split()):
            if word not in word_vectors:
                root = stemmer.stem(word)
                if root in word_vectors:
                    word_vectors[word] = word_vectors[root]
                    success += 1
                else:
                    word_vectors[word] = xavier_vector(word)
                    fail += 1
            else:
                success +=1
            full_fv[word_idx * word_vector_size: (word_idx + 1) * word_vector_size] = word_vectors[word]
            ngram_src_vectors[idx] = numpy.reshape(full_fv, (longest_utterance_length, word_vector_size))

        full_fv = numpy.zeros((longest_utterance_length * word_vector_size,), dtype="float32")
        for word_idx, word in enumerate(t.split()):
            if word not in foreign_word_vectors:
                root = foreign_stemmer.stem(word)
                if root in foreign_word_vectors:
                    foreign_word_vectors[word] = foreign_word_vectors[root]
                    success += 1
                else:
                    foreign_word_vectors[word] = xavier_vector(word)
                    fail += 1
            else:
                success += 1
            full_fv[word_idx * word_vector_size: (word_idx + 1) * word_vector_size] = foreign_word_vectors[word]
            ngram_trg_vectors[idx] = numpy.reshape(full_fv, (longest_utterance_length, word_vector_size))
    print "reading translation success={}, fail={}".format(success, fail)

    return ngram_src_vectors, ngram_trg_vectors

def generate_data(utterances, dialogue_ontology):
    positive_examples = {}
    negative_examples = {}

    list_of_slots = dialogue_ontology.keys()
    for slot_idx, slot in enumerate(list_of_slots):

        positive_examples[slot] = []
        negative_examples[slot] = []

        for utterance_idx, utterance in enumerate(utterances):
            slot_expressed_in_utterance = False

            for (slotA, valueA) in utterance[4]:
                if slotA == slot and (valueA != "none" and valueA != []):
                    slot_expressed_in_utterance = True  #  if this is True, no negative examples for softmax.

            if slot != "request":
                for value_idx, value in enumerate(dialogue_ontology[slot]):
                    if (slot, value) in utterance[4]:  # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                        positive_examples[slot].append((utterance_idx, utterance, value_idx))
                    else:
                        if not slot_expressed_in_utterance:
                            negative_examples[slot].append((utterance_idx, utterance, value_idx))
            elif slot == "request":
                if not slot_expressed_in_utterance:
                    negative_examples[slot].append((utterance_idx, utterance, []))
                    # print utterance[0][0], utterance[4]
                else:
                    values_expressed = []
                    for value_idx, value in enumerate(dialogue_ontology[slot]):
                        if (slot, value) in utterance[4]:  # utterances are ((trans, asr), sys_req_act, sys_conf, labels)
                            values_expressed.append(value_idx)
                    positive_examples[slot].append((utterance_idx, utterance, values_expressed))
    return positive_examples, negative_examples
# -*- coding: utf-8 -*-
import sys
from Managerv2 import CrossNeuralBeliefTracker

global_var_asr_count = 1

def main():
    try: 
        config_filepath = sys.argv[2]
    except:
        print "Config not specified."  
        return

    do_training = False
    do_test = False
    do_sent_transfer = False
    do_word_transfer = False
    do_transfer_test = False

    try: 
        switch = sys.argv[1]
        if switch == "train":
            do_training = True
        elif switch == "test":
            do_test = True
        elif switch == 'corpus_transfer':
            do_sent_transfer = True
        elif switch == 'dict_transfer':
            do_word_transfer = True
        elif switch == 'transfer_test':
            do_transfer_test = True
        else:
            print "No such task"
    except:
        print "Training/Testing not specified, defaulting to input testing."  

    NBT = CrossNeuralBeliefTracker(config_filepath)
    if do_training:
        print "building NBT"
        NBT.train()
    elif do_test:
        print "test NBT"
        NBT.test()
    elif do_sent_transfer:
        print "building corpus transfer NBT"
        NBT.transfer_corpus()
    elif do_word_transfer:
        print "building dictionary transfer NBT"
        NBT.transfer_dict()
    elif do_transfer_test:
        print "test transfer NBT"
        NBT.test_foreign()
    else:
        print "No such task"

if __name__ == "__main__":
    main()              


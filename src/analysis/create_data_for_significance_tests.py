import os
import sys
import random
import argparse
import _pickle as cPickle
from random import choices
random.seed(1)

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

sys.path.append("/src/shared/")

from eval_utils import *
from classes import *

"""
--test_A_path {baseline model results}/test_topics
--test_B_path {comparison model results}/test_topics
--measure evaluation measure score to compare
--out_dir src/analysis/data1
"""

parser = argparse.ArgumentParser(description='Creating data for statistical significance tests')

parser.add_argument('--test_A_path', type=str,
                    help=' The path to the test corpus of the full model '
                         '(which includes the mentions with their predicted coreference chains)')
parser.add_argument('--test_B_path', type=str,
                    help='  The path to the test corpus of the disjoint model or another model ablation '
                         '(which includes the mentions with their predicted coreference chains)')
parser.add_argument('--measure', type=str,
                    help=' To which measure save the f1 scores (MUC\Bcubed\CEAFe\CoNLL)')
parser.add_argument('--out_dir', type=str,
                    help='Output folder')

args = parser.parse_args()


def sample_test_topics():
    """
    This function samples 1000 topics combinations and writes the coreference clusters of those topic
     of system A and system B (in this case, system A is the joint model and
    system B is the disjoint model) into a files in CoNLL format.
    """

    test_A_path = args.test_A_path # baseline model test corpus object
    test_B_path = args.test_B_path # chirps* features model test corpus object

    print('Loading predictions files...')

    with open(test_A_path, 'rb') as f:
        test_data_a = cPickle.load(f)
    with open(test_B_path, 'rb') as f:
        test_data_b = cPickle.load(f)

    print('Finish loading predictions files.')
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    with open(os.path.join(args.out_dir, 'scorer_commands'), 'w') as f:
        for i in range(1000):
            print('Create test sample {}'.format(i))
            topic_keys = list(test_data_a.keys())
            sampled_keys = choices(topic_keys, k=20)

            corpus_a = Corpus()
            corpus_b = Corpus()
            for key in sampled_keys:
                topic_a = test_data_a[key]
                corpus_a.add_topic(key, topic_a)
                topic_b = test_data_b[key]
                corpus_b.add_topic(key, topic_b)

            out_dir = os.path.join(args.out_dir, 'run_{}'.format(i))
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            gold_out_file = os.path.join(out_dir, 'CD_test_event_mention_based_{}.key_conll'.format(i))
            write_mention_based_cd_clusters(corpus_a, is_event=True, is_gold=True, out_file=gold_out_file)

            a_out_file = os.path.join(out_dir, 'CD_test_event_mention_based_{}_{}.response_conll'.format('A', i))
            write_mention_based_cd_clusters(corpus_a, is_event=True, is_gold=False, out_file=a_out_file)

            b_out_file = os.path.join(out_dir, 'CD_test_event_mention_based_{}_{}.response_conll'.format('B', i))
            write_mention_based_cd_clusters(corpus_b, is_event=True, is_gold=False, out_file=b_out_file)

            conll_file_a = os.path.join(out_dir, 'conll_a_{}'.format(i))
            conll_file_b = os.path.join(out_dir, 'conll_b_{}'.format(i))
            f.write('perl scorer/scorer.pl all {} {} none > {} \n'.format
                    (gold_out_file, a_out_file, conll_file_a))
            f.write('perl scorer/scorer.pl all {} {} none > {} \n'.format
                    (gold_out_file, b_out_file, conll_file_b))


def run_scorers():
    """
    This function runs the CoNLL scorer for each topics combination.
    """
    import subprocess
    parallel_tasks = 40
    tasks = []
    with open(os.path.join(args.out_dir,'scorer_commands'), "r") as ins:
        for line in ins:
            tasks.append(line.strip())
    processes = []
    tasks_count = 0
    tasks_len = len(tasks)

    for i in range(parallel_tasks):
        print('Run command {}'.format(i))
        processes.append(subprocess.Popen(tasks[tasks_count], shell=True))
        tasks_count += 1

    while tasks_count < tasks_len:
        i = 0
        status = None
        while status is None:
            status = processes[i].poll()
            i += 1
            if i == parallel_tasks:
                if status is None:
                    i = 0
        if status is not None:
            print('Run command {}'.format(tasks_count))
            processes[i - 1] = subprocess.Popen(tasks[tasks_count], shell=True)
            tasks_count += 1


def read_conll_f1(filename, measure):
    """
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUC,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :param measure: the measure scores to return - MUC\Bcubed\CEAFe\CoNLL
    :return: the measure score
    """
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    scores = {}
    scores['muc_f1'] = f1_list[1]
    scores['bcubed_f1'] = f1_list[3]
    scores['ceafe_f1'] = f1_list[7]
    scores['conll_f1'] = (scores['muc_f1'] + scores['bcubed_f1'] + scores['ceafe_f1'])/float(3)
    return scores[measure.lower() + '_f1']


def parse_scorer_output():
    """
    This function reads the results of the CoNLL scorer for all the 1000 topics combinations
    (for both system A and system B), extracts the F1 measures of the MUS, B-cubed and the CEAF-e, calculates CoNLL F1 scores
    and write them to a file.
    :return:
    """
    a_scores_file = open(os.path.join(args.out_dir, 'a_scores_{}.txt'.format(args.measure.lower())), 'w')
    b_scores_file = open(os.path.join(args.out_dir, 'b_scores_{}.txt'.format(args.measure.lower())), 'w')
    for i in range(1000):
        out_dir = os.path.join(args.out_dir, 'run_{}'.format(i))
        conll_file_a = os.path.join(out_dir, 'conll_a_{}'.format(i))
        conll_file_b = os.path.join(out_dir, 'conll_b_{}'.format(i))

        a_f1 = read_conll_f1(conll_file_a, args.measure)
        b_f1 = read_conll_f1(conll_file_b, args.measure)
        a_scores_file.write('{}\n'.format(a_f1))
        b_scores_file.write('{}\n'.format(b_f1))

    a_scores_file.close()
    b_scores_file.close()


def test_significance():
    """
    Runs the whole process of creating the results for statistical significance tests.
    """
    #sample_test_topics()
    #run_scorers()
    parse_scorer_output()


def main():
    """
    This script runs the whole process of creating the results for statistical significance tests,
    which includes sampling of 1000 topics combinations, extracting the results of system A and system
    B for those combinations, running CoNLL scorer for each system in each topics combination
    and extracting the CoNLL results.
    :return:
    """
    test_significance()


if __name__ == '__main__':
    main()

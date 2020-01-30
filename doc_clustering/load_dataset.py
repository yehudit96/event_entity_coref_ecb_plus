# -*- coding: utf-8 -*-
import os
import _pickle as cPickle
import logging
import argparse
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')

parser.add_argument('--ecb_path', type=str,
                    help=' The path to the ECB+ corpus')
parser.add_argument('--output_dir', type=str,
                    help=' The directory of the output files')
parser.add_argument('--split', type=str,
                    help=' The split that should be loaded - train|dev|test')


args = parser.parse_args()

VALIDATION = [2, 5, 12, 18, 21, 23, 34, 35]
TRAIN = [i for i in range(1, 36) if i not in VALIDATION]
TEST = [i for i in range(36, 46)]

class Token(object):
    def __init__(self, text, sent_id, tok_id, rel_id=None):
        '''

        :param text: The token text
        :param sent_id: The sentence id
        :param tok_id: The token id
        :param rel_id: The relation id
        '''

        self.text = text
        self.sent_id = sent_id
        self.tok_id = tok_id
        self.rel_id = rel_id


def load_ecb_plus_raw_doc(doc_filename, doc_id):
    ecb_file = open(doc_filename, 'r')
    tree = ET.parse(ecb_file)
    root = tree.getroot()

    tokens = []
    for token in root.findall('token'):
        if 'plus' in doc_id and int(token.attrib['sentence']) == 0:
            continue
        tokens.append(token.text)

    return ' '.join(tokens)


def load_raw_test_data():
    if 'dev' == args.split:
        test_topics = VALIDATION
    elif 'train' in args.split:
        test_topics = TRAIN
    else:
        test_topics = TEST
    dirs = os.listdir(args.ecb_path)
    docs = {}

    for dir in dirs:
        dir_path = os.path.join(args.ecb_path, dir)
        if os.path.isfile(dir_path):
            continue
        doc_files = os.listdir(os.path.join(args.ecb_path, dir))
        for doc_file in doc_files:
            doc_id = doc_file.replace('.xml', '')
            if int(doc_id.split('_')[0]) in test_topics:
                xml_filename = os.path.join(os.path.join(args.ecb_path, dir),doc_file)
                raw_doc = load_ecb_plus_raw_doc(xml_filename, doc_id)
                docs[doc_id] = raw_doc

    return docs


def main():
    test_docs = load_raw_test_data()
    with open(os.path.join(args.output_dir, '{}_raw_docs'.format(args.split)), 'wb') as f:
        cPickle.dump(test_docs, f)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()

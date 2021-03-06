from models import *
from model_utils import *
import torch.nn as nn
import torch.optim as optim
import logging

import sys
sys.path.append('src/shared')

from coreferability import get_corefferabiliy_dim

word_embeds = None
word_to_ix = None
char_embeds = None
char_to_ix = None

'''
All functions in this script requires a configuration dictionary which contains flags and 
other attributes for configuring the experiments.
In this project, the configuration dictionaries are stored as JSON files (e.g. train_config.json)
and are loaded before the training/inference starts.

'''


def factory_load_embeddings(config_dict):
    '''
    Given a configuration dictionary, containing the paths to the embeddings files,
    this function loads the initial character embeddings and pre-trained word embeddings.
    :param config_dict: s configuration dictionary
    '''
    global word_embeds, word_to_ix, char_embeds, char_to_ix
    word_embeds, word_to_ix, char_embeds, char_to_ix = load_model_embeddings(config_dict)


def create_model(config_dict, is_event):
    '''
    Given a configuration dictionary, containing flags for configuring the current experiment,
    this function creates a model according to those flags and returns that model.
    :param config_dict: a configuration dictionary
    :return: an initialized model - CDCorefScorer object
    '''
    global word_embeds, word_to_ix, char_embeds, char_to_ix

    context_vector_size = 1024

    if config_dict["use_args_feats"]:
        mention_rep_size = context_vector_size + \
                            ((word_embeds.shape[1] + config_dict["char_rep_size"]) * 5)
    else:
        mention_rep_size = context_vector_size + word_embeds.shape[1] + config_dict["char_rep_size"]

    if config_dict["use_paraphrase"] and is_event:
        mention_rep_size += word_embeds.shape[1]
    
    input_dim = mention_rep_size * 3

    if config_dict["use_binary_feats"]:
        input_dim += 4 * config_dict["feature_size"]
    
    if config_dict["coreferability"] != 'None' and (is_event or (not is_event and config_dict['entity_coref'])):
        if config_dict["coreferability"] == 'attention':
            input_dim = input_dim + config_dict["attention_hidden_size"]# for the rule score feature
        else:
            input_dim = input_dim + get_corefferabiliy_dim(config_dict["coreferability"])  # for the rule score feature

    print("input dim {}".format(input_dim))
    second_dim = int(input_dim / 2)
    third_dim = second_dim

    if config_dict["coreferability"] == 'linear' and (is_event or (not is_event and config_dict['entity_coref'])):
        coref_input_dim = 17
        coref_second_dim = 50
        coref_third_dim = 100

        model_dims = [input_dim, second_dim, third_dim, coref_input_dim, coref_second_dim, coref_third_dim]

    else:
        model_dims = [input_dim, second_dim, third_dim]

    coreferability_type = config_dict["coreferability"]\
        if is_event or (not is_event and config_dict['entity_coref']) \
        else 'None'

    model = CDCorefScorer(word_embeds, word_to_ix, word_embeds.shape[0],
                          char_embedding=char_embeds, char_to_ix=char_to_ix,
                          char_rep_size=config_dict["char_rep_size"],
                          dims=model_dims,
                          use_mult=config_dict["use_mult"],
                          use_diff=config_dict["use_diff"],
                          feature_size=config_dict["feature_size"],
                          coreferability_type=coreferability_type,
                          atten_hidden_size=config_dict["attention_hidden_size"])

    return model


def create_optimizer(config_dict, model):
    '''
    Given a configuration dictionary, containing the string attribute "optimizer" that determines
    in which optimizer to use during the training.
    :param config_dict: a configuration dictionary
    :param model: an initialized CDCorefScorer object
    :return: Pytorch optimizer
    '''
    lr = config_dict["lr"]
    optimizer = None
    parameters = filter(lambda p: p.requires_grad,model.parameters())
    if config_dict["optimizer"] == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr,
                                   weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=config_dict["momentum"],
                              nesterov=True)

    assert (optimizer is not None), "Config error, check the optimizer field"

    return optimizer


def create_loss(config_dict):
    '''
    Given a configuration dictionary, containing the string attribute "loss" that determines
    in which loss function to use during the training.
    :param config_dict: a configuration dictionary
    :param model: an initialized CDCorefScorer object
    :return: Pytorch loss function

    '''
    loss_function = None

    if config_dict["loss"] == 'bce':
        loss_function = nn.BCELoss()

    assert (loss_function is not None), "Config error, check the loss field"

    return loss_function


def load_model_embeddings(config_dict):
    '''
    Given a configuration dictionary, containing the paths to the embeddings files,
    this function loads the initial character embeddings and pre-trained word embeddings.
    :param config_dict: s configuration dictionary
    '''
    logging.info('Loading word embeddings...')

    # load pre-trained word embeddings
    vocab, embd = loadGloVe(config_dict["glove_path"])
    word_embeds = np.asarray(embd, dtype=np.float64)

    i = 0
    word_to_ix = {}
    for word in vocab:
        if word in word_to_ix:
            continue
        word_to_ix[word] = i
        i += 1

    logging.info('Word embeddings have been loaded.')

    if config_dict["use_pretrained_char"]:
        logging.info('Loading pre-trained char embeddings...')
        char_embeds, vocab = load_embeddings(config_dict["char_pretrained_path"],
                                             config_dict["char_vocab_path"])

        char_to_ix = {}
        for char in vocab:
            char_to_ix[char] = len(char_to_ix)

        char_to_ix[' '] = len(char_to_ix)
        space_vec = np.zeros((1, char_embeds.shape[1]))
        char_embeds = np.append(char_embeds, space_vec, axis=0)

        char_to_ix['<UNK>'] = len(char_to_ix)
        unk_vec = np.random.rand(1, char_embeds.shape[1])
        char_embeds = np.append(char_embeds, unk_vec, axis=0)

        logging.info('Char embeddings have been loaded.')
    else:
        logging.info('Loading one-hot char embeddings...')
        char_embeds, char_to_ix = load_one_hot_char_embeddings(config_dict["char_vocab_path"])

    return word_embeds, word_to_ix, char_embeds, char_to_ix
import _pickle as cPickle
from nltk.stem import WordNetLemmatizer
import numpy as np
from enum import Enum
lemmatizer = WordNetLemmatizer()

rules_score_path = 'data/external/coreferability/rf_rules_scores'
an_rules_score_path = 'data/external/coreferability/annotated_rules_scores'
vectors_path = 'data/external/coreferability/vectors'
top_5_paraphrases = 'data/external/coreferability/rules_dict_top_5'

with open(an_rules_score_path, 'rb') as rf:
    an_rules_score = cPickle.load(rf)

with open(rules_score_path, 'rb') as rf:
    rules_score = cPickle.load(rf)

with open(top_5_paraphrases, 'rb') as rf:
    predicate_top_paraphrases = cPickle.load(rf)
    
    
#avg = 0.11507013912825699 #average of only train event pairs
#an_avg = 0.6383347788378144

avg = 0.43107517809968393

with open(vectors_path, 'rb') as f:
    vectors = cPickle.load(f)

corefferabiliy_dim = 100
linear_dim = 100
feature_vector_dim = 17
np.random.seed(seed=48)
mat = np.random.rand(linear_dim, feature_vector_dim)


class CoreferabilityType(Enum):
    none = 1
    linear = 2
    score = 3
    an_score = 4
    bidding = 5
    bidding_fill = 6
    an_bidding_fill = 7
    attention = 8


coreferability_dims = {'none': 0,
                       'linear': 100,
                       'score': 1,
                       'an_score': 1,
                       'bidding': 100,
                       'bidding_fill': 100,
                       'an_bidding_fill': 100,
                       'attention': 75}  # number of features * transformer hidden size

default_vector_size = 17
def mention_to_lemma(mention):
    return lemmatize(mention.mention_str.lower())

    
def lemmatize(words):
    return ' '.join(map(lambda x: lemmatizer.lemmatize(x, 'v'), words.split()))
    
    
def get_corefferabiliy_dim(coref_type):
    if coref_type.lower() not in [c.name for c in CoreferabilityType]:
        raise Exception('the system doesn\'t support {} coref type'.format(coref_type))

    return coreferability_dims[coref_type.lower()]

    if 'bidding' in coref_type:
        return corefferabiliy_dim
    elif coref_type == 'None':
        return 0
    elif coref_type == 'linear':
        return linear_dim
    else: #'score', 'an_score'
      return 1


def get_pair_score(mention1, mention2, coref_type):
    lemma1 = mention_to_lemma(mention1)
    lemma2 = mention_to_lemma(mention2)
    rule = '_'.join(sorted([lemma1, lemma2]))

    # if the mention pair has a representation by Chirps
    if rule in rules_score:
        if coref_type == 'an_score' or coref_type == 'an_bidding_fill':
            score = an_rules_score[rule]
        else:
            score = rules_score[rule]
        vector = vectors[rule]
    # if the mention pair doesn't have a representation in Chirps assign the default scores\features
    else:
        if coref_type == 'an_score' or coref_type == 'an_bidding_fill':
            score = an_avg
        else:
            score = avg
        if coref_type == 'linear':
            vector = np.zeros(default_vector_size)
        elif coref_type == 'attention':
            vector = np.ones(default_vector_size) * -1

    if coref_type in ['bidding', 'bidding_fill', 'an_bidding_fill']:
        emb = np.zeros(corefferabiliy_dim)
        if coref_type in ['bidding_fill', 'an_bidding_fill']:
            emb[:int(score*corefferabiliy_dim)] = 1
        else:
            emb[int(score*corefferabiliy_dim)] = 1
        return emb.reshape(1, corefferabiliy_dim)
    elif coref_type in ['score', 'an_score']:
        return np.array([score]).reshape(1, 1)
    elif coref_type in ['attention', 'linear']:
        return vector.reshape(1, -1)

def has_rule(mention1, mention2):
    lemma1 = mention_to_lemma(mention1)
    lemma2 = mention_to_lemma(mention2)
    rule = '_'.join(sorted([lemma1, lemma2]))
    return rule in rules_score
    

def get_paraphrases(predicate):
    if predicate.lower() in predicate_top_paraphrases:
        return [m_predicate for m_predicate, score in predicate_top_paraphrases[predicate.lower()]] 
    return []
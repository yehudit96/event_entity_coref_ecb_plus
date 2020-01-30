import _pickle as cPickle
from nltk.stem import WordNetLemmatizer
import numpy as np

lemmatizer = WordNetLemmatizer()

rules_score_path = 'data/external/coreferability/rules_scores'
vectors_path = 'data/external/coreferability/vectors'

with open(rules_score_path, 'rb') as rf:
    rules_score = cPickle.load(rf)

avg = 0.11507013912825699 #average of only train event pairs

with open(vectors_path, 'rb') as f:
    vectors = cPickle.load(f)

corefferabiliy_dim = 100
linear_dim = 100
feature_vector_dim = 17
np.random.seed(seed=48)
mat = np.random.rand(linear_dim, feature_vector_dim)


def mention_to_lemma(mention):
    return lemmatize(mention.mention_str.lower())

    
def lemmatize(words):
    return ' '.join(map(lambda x: lemmatizer.lemmatize(x, 'v'), words.split()))
    
    
def get_corefferabiliy_dim(coref_type):
    if 'bidding' in coref_type:
        return corefferabiliy_dim
    elif coref_type == 'None':
        return 0
    elif coref_type == 'linear':
        return linear_dim
    else: #'score'
      return 1


def get_pair_score(mention1, mention2, coref_type):
    lemma1 = mention_to_lemma(mention1)
    lemma2 = mention_to_lemma(mention2)
    rule = '_'.join(sorted([lemma1, lemma2]))
    if rule in rules_score:
        score = rules_score[rule]
        vector = vectors[rule]
    else:
        score = avg
        vector = np.zeros(feature_vector_dim)

    if 'bidding' in coref_type:
        emb = np.zeros(corefferabiliy_dim)
        if coref_type == 'bidding_fill':
            emb[:int(score*corefferabiliy_dim)] = 1
        else:
            emb[int(score*corefferabiliy_dim)] = 1
        return emb.reshape(1, corefferabiliy_dim)
    elif coref_type == 'score':
        return np.array([score]).reshape(1, 1)
    else:
        return vector.reshape(1, -1)

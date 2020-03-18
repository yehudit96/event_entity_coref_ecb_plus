import math
import numpy as np
import torch
import torch.nn as nn
from model_utils import *
import torch.nn.functional as F
import torch.autograd as autograd


class CDCorefScorer(nn.Module):
    '''
    An abstract class represents a coreference pairwise scorer.
    Inherits Pytorch's Module class.
    '''

    def __init__(self, word_embeds, word_to_ix, vocab_size, char_embedding, char_to_ix, char_rep_size,
                 dims, use_mult, use_diff, feature_size, coreferability_type, atten_hidden_size=None):
        '''
        C'tor for CorefScorer object
        :param word_embeds: pre-trained word embeddings
        :param word_to_ix: a mapping between a word (string) to
        its index in the word embeddings' lookup table
        :param vocab_size:  the vocabulary size
        :param char_embedding: initial character embeddings
        :param char_to_ix:  mapping between a character to
        its index in the character embeddings' lookup table
        :param char_rep_size: hidden size of the character LSTM
        :param dims: list holds the layer dimensions
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the
        input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the
        input layer
        :param feature_size: embeddings size of binary features


        '''
        super(CDCorefScorer, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_embeds.shape[1])

        self.embed.weight.data.copy_(torch.from_numpy(word_embeds))
        self.embed.weight.requires_grad = False  # pre-trained word embeddings are fixed
        self.word_to_ix = word_to_ix

        self.char_embeddings = nn.Embedding(len(char_to_ix.keys()), char_embedding.shape[1])
        self.char_embeddings.weight.data.copy_(torch.from_numpy(char_embedding))
        self.char_embeddings.weight.requires_grad = True
        self.char_to_ix = char_to_ix
        self.embedding_dim = word_embeds.shape[1]
        self.char_hidden_dim = char_rep_size

        self.char_lstm = nn.LSTM(input_size=char_embedding.shape[1], hidden_size=self.char_hidden_dim, num_layers=1,
                                 bidirectional=False)

        self.coreferability_type = coreferability_type

        #  binary features for coreferring arguments/predicates
        self.coref_role_embeds = nn.Embedding(2, feature_size)

        self.use_mult = use_mult
        self.use_diff = use_diff
        self.input_dim = dims[0]
        self.hidden_dim_1 = dims[1]
        self.hidden_dim_2 = dims[2]
        self.out_dim = 1

        self.hidden_layer_1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out_layer = nn.Linear(self.hidden_dim_2, self.out_dim)

        if self.coreferability_type == 'linear':
            self.coref_input_dim = dims[3]
            self.coref_second_dim = dims[4]
            self.coref_third_dim = dims[5]
            self.hidden_layer_coref_1 = nn.Linear(self.coref_input_dim, self.coref_second_dim)
            self.hidden_layer_coref_2 = nn.Linear(self.coref_second_dim, self.coref_third_dim)
            self.dropout_coref = nn.Dropout(p=0.2)

        elif self.coreferability_type == 'attention':
            self.trasformer = FeaturesSelfAttention(vocab_size=20001, hidden_size=atten_hidden_size)
            self.attention_features = [0, 3, 5, 6] + list(range(8, 17))
        self.model_type = 'CD_scorer'

    def forward(self, clusters_pair_tensor):
        '''
        The forward method - pass the input tensor through a feed-forward neural network
        :param clusters_pair_tensor: an input tensor consists of a concatenation between
        two mention representations, their element-wise multiplication and a vector of binary features
        (each feature embedded as 50 dimensional embeddings)
        :return: a predicted confidence score (between 0 to 1) of the mention pair to be in the
        same coreference chain (aka cluster).
        '''
        if self.coreferability_type == 'linear':
            coref_features = clusters_pair_tensor[:, :17]
            coref_first_hidden = F.relu(self.hidden_layer_coref_1(coref_features))
            coref_second_hidden = F.relu(self.hidden_layer_coref_2(coref_first_hidden))
            coref_dropout = self.dropout_coref(coref_second_hidden)

            clusters_tensor = torch.cat([clusters_pair_tensor[:, 17:], coref_dropout], dim=1)

        elif self.coreferability_type == 'attention':
            coref_features = clusters_pair_tensor[:, :17]
            features_vector = coref_features[:, self.attention_features]
            #features_vector = features_vector.type(torch.IntTensor)

            attention = self.trasformer(features_vector)
            clusters_tensor = torch.cat([clusters_pair_tensor[:, 17:], attention], dim=1)
        else:
            clusters_tensor = clusters_pair_tensor

        first_hidden = F.relu(self.hidden_layer_1(clusters_tensor))
        # first_hidden = F.relu(self.hidden_layer_1(clusters_pair_tensor))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = F.sigmoid(self.out_layer(second_hidden))

        return out

    def init_char_hidden(self, device):
        '''
        initializes hidden states the character LSTM
        :param device: gpu/cpu Pytorch device
        :return: initialized hidden states (tensors)
        '''
        return (torch.randn((1, 1, self.char_hidden_dim), requires_grad=True).to(device),
                torch.randn((1, 1, self.char_hidden_dim), requires_grad=True).to(device))

    def get_char_embeds(self, seq, device):
        '''
        Runs a LSTM on a list of character embeddings and returns the last output state
        :param seq: a list of character embeddings
        :param device:  gpu/cpu Pytorch device
        :return: the LSTM's last output state
        '''
        char_hidden = self.init_char_hidden(device)
        input_char_seq = self.prepare_chars_seq(seq, device)
        char_embeds = self.char_embeddings(input_char_seq).view(len(seq), 1, -1)
        char_lstm_out, char_hidden = self.char_lstm(char_embeds, char_hidden)
        char_vec = char_lstm_out[-1]

        return char_vec

    def prepare_chars_seq(self, seq, device):
        '''
        Given a string represents a word or a phrase, this method converts the sequence
        to a list of character embeddings
        :param seq: a string represents a word or a phrase
        :param device: device:  gpu/cpu Pytorch device
        :return: a list of character embeddings
        '''
        idxs = []
        for w in seq:
            if w in self.char_to_ix:
                idxs.append(self.char_to_ix[w])
            else:
                lower_w = w.lower()
                if lower_w in self.char_to_ix:
                    idxs.append(self.char_to_ix[lower_w])
                else:
                    idxs.append(self.char_to_ix['<UNK>'])
                    print('can find char {}'.format(w))
        tensor = torch.tensor(idxs, dtype=torch.long).to(device)

        return tensor


max_value = {
    'NE_0.26': 3400,
    'chirps_days': 1500,
    'chirps_num': 20000,
    'chirps_rules_num': 20,
    'component_num': 1500,
    'day_num': 600,
    'entity_ipc': 2000,
    'entity_pc': 7000,
    'entity_wc': 1000,
    'event_ipc': 9000,
    'event_pc': 7000,
    'in_clique': 5000,
    'pairs_num': 9000
}


class FeaturesSelfAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_attention_heads=1, attention_probs_dropout_prob=0.1,
                 hidden_dropout_probs=0.1):
        super(FeaturesSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.numerical_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=vocab_size-1)
        self.features_embedding = nn.Embedding(13, hidden_size)
        # self.embeddings = {i: nn.Embedding(dim, hidden_size) for i, dim in enumerate(max_value.values())}
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)
        # self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.device = torch.cuda.current_device()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        embedding = self.create_features_embedding(hidden_states)
        mixed_query_layer = self.query(embedding)
        mixed_key_layer = self.key(embedding)
        mixed_value_layer = self.value(embedding)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        batch_vectors = torch.mean(context_layer, dim=1)

        return batch_vectors

    def create_features_embedding(self, features_vector):
        batch_vectors = []
        for row in features_vector:
            row_features = []
            for feature_inx, feature_val in enumerate(row):
                feature_val = int(feature_val) if feature_val > -1 else self.numerical_embedding.num_embeddings-1
                feature_val = torch.tensor(feature_val).to(self.device)
                feature_inx = torch.tensor(feature_inx).to(self.device)
                feature_tensor = (self.numerical_embedding(feature_val) + self.features_embedding(feature_inx))/2
                feature_tensor = feature_tensor.reshape(1, -1)
                row_features.append(feature_tensor)
            batch_vectors.append(torch.cat(row_features, dim=-2).unsqueeze(0))
        return torch.cat(batch_vectors, dim=0)

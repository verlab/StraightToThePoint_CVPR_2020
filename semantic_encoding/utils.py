import os
import numpy as np
import nltk
import torch
import torch.hub
import torch.nn as nn
from datetime import datetime
from pycocotools.coco import COCO
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))
from models import VDAN

def load_embeddings_matrix(embeddings_file, embeddings_dim, use_fake_embeddings=False):

    # Creating representation for PAD and UNK
    if use_fake_embeddings:
        vocabulary = np.concatenate([['PAD'], ['UNK']])
        vocab_size = len(vocabulary)

        word_map = {k: v for v, k in enumerate(vocabulary)}

        print('[{}] Loading fake word embeddings to speed-up debugging...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        embeddings = np.random.random((vocab_size, embeddings_dim)).astype(np.float32)
    else:
        print('[{}] Loading word embeddings and vocab...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        embeddings = []
        vocab = []
        embeddings.append(np.zeros(embeddings_dim))  # PAD vector
        embeddings.append(np.random.random(embeddings_dim))  # UNK vector (unknown word)
        f = open(embeddings_file, 'r', encoding='utf-8')
        for idx, line in enumerate(tqdm(f)):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            vocab.append(word)
            embeddings.append(embedding)
        embeddings = np.array(embeddings, dtype=np.float32)

        vocabulary = np.concatenate([['PAD'], ['UNK'], vocab])
        vocab_size = len(vocabulary)

        word_map = {k: v for v, k in enumerate(vocabulary)}

    embeddings = torch.from_numpy(embeddings)

    print('[{}] Done!'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    return embeddings, word_map


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def init_weights(m):
    np.random.seed(123456)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def save_checkpoint(epoch, model, optimizer, word_map, datetimestamp, model_params, train_params):
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'word_map': word_map,
             'model_params': model_params,
             'train_params': train_params}

    if not os.path.isdir(train_params['checkpoint_folder']):
        print('Folder "{}" does not exist. We are attempting creating it... '.format(train_params['checkpoint_folder']))
        os.mkdir(train_params['checkpoint_folder'])
        print('Folder created!')

    if train_params['finetune_semantic_model']:
        filename = '{}/{}_checkpoint_lr{}_{}eps{}{}{}_{}_{}_ft{}.pth'.format(train_params['checkpoint_folder'], datetimestamp, train_params['learning_rate'], train_params['num_epochs'], '_w-att' if model_params['use_word_level_attention'] else '', '_s-att' if model_params['use_sentence_level_attention'] else '',
                                                                             '_lrdecay{}'.format(train_params['learning_rate_decay']) if train_params['learning_rate_decay'] is not None else '', train_params['hostname'], train_params['username'], train_params['model_checkpoint_filename'].split('/')[-1].split('.')[0])
    else:
        filename = '{}/{}_checkpoint_lr{}_{}eps{}{}{}_{}_{}.pth'.format(train_params['checkpoint_folder'], datetimestamp, train_params['learning_rate'], train_params['num_epochs'], '_w-att' if model_params['use_word_level_attention'] else '', '_s-att' if model_params['use_sentence_level_attention'] else '', '_lrdecay{}'.format(train_params['learning_rate_decay']) if train_params['learning_rate_decay'] is not None else '', train_params['hostname'], train_params['username'])

    print('\t[{}] Saving checkpoint file for epoch {}: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, filename))
    torch.save(state, filename)
    print('\t[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def load_checkpoint(filename):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filename)

    epoch = checkpoint['epoch']
    word_map = checkpoint['word_map']
    model_params = checkpoint['model_params']
    train_params = checkpoint['train_params']
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']

    vocab_size = len(word_map)

    model = VDAN(vocab_size=vocab_size,
                       doc_emb_size=model_params['doc_embed_size'],
                       sent_emb_size=model_params['sent_embed_size'],
                       word_emb_size=model_params['word_embed_size'],
                       hidden_feat_emb_size=model_params['hidden_feat_size'],
                       final_feat_emb_size=model_params['feat_embed_size'],
                       sent_rnn_layers=model_params['sent_rnn_layers'],
                       word_rnn_layers=model_params['word_rnn_layers'],
                       sent_att_size=model_params['sent_att_size'],
                       word_att_size=model_params['word_att_size'],
                       use_visual_shortcut=model_params['use_visual_shortcut'],
                       use_sentence_level_attention=model_params['use_sentence_level_attention'],
                       use_word_level_attention=model_params['use_word_level_attention'])

    # Init word embeddings layer with random embeddings
    # model.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(torch.rand(vocab_size, model_params['word_embed_size']))
    model.load_state_dict(model_state_dict)

    return epoch, model, optimizer_state_dict, word_map, model_params, train_params


def convert_sentences_to_word_idxs(sentences, max_words, word_map):
    converted_sentences = np.zeros((len(sentences), max_words), dtype=int)
    words_per_sentence = np.zeros((len(sentences),), dtype=int)
    for aid, annotation in enumerate(sentences):
        tokenized_annotation = nltk.tokenize.word_tokenize(annotation.lower())
        for wid, word in enumerate(tokenized_annotation[:min(len(tokenized_annotation), max_words)]):
            if word in word_map:
                converted_sentences[aid, wid] = word_map[word]
            else:
                converted_sentences[aid, wid] = word_map['UNK']

            words_per_sentence[aid] += 1  # Increment number of words

    return converted_sentences, words_per_sentence


def get_all_coco_annotations(annotations_file):
    coco = COCO(annotations_file)
    annotations = coco.getAnnIds()

    return annotations


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from models import VDAN
from utils import *
from coco_captions_dataset import CocoCaptionsDataset
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import time
import socket
import getpass
import multiprocessing
import argparse
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

WORKERS = int(0.9*multiprocessing.cpu_count())  # number of workers for loading data in the DataLoader
PRINT_FREQ = 100  # print training or validation status every __ batches

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def write_batch_to_log(writer, root_path, imgs_paths, documents, i_epoch):
    img = mpimg.imread(os.path.join(root_path, imgs_paths[0]))

    fig, axs = plt.subplots(2, 1, figsize=(10, 20))
    axs[0].imshow(img)
    axs[1].text(0, 0, '.\n '.join([doc[0] for doc in documents]), wrap=True)

    writer.add_figure('img_{}'.format(0), fig, i_epoch)
    writer.add_text('img_{}'.format(0), '. '.join([doc[0] for doc in documents]), i_epoch)


def create_sets(word_map, train_params):
    if train_params['do_random_horizontal_flip']:
        train_transform = T.Compose([T.Resize(train_params['resize_size']),
                                     T.RandomCrop(train_params['random_crop_size']),
                                     T.RandomHorizontalFlip(),
                                     T.ToTensor(),
                                     T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    else:
        train_transform = T.Compose([T.Resize(train_params['resize_size']),
                                     T.RandomCrop(train_params['random_crop_size']),
                                     T.ToTensor(),
                                     T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    val_transform = T.Compose([T.Resize(train_params['resize_size']),
                               T.CenterCrop(train_params['random_crop_size']),
                               T.ToTensor(),
                               T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    # Data location and settings
    training_data = CocoCaptionsDataset(root=train_params['train_data_path'],
                                        annFile=train_params['captions_train_fname'],
                                        word_map=word_map,
                                        img_transform=train_transform,
                                        annotations_transform=T.ToTensor(),
                                        num_sentences=train_params['max_sents'],
                                        max_words=train_params['max_words'],
                                        dataset_proportion=train_params['train_data_proportion'])

    validation_data = CocoCaptionsDataset(root=train_params['val_data_path'],
                                          annFile=train_params['captions_val_fname'],
                                          word_map=word_map,
                                          img_transform=val_transform,
                                          annotations_transform=T.ToTensor(),
                                          num_sentences=train_params['max_sents'],
                                          max_words=train_params['max_words'],
                                          dataset_proportion=train_params['val_data_proportion'])

    # Data loaders
    training_dataloader = torch.utils.data.DataLoader(training_data,
                                                      batch_size=train_params['train_batch_size'],
                                                      num_workers=WORKERS,
                                                      shuffle=True)

    validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=train_params['val_batch_size'],
                                                        num_workers=WORKERS,
                                                        shuffle=False)

    return training_dataloader, validation_dataloader, training_data, validation_data


def train(training_dataloader, training_data, model, criterion, optimizer, epoch, writer):
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss

    start = time.time()

    num_batches = len(training_dataloader)
    # Batches
    for i, (imgs_paths, captions_docs, imgs, documents, sentences_per_document, words_per_sentence, labels) in enumerate(training_dataloader):

        data_time.update(time.time() - start)

        imgs = imgs.to(device)

        # pdb.set_trace()
        documents = documents.squeeze(1).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        imgs_embeddings, texts_embeddings, word_alphas, sentence_alphas = model(imgs, documents, sentences_per_document, words_per_sentence)

        # Loss
        loss = criterion(imgs_embeddings, texts_embeddings, labels)  # scalar

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if train_params['grad_clip'] is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print training status
        if i % PRINT_FREQ == 0:
            print('[{0}] Epoch: [{1}][{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                                epoch+1, i, num_batches,
                                                                batch_time=batch_time,
                                                                data_time=data_time,
                                                                loss=losses))

        writer.add_scalar('Batch_Loss/train', losses.val, epoch*num_batches + i)

    writer.add_scalar('Epoch_Loss/train', losses.avg, epoch)

    return losses.avg


def validate(validation_dataloader, validation_data, model, criterion, epoch, writer):
    model.eval()  # training mode enables dropout

    # UNCOMMENT TO PERFORM VALIDATION
    val_batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    val_data_time = AverageMeter()  # data loading time per batch
    val_losses = AverageMeter()  # cross entropy loss

    val_start = time.time()

    num_batches = len(validation_dataloader)
    val_dots = np.ndarray((len(validation_data),), dtype=np.float32)

    for i, (imgs_paths, captions_docs, imgs, documents, sentences_per_document, words_per_sentence, labels) in enumerate(validation_dataloader):

        val_data_time.update(time.time() - val_start)

        imgs = imgs.to(device)

        documents = documents.squeeze(1).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        with torch.no_grad():
            imgs_embeddings, texts_embeddings, word_alphas, sentence_alphas = model(imgs, documents, sentences_per_document, words_per_sentence)

        # Loss
        loss = criterion(imgs_embeddings, texts_embeddings, labels)  # scalar

        imgs_embeddings = imgs_embeddings.detach().cpu()
        texts_embeddings = texts_embeddings.detach().cpu()

        val_dots[i*train_params['val_batch_size']:(i+1)*train_params['val_batch_size']] = np.dot(imgs_embeddings, texts_embeddings.T).diagonal()/(np.linalg.norm(imgs_embeddings, axis=1)*np.linalg.norm(texts_embeddings, axis=1))

        # Keep track of metrics
        val_losses.update(loss.item(), labels.size(0))
        val_batch_time.update(time.time() - val_start)

        val_start = time.time()

        # Print training status
        if i % PRINT_FREQ == 0:
            print('\tEpoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch+1, i, len(validation_dataloader),
                                                                batch_time=val_batch_time,
                                                                data_time=val_data_time,
                                                                loss=val_losses))
            write_batch_to_log(writer, validation_data.root, imgs_paths, captions_docs, i)

    writer.add_scalar('Epoch_Loss/val', val_losses.avg, epoch)
    writer.add_histogram('Val_Dots_Distribution', val_dots, epoch)

    return val_losses.avg


def main(model_params, train_params):

    if not os.path.isdir(train_params['log_folder']):
        print('Log folder "{}" does not exist. We are attempting creating it... '.format(train_params['log_folder']))
        os.mkdir(train_params['log_folder'])
        print('Folder created!')

    if train_params['finetune_semantic_model']:
        print('[{}] Loading saved model weights to finetune (or continue training): {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), train_params['model_checkpoint_filename']))
        _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(train_params['model_checkpoint_filename'])

        datetimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter(log_dir='{}/{}_{}_lr{}_{}eps_ft/'.format(train_params['log_folder'], datetimestamp, train_params['hostname'], train_params['learning_rate'], train_params['num_epochs']), filename_suffix='_{}'.format(datetimestamp))
    else:
        embeddings, word_map = load_embeddings_matrix(train_params['embeddings_filename'], model_params['word_embed_size'], train_params['use_fake_embeddings'])

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
                           use_word_level_attention=model_params['use_word_level_attention'],
                           pretrained_img_embedder=True)  # Pretrained on ImageNet

        # Init word embeddings layer with pretrained embeddings
        model.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(embeddings)
        model.text_embedder.doc_embedder.sent_embedder.allow_word_embeddings_finetunening(False)  # Make it available to finetune the word embeddings
        model.img_embedder.fine_tune(False)  # Freeze/Unfreeze ResNet-50 layers. We didn't use it in our paper. But, feel free to try ;)
        model.apply(init_weights)  # Apply function "init_weights" to all FC layers of our model.

        datetimestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        writer = SummaryWriter(log_dir='{}/{}_{}_lr{}_{}eps/'.format(train_params['log_folder'], datetimestamp, train_params['hostname'], train_params['learning_rate'], train_params['num_epochs']), filename_suffix='_{}'.format(datetimestamp))

    training_dataloader, validation_dataloader, training_data, validation_data = create_sets(word_map, train_params)

    if train_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])
    elif train_params['optimizer'] == 'SGD':
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=train_params['learning_rate'])

    # Loss functions
    criterion = train_params['criterion']

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    print(model)

    # Epochs
    curr_val_loss = float('inf')
    for epoch in range(0, train_params['num_epochs']):
        # One epoch's training
        train_loss = train(training_dataloader=training_dataloader,
                           training_data=training_data,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch,
                           writer=writer)

        val_loss = validate(validation_dataloader=validation_dataloader,
                            validation_data=validation_data,
                            model=model,
                            criterion=criterion,
                            epoch=epoch,
                            writer=writer)

        if train_params['learning_rate_decay'] is not None and epoch % train_params['decay_at_every'] == train_params['decay_at_every']-1:
            # Decay learning rate every epoch
            adjust_learning_rate(optimizer, train_params['learning_rate_decay'])

        if val_loss < curr_val_loss:
            # Save checkpoint
            # save_checkpoint(epoch+1, model, optimizer, word_map, datetimestamp, model_params, train_params)
            curr_val_loss = val_loss


if __name__ == '__main__':
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', '--model_checkpoint_filename', type=str, default=None, dest='model_checkpoint_filename', help="Name (complete path) of the trained model (or checkpoint) file you want to FINE TUNE.")

    args = parser.parse_args()

    model_params = {
        'word_embed_size': 300,
        'sent_embed_size': 1024,
        'doc_embed_size': 2048,
        'hidden_feat_size': 512,
        'feat_embed_size': 128,
        'sent_rnn_layers': 1,
        'word_rnn_layers': 1,
        'word_att_size': 1024,  # Same as sent_embed_size
        'sent_att_size': 2048,  # Same as doc_embed_size

        'use_sentence_level_attention': True,
        'use_word_level_attention': True,
        'use_visual_shortcut': True,  # Uses the ResNet-50 output as the first hidden state (h_0) of the document embedder Bi-GRU.
    }

    train_params = {

        ##### Train data files #####

        # COCO 2017 TODO: Download COCO 2017 and set the following folders according to your root for COCO 2017
        'captions_train_fname': 'resources/COCO_2017/annotations/captions_train2017.json',  # TODO: Download the annotation file available at: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        'captions_val_fname': 'resources/COCO_2017/annotations/captions_val2017.json',  # TODO: Download the nnotation file available at: http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        'train_data_path': 'resources/COCO_2017/train2017/',  # TODO: Download and unzip the folder available at http://images.cocodataset.org/zips/train2017.zip
        'val_data_path': 'resources/COCO_2017/val2017/',  # Download and unzip the folder available at http://images.cocodataset.org/zips/val2017.zip

        'embeddings_filename': 'resources/glove.6B.300d.txt',  # TODO: Download and unzip the file "glove.6B.300d.txt" from the folder "glove.6B" currently available at http://nlp.stanford.edu/data/glove.6B.zip
        'use_fake_embeddings': False,  # Choose if you want to use fake embeddings (Tip: Activate to speed-up debugging) -- It adds random word embeddings, removing the demand of loading the embeddings.

        # Choose how much data you want to use for training and validating (Tip: Use lower values to speed-up debugging)
        'train_data_proportion': 1.,
        'val_data_proportion': 1.,

        # Training parameters (Values for the pretrained model may be different from these values below)
        'max_sents': 10,  # maximum number of sentences per document
        'max_words': 20,  # maximum number of words per sentence

        'train_batch_size': 64,
        'val_batch_size': 64,
        'num_epochs': 30,
        'learning_rate': 1e-5,
        'learning_rate_decay': None,  # We didn't use it in our paper. But, feel free to try ;)
        'decay_at_every': None,  # We didn't use it in our paper. But, feel free to try ;)
        'grad_clip': None,  # clip gradients at this value. We didn't use it in our paper. But, feel free to try ;)
        'finetune_semantic_model': args.model_checkpoint_filename is not None,
        'model_checkpoint_filename': args.model_checkpoint_filename,

        # Image transformation parameters
        'resize_size': 256,
        'random_crop_size': 224,
        'do_random_horizontal_flip': True,

        # Machine and user data
        'username': getpass.getuser(),
        'hostname': socket.gethostname(),

        # Training process
        'optimizer': 'Adam',  # We also tested with SGD -- No improvement over Adam
        'criterion': nn.CosineEmbeddingLoss(0.),

        'checkpoint_folder': 'models',
        'log_folder': 'logs'
    }

    main(model_params, train_params)

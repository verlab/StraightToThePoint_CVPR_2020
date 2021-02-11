from utils import load_checkpoint, convert_sentences_to_word_idxs
from PIL import Image
from datetime import datetime
from _utils.experiment_to_video_mapping import Experiment2VideoMapping
from tqdm import tqdm

import os
import argparse
import cv2

import numpy as np
import torch
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

import matplotlib

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def colorize(words, color_array, sent_color_array, sent_idx):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    # normalized_color_array = (color_array - np.min(color_array))/float((np.max(color_array) - np.min(color_array)))
    # normalized_sent_array = (sent_color_array - np.min(sent_color_array))/float((np.max(sent_color_array) - np.min(sent_color_array)))

    # pdb.set_trace()
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    sent_color = matplotlib.colors.rgb2hex(cmap(sent_color_array[sent_idx])[:3])
    colored_string_prefix = '<span class="barcode"; style="color: black; background-color: {}">{}</span>&nbsp'
    sent_color_array = np.around(sent_color_array, decimals=3)  # Round numbers to show
    colored_string = colored_string_prefix.format(sent_color, '&nbsp' + str(sent_color_array[sent_idx]) + '&nbsp')
    for i, (word, color) in enumerate(zip(words, color_array)):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    colored_string += '<br/>&nbsp&nbsp&nbsp&nbsp<span>{}</span>'.format(np.around(color_array, decimals=3))
    return colored_string


def extract_feats(model, word_map, max_words, imgs=None, docs=None):
    model.eval()  # eval mode disables dropout

    if imgs is not None and docs is not None:
        # Image features extraction
        imgs = imgs.to(device)
        imgs_feats, resnet_output = model.get_img_embedding(imgs)

        # Text features extraction
        words_per_sentence = np.ndarray((len(docs),), dtype=object)
        converted_sentences = np.ndarray((len(docs),), dtype=object)
        sentences_per_document = np.ndarray((len(docs),))
        for i, document in enumerate(docs):
            converted_sentences[i], words_per_sentence[i] = convert_sentences_to_word_idxs(document, max_words, word_map)
            sentences_per_document[i] = converted_sentences[i].shape[0]

        documents = np.vstack(converted_sentences).reshape(-1, int(sentences_per_document[0]), max_words)
        documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device)  # (batch_size)
        words_per_sentence = np.vstack(words_per_sentence).reshape(-1, int(sentences_per_document[0]))
        words_per_sentence = torch.from_numpy(words_per_sentence).to(device)  # (batch_size, sentence_limit)

        docs_feats, word_alphas, sentence_alphas = model.get_text_embedding(documents, sentences_per_document, words_per_sentence, resnet_output)

        return imgs_feats, docs_feats, word_alphas, sentence_alphas

    elif imgs is not None:
        imgs = imgs.to(device)
        imgs_feats, _ = model.get_img_embedding(imgs)

        return imgs_feats
    elif docs is not None:
        words_per_sentence = np.ndarray((len(docs),), dtype=object)
        converted_sentences = np.ndarray((len(docs),), dtype=object)
        sentences_per_document = np.ndarray((len(docs),))
        for i, document in enumerate(docs):
            converted_sentences[i], words_per_sentence[i] = convert_sentences_to_word_idxs(document, max_words, word_map)
            sentences_per_document[i] = converted_sentences[i].shape[0]

        documents = np.vstack(converted_sentences).reshape(-1, int(sentences_per_document[0]), max_words)
        documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device)  # (batch_size)
        words_per_sentence = np.vstack(words_per_sentence).reshape(-1, int(sentences_per_document[0]))
        words_per_sentence = torch.from_numpy(words_per_sentence).to(device)  # (batch_size, sentence_limit)

        docs_feats, word_alphas, sentence_alphas = model.get_text_embedding(documents, sentences_per_document, words_per_sentence)

        return docs_feats, word_alphas, sentence_alphas


if __name__ == '__main__':
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', '--model_checkpoint_filename', type=str, required=True, dest='model_checkpoint_filename', help="Name (complete path) of the trained model (or checkpoint) file.")
    parser.add_argument('-v', '--video_filename', type=str, default=None, dest='video_filename', help="Filename of the video to generate the VDAN features to")
    parser.add_argument("-e", "--experiment", dest="experiment", default=None, type=str, help="Experiment Name. Eg.: k3nRPKCyyVg, D4AnZ0ymfzw, LA6DXaQ5vGQ")
    parser.add_argument('-i', '--image_filename', type=str, default=None, dest='image_filename', help="Filename of the image to generate the VDAN features to")
    parser.add_argument('-u', '--user_text_filename', type=str, default=None, dest='user_text_filename', help="Any text document to be used in the VDAN feature extraction")
    parser.add_argument('-o', '--output_folder_path', type=str, default=None, dest='output_folder_path', help="Path for the output file")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, dest='batch_size', help="Batch size for the extraction")

    args = parser.parse_args()

    if args.experiment is not None:
        exp_map = Experiment2VideoMapping(args.experiment)
        output_filename = f'resources/{exp_map.dataset}/VDAN/'
    else:
        output_filename = './'

    if args.output_folder_path is not None:
        output_filename = args.output_folder_path + '/'

    batch_size = args.batch_size

    print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.model_checkpoint_filename))
    _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(args.model_checkpoint_filename)
    model.to(device)

    img_transform = T.Compose([T.Resize((train_params['random_crop_size'], train_params['random_crop_size'])),
                               T.ToTensor(),
                               T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    if args.image_filename is not None and args.user_text_filename is not None:
        print('[{}] Extraction Mode: Image-Text'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        img_output_filename = output_filename + os.path.split(os.path.splitext(args.image_filename)[0])[1] + '_vdan_pytorch_img_feats.npz'
        text_output_filename = output_filename + os.path.split(os.path.splitext(args.user_text_filename)[0])[1] + '_vdan_pytorch_user_feats.npz'
        print('[{}] Image Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), img_output_filename))
        print('[{}] Text Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), text_output_filename))

        print('[{}] Extracting...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        img = Image.open(args.image_filename).convert('RGB')
        img = img_transform(img)
        document = np.loadtxt(args.user_text_filename, delimiter='\n', dtype=str, encoding='utf-8')

        with torch.no_grad():
            img_feats, document_feats, word_alphas, sentence_alphas = extract_feats(model, word_map, train_params['max_words'], imgs=img.unsqueeze(0), docs=[document])

        # pdb.set_trace()
        s = []
        for i, sentence in enumerate(document):
            words_color_array = word_alphas.cpu().detach().numpy()[0][i] if word_alphas is not None else np.array([1.]*len(sentence))
            sents_color_array = sentence_alphas.cpu().detach().numpy()[0]
            words = sentence.split()
            s.append(colorize(words, words_color_array, sents_color_array, i))

        img_feats = img_feats.detach().cpu().numpy()[0]
        document_feats = document_feats.detach().cpu().numpy()[0]

        euc_dist = np.linalg.norm(img_feats - document_feats)
        dot_product = np.dot(img_feats, document_feats.T)/(np.linalg.norm(img_feats)*np.linalg.norm(document_feats))

        colorized_filename = output_filename + os.path.split(os.path.splitext(args.user_text_filename)[0])[1] + '_img_' + os.path.split(os.path.splitext(args.image_filename)[0])[1] + '_colorized.html'
        with open(colorized_filename, 'w') as f:
            f.write('<b>Euclidean Distance:</b> {:.3f}\t|\t<b>Cosine Similarity:</b> {:.3f}<br/><br/>'.format(euc_dist, dot_product))
            for sentence in s:
                f.write(sentence + '<br/>')

            f.write('<br/><img src="{}">'.format(args.image_filename))
        # pdb.set_trace()

        print('[{}] Euclidean Distance for these embeddings: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), euc_dist))
        print('[{}] Dot Product for these embeddings: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), dot_product))

        np.savez_compressed(img_output_filename, features=img_feats)
        np.savez_compressed(text_output_filename, features=document_feats)
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    elif args.image_filename is not None:
        print('[{}] Extraction Mode: Image'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        output_filename += os.path.split(os.path.splitext(args.image_filename)[0])[1] + '_vdan_pytorch_img_feats.npz'
        print('[{}] Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_filename))

        print('[{}] Extracting...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        img = Image.open(args.image_filename).convert('RGB')
        img = img_transform(img)
        with torch.no_grad():
            img_feats = extract_feats(model, word_map, train_params['max_words'], imgs=img.unsqueeze(0))

        np.savez_compressed(output_filename, features=img_feats.detach().cpu().numpy()[0])
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    elif args.video_filename is not None and args.user_text_filename is not None:
        print('[{}] Extraction Mode: Video'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        frames_feats_filename = '{}/{}_img_feats.npz'.format(output_filename, os.path.basename(args.video_filename).split('.')[0])
        doc_feats_filename = '{}/{}_{}_doc_feats.npz'.format(output_filename, os.path.basename(args.video_filename).split('.')[0], os.path.basename(args.user_text_filename).split('.')[0])
        print('[{}] Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_filename))

        video = cv2.VideoCapture(args.video_filename)

        num_frames = int(video.get(7))
        num_batches = int(np.ceil(num_frames/float(batch_size)))

        if not os.path.isfile(args.user_text_filename):
            print('ERROR: Please provide a document via parameter "-u"!')
            exit(1)
        else:
            document = np.loadtxt(args.user_text_filename, delimiter='\n', dtype=str, encoding='utf-8')

        frames_feats = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
        doc_feats = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
        for idx in tqdm(range(num_batches)):
            current_batch_size = min(num_frames - idx*batch_size, batch_size)

            X = np.zeros((current_batch_size, 3, train_params['random_crop_size'], train_params['random_crop_size']))
            for idx_j in range(current_batch_size):
                ret, frame = video.read()

                if not ret:
                    print('Error reading frame: {}'.format(idx*batch_size+idx_j))
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                x = img_transform(frame)

                X[idx_j, :, :, :] = x

            with torch.no_grad():
                imgs_feats, docs_feats, word_alphas, sentence_alphas = extract_feats(model, word_map, train_params['max_words'], imgs=torch.from_numpy(X).float(), docs=np.array([document]*current_batch_size))

            doc_feats[idx*batch_size:idx*batch_size+current_batch_size, :] = docs_feats.detach().cpu().numpy()
            frames_feats[idx*batch_size:idx*batch_size+current_batch_size, :] = imgs_feats.detach().cpu().numpy()

        if not os.path.isdir(os.path.dirname(doc_feats_filename)):  # Creating unexistent directory if needed
            os.mkdir(os.path.dirname(doc_feats_filename))
        semantic_encoder_name = os.path.basename(args.model_checkpoint_filename).split('.')[0]
        np.savez_compressed(frames_feats_filename, features=frames_feats, semantic_encoder_name=semantic_encoder_name)
        np.savez_compressed(doc_feats_filename, features=doc_feats, document=document, semantic_encoder_name=semantic_encoder_name)

        print(f'Frame feats saved to: {frames_feats_filename}')
        print(f'Document feats saved to: {doc_feats_filename}')
    elif args.experiment is not None:
        # pdb.set_trace()
        print('[{}] Extraction Mode: YouCook2 Experiment'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        frames_feats_filename = '{}/{}_img_feats.npz'.format(output_filename, exp_map.video_name)
        doc_feats_filename = '{}/{}_{}_doc_feats.npz'.format(output_filename, exp_map.video_name, os.path.basename(exp_map.user_document_filename).split('.')[0])
        print('[{}] Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_filename))

        video = cv2.VideoCapture(exp_map.video_filename)

        num_frames = int(video.get(7))
        num_batches = int(np.ceil(num_frames/float(batch_size)))

        if not os.path.isfile(exp_map.user_document_filename):
            print('ERROR: Please run rl_fast_forward/resources/create_youcook2_recipe_documents.py first!')
            exit(1)
        else:
            document = np.loadtxt(exp_map.user_document_filename, delimiter='\n', dtype=str, encoding='utf-8')

        frames_feats = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
        doc_feats = np.zeros((num_frames, model_params['feat_embed_size']), dtype=np.float32)
        for idx in tqdm(range(num_batches)):
            current_batch_size = min(num_frames - idx*batch_size, batch_size)

            X = np.zeros((current_batch_size, 3, train_params['random_crop_size'], train_params['random_crop_size']))
            for idx_j in range(current_batch_size):
                ret, frame = video.read()

                if not ret:
                    print('Error reading frame: {}'.format(idx*batch_size+idx_j))
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                x = img_transform(frame)

                X[idx_j, :, :, :] = x

            with torch.no_grad():
                imgs_feats, docs_feats, word_alphas, sentence_alphas = extract_feats(model, word_map, train_params['max_words'], imgs=torch.from_numpy(X).float(), docs=np.array([document]*current_batch_size))

            doc_feats[idx*batch_size:idx*batch_size+current_batch_size, :] = docs_feats.detach().cpu().numpy()
            frames_feats[idx*batch_size:idx*batch_size+current_batch_size, :] = imgs_feats.detach().cpu().numpy()

        if not os.path.isdir(os.path.dirname(doc_feats_filename)):  # Creating unexistent directory if needed
            os.mkdir(os.path.dirname(doc_feats_filename))
        semantic_encoder_name = os.path.basename(args.model_checkpoint_filename).split('.')[0]
        np.savez_compressed(frames_feats_filename, features=frames_feats, semantic_encoder_name=semantic_encoder_name)
        np.savez_compressed(doc_feats_filename, features=doc_feats, document=document, semantic_encoder_name=semantic_encoder_name)

        print(f'Frame feats saved to: {frames_feats_filename}')
        print(f'Document feats saved to: {doc_feats_filename}')
    elif args.user_text_filename is not None:
        model.text_embedder.doc_embedder.use_visual_shortcut = False

        print('[{}] Extraction Mode: Text'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        colorized_filename = output_filename + os.path.split(os.path.splitext(args.user_text_filename)[0])[1] + '_colorized.html'
        output_filename += os.path.split(os.path.splitext(args.user_text_filename)[0])[1] + '_vdan_pytorch_user_feats.npz'
        print('[{}] Output file: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_filename))

        document = np.loadtxt(args.user_text_filename, delimiter='\n', dtype=str, encoding='utf-8')

        print('[{}] Extracting...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        with torch.no_grad():
            document_feats, word_alphas, sentence_alphas = extract_feats(model, word_map, train_params['max_words'], docs=[document])

        # pdb.set_trace()
        s = []
        for i, sentence in enumerate(document):
            words_color_array = word_alphas.cpu().detach().numpy()[0][i]
            sents_color_array = sentence_alphas.cpu().detach().numpy()[0]
            words = sentence.split()
            s.append(colorize(words, words_color_array, sents_color_array, i))

        with open(colorized_filename, 'w') as f:
            f.write('<br/>')
            for sentence in s:
                f.write(sentence + '<br/>')
        # pdb.set_trace()

        np.savez_compressed(output_filename, features=document_feats.detach().cpu().numpy()[0])
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # torch.cuda.empty_cache()

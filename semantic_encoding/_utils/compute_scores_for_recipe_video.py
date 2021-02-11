from matplotlib import pyplot as plt
import matplotlib
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
import torch
import numpy as np
import json
import cv2
import argparse
import os
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from utils import load_checkpoint, convert_sentences_to_word_idxs

plt.gcf().subplots_adjust(bottom=0.40)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = os.path.abspath(__file__)

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def colorize(words, color_array, sent_color_array, sent_idx):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    normalized_color_array = (color_array - np.min(color_array))/float((np.max(color_array) - np.min(color_array)))
    normalized_sent_array = (sent_color_array - np.min(sent_color_array))/float((np.max(sent_color_array) - np.min(sent_color_array)))

    # pdb.set_trace()
    cmap = matplotlib.cm.get_cmap('RdBu')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    sent_color = matplotlib.colors.rgb2hex(cmap(normalized_sent_array[sent_idx])[:3])
    colored_string_prefix = '<span class="barcode"; style="color: black; background-color: {}">{}</span>&nbsp'
    sent_color_array = np.around(sent_color_array, decimals=3)  # Round numbers to show
    colored_string = colored_string_prefix.format(sent_color, '&nbsp' + str(sent_color_array[sent_idx]) + '&nbsp')
    for i, (word, color) in enumerate(zip(words, normalized_color_array)):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, '&nbsp' + word + '&nbsp')
    colored_string += '<br/>&nbsp&nbsp&nbsp&nbsp<span>{}</span>'.format(np.around(color_array, decimals=3))
    return colored_string


def extract_feats(model, max_words, word_map, imgs=None, docs=None):
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

        documents = np.vstack(converted_sentences).reshape(-1, int(sentences_per_document[0]), max_words)  # Use the same MAX_WORDS as in semantic_encoding/main.py
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
            converted_sentences[i], words_per_sentence[i] = convert_sentences_to_word_idxs(document, word_map)
            sentences_per_document[i] = converted_sentences[i].shape[0]

        documents = np.vstack(converted_sentences).reshape(-1, int(sentences_per_document[0]), 60)
        documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device)  # (batch_size)
        words_per_sentence = np.vstack(words_per_sentence).reshape(-1, int(sentences_per_document[0]))
        words_per_sentence = torch.from_numpy(words_per_sentence).to(device)  # (batch_size, sentence_limit)

        # pdb.set_trace()
        docs_feats, word_alphas, sentence_alphas = model.get_text_embedding(documents, sentences_per_document, words_per_sentence)

        return docs_feats, word_alphas, sentence_alphas


if __name__ == '__main__':
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', '--model_checkpoint_filename', type=str, required=True, dest='model_checkpoint_filename', help="Name (complete path) of the trained model (or checkpoint) file.")
    parser.add_argument('-a', '--annotations_filename', type=str, required=True, dest='annotations_filename', help="Filename of the annotations with the recipe videos (currently working only with the YouCook2 dataset annotation file -- youcookii_annotations_trainval.json)")
    parser.add_argument('-v', '--video_id', type=str, required=True, dest='video_id', help="ID of the video you want to compute scores for (The ID must be from the YouCook2 dataset)")
    parser.add_argument("-r", "--recipe_number", dest="recipe_number", default=None, type=str, help="Number of the recipe you want to extract. Eg.: 113, 405, etc. If not informed, it will infer by the video_id")
    parser.add_argument('-o', '--output_folder_path', type=str, default=None, dest='output_folder_path', help="Path for the output file")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, dest='batch_size', help="Batch size for the extraction")

    args = parser.parse_args()

    batch_size = args.batch_size
    model_name = '_'.join(os.path.split(os.path.splitext(args.model_checkpoint_filename)[0])[1].split('_')[:2])
    video_id = args.video_id
    if video_id[0] == '\\':  # We must use \ (backslash) to escape videos starting with the '-' symbol
        video_id = video_id[1:]
    annotations = json.load(open(args.annotations_filename))
    recipe_number = annotations['database'][video_id]['recipe_type']

    if 'annotations' in annotations['database'][video_id].keys():
        document = np.array([annotations['database'][video_id]['annotations'][i]['sentence'] for i in range(len(annotations['database'][video_id]['annotations']))])
        subset = annotations['database'][video_id]['subset']
    else:
        document = np.array([annotations['database'][video_id]['segments']['{}'.format(i)]['sentence'] for i in range(len(annotations['database'][video_id]['segments']))])
        if 'training' in args.annotations_filename:
            subset = 'training'
        elif 'val' in args.annotations_filename:
            subset = 'validation'
        else:
            subset = 'testing'

    video_filename = f'{os.path.dirname(os.path.dirname(file_path))}/rl_fast_forward/resources/YouCook2/raw_videos/{subset}/{recipe_number}/{video_id}'

    # Check if need to reload correct recipe
    text_img_correspond = True
    if args.recipe_number is not None and args.recipe_number != recipe_number:
        vids = [*annotations['database'].keys()]
        idx = -1
        while recipe_number != args.recipe_number and idx < len(vids)-1:
            idx += 1
            recipe_number = annotations['database'][vids[idx]]['recipe_type']

        if recipe_number != args.recipe_number:
            print('Recipe number {} not found! Try another one'.format(args.recipe_number))
            exit(1)

        text_img_correspond = False
        recipe_number = args.recipe_number

        # Recipe is different from the expected. Reload document
        if 'annotations' in annotations['database'][vids[idx]].keys():
            document = np.array([annotations['database'][vids[idx]]['annotations'][i]['sentence'] for i in range(len(annotations['database'][vids[idx]]['annotations']))])
        else:
            document = np.array([annotations['database'][vids[idx]]['segments']['{}'.format(i)]['sentence'] for i in range(len(annotations['database'][vids[idx]]['segments']))])

    # Setting output path and filename
    if args.output_folder_path is not None:
        video_basename = os.path.split(os.path.splitext(video_filename)[0])[1]
        output_filename = args.output_folder_path + '/mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_dists.npy'
        output_filename_cos = args.output_folder_path + '/mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_cos.npy'
        output_filename_atts = args.output_folder_path + '/mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_atts.npz'
    else:
        video_basename = os.path.split(os.path.splitext(video_filename)[0])[1]
        output_filename = './mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_dists.npy'
        output_filename_cos = './mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_cos.npy'
        output_filename_atts = './mdl-' + model_name + '_rcp-' + recipe_number + '_vid-' + video_basename + '_atts.npz'

    img_transform = T.Compose([T.Resize((224, 224)),
                               T.ToTensor(),
                               T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.model_checkpoint_filename))
    epoch, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(args.model_checkpoint_filename)
    model.to(device)
    model.eval()

    print(model)
    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    video = cv2.VideoCapture(video_filename)

    num_frames = int(video.get(7))
    frame_rate = video.get(5)
    num_batches = int(np.ceil(num_frames/float(batch_size)))

    cos = np.zeros((num_frames,))
    words_atts = np.zeros((num_frames, train_params['max_sents'], train_params['max_words']), dtype=np.float32)
    sentences_atts = np.zeros((num_frames, train_params['max_sents']), dtype=np.float32)
    for idx in tqdm(range(num_batches)):
        current_batch_size = min(num_frames - idx*batch_size, batch_size)

        X = np.zeros((current_batch_size, 3, 224, 224))
        for idx_j in range(current_batch_size):
            ret, frame = video.read()

            if not ret:
                print('Error reading frame: {}'.format(idx*batch_size+idx_j))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            x = img_transform(frame)

            X[idx_j, :, :, :] = x

        imgs_feats, documents_feats, words_alphas, sentences_alphas = extract_feats(model, train_params['max_words'], word_map, imgs=torch.from_numpy(X).float(), docs=np.vstack([document]*current_batch_size))

        cos[idx*batch_size:idx*batch_size+current_batch_size] = torch.bmm(imgs_feats.view(current_batch_size, 1, -1), documents_feats.view(current_batch_size, -1, 1)).view(-1).detach().cpu().numpy()
        # pdb.set_trace()
        words_atts[idx*batch_size:idx*batch_size+current_batch_size, :words_alphas.shape[1], :words_alphas.shape[2]] = words_alphas.detach().cpu().numpy()
        sentences_atts[idx*batch_size:idx*batch_size+current_batch_size, :sentences_alphas.shape[1]] = sentences_alphas.detach().cpu().numpy()

    np.save(output_filename_cos, cos)
    np.savez_compressed(output_filename_atts, words_atts=words_atts, sentences_atts=sentences_atts, document=document)

    video_annotations = annotations['database'][video_id]['annotations']

    plot_title = 'Model: {} | Corr: {}\nVideo: {} | Recipe No: {}'.format(model_name, 'Yes' if text_img_correspond else 'No', video_basename, recipe_number)
    plt.plot(cos, linewidth=1)

    cmap = plt.cm.get_cmap('hsv', len(video_annotations) + 1)
    sentences_string = ' '

    for idx, region in enumerate(video_annotations):
        video_region = region['segment']
        plt.axvspan(round(video_region[0] * frame_rate), round(video_region[1] * frame_rate), alpha=0.5, color=cmap(idx))
        sentences_string = sentences_string + '\n' + region['sentence']
    plt.xlabel(sentences_string)

    plt.suptitle(plot_title)
    plt.savefig(os.path.splitext(output_filename_cos)[0] + '.png', dpi=300)

    print(plot_title)
    print('[{}] Done! Saved at: {}\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), output_filename_cos))

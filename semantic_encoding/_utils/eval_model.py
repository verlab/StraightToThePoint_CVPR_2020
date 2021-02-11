import sys
sys.path.append("..")
from coco_captions_dataset import CocoCaptionsDataset
from utils import load_checkpoint, convert_sentences_to_word_idxs
from datetime import datetime
from tqdm import tqdm

import argparse
import multiprocessing

import numpy as np
import torch
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]
FEAT_EMBED_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

def extract_feats(model, word_map, imgs=None, docs=None):
    model.eval()  # eval mode disables dropout

    if imgs is not None:
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
        
        documents = np.vstack(converted_sentences).reshape(-1,int(sentences_per_document[0]),60)
        documents = torch.from_numpy(documents).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device) # (batch_size)
        words_per_sentence = np.vstack(words_per_sentence).reshape(-1,5)
        words_per_sentence = torch.from_numpy(words_per_sentence).to(device)  # (batch_size, sentence_limit)

        # pdb.set_trace()
        docs_feats, _, _ = model.get_text_embedding(documents, sentences_per_document, words_per_sentence)
        #pdb.set_trace()
        return docs_feats
    
if __name__=='__main__':
    """
    Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', '--model_checkpoint_filename', type=str, required=True, dest='model_checkpoint_filename', help="Name (complete path) of the trained model (or checkpoint) file.")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, dest='batch_size', help="Batch size for the extraction")
    parser.add_argument('-cos', '--use_cosine', dest='use_cosine', action='store_true', help="Use cosine similarity for computing the MRR")

    args = parser.parse_args()

    captions_val_fname = 'resources/COCO_2017/annotations/captions_val2017.json'
    val_data_path = 'resources/COCO_2017/val2017/'
    num_workers = int(multiprocessing.cpu_count()*0.8) # Using 80% of CPU cores
    batch_size = args.batch_size
    print('Using %d CPU cores...' % num_workers)

    img_transform = T.Compose( [T.Resize((224,224)),
                            T.ToTensor(),
                            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.model_checkpoint_filename))
    _, model, optimizer_state_dict, word_map, model_params, train_params = load_checkpoint(args.model_checkpoint_filename)
    model.to(device)
    model.eval()
    print(model)
    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    test_data = CocoCaptionsDataset(root = val_data_path,
                                    annFile = captions_val_fname,
                                    word_map = word_map,
                                    img_transform=img_transform,
                                    annotations_transform=T.ToTensor(),
                                    dataset_proportion=1.,
                                    generate_negatives=False)
    
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                shuffle=False)

    # coco = COCO(captions_val_fname)
    # img_ids = list(coco.imgs.keys())
    num_imgs = len(test_data)

    print('[{}] Extracting COCO annotations features...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))    
    X = np.ndarray((num_imgs,FEAT_EMBED_SIZE), dtype=np.float32)
    Y = np.ndarray((num_imgs,FEAT_EMBED_SIZE), dtype=np.float32)

    for i, (imgs_paths, captions_docs, imgs, documents, sentences_per_document, words_per_sentence, labels) in enumerate(tqdm(test_dataloader)):
        
        imgs = imgs.to(device)

        documents = documents.squeeze(1).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.to(device) # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        #labels = labels.squeeze(1).to(device)  # (batch_size)

        #pdb.set_trace()

        # Extracting COCO image features
        with torch.no_grad():
            imgs_feats, resnet_output = model.get_img_embedding(imgs)
        X[i*batch_size:min((i+1)*batch_size,num_imgs),:] = imgs_feats.detach().cpu().numpy()

        # Extracting COCO annotations features

        with torch.no_grad():
            docs_feats, _, _ = model.get_text_embedding(documents, sentences_per_document, words_per_sentence, resnet_output)
        Y[i*batch_size:min((i+1)*batch_size,num_imgs),:] = docs_feats.detach().cpu().numpy()
        
    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    print('[{}] Computing distances to features...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    dists = np.ndarray((num_imgs,), dtype=np.float32)
    MRR = 0. # Mean Reciprocal Rank

    if args.use_cosine:
        norms_X = np.linalg.norm(X, axis=1)
        norms_Y = np.linalg.norm(Y, axis=1)
        X_norm = X/np.repeat(norms_X.T, X.shape[1]).reshape(X.shape[0],-1) # Normalize X
        Y_norm = Y/np.repeat(norms_Y.T, Y.shape[1]).reshape(Y.shape[0],-1) # Normalize Y
        X_dot_Y = np.matmul(X_norm, Y_norm.transpose())
        for i in tqdm(range(num_imgs)):
            max_cos_idxs = np.argsort(X_dot_Y[i,:])[::-1]
            MRR += 1./(max_cos_idxs.tolist().index(i)+1)
            #pdb.set_trace()
    else:
        for i in tqdm(range(num_imgs)):
            dists = np.linalg.norm(np.tile(X[i,:],num_imgs).reshape(X.shape[0], -1) - Y, axis=1)
            
            # for j in range(num_imgs):
            #     dists[j] = np.linalg.norm(X[i,:] - Y[j,:])

            min_dists_idxs = np.argsort(dists)#[::-1]

            # Calculate and sum for the Mean Reciprocal Rank scoring (MRR)
            MRR += 1./(min_dists_idxs.tolist().index(i)+1) # Finds the first time the index occur
            #pdb.set_trace()

    MRR /= num_imgs
    print("We evaluate the model by the position of the correct answer for each query image.")
    print("The Mean Reciprocal Rank (MRR) score is: {}".format(MRR))
    print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    #[ann['caption'] for ann in coco.loadAnns(coco.getAnnIds(img_ids[4307]))]

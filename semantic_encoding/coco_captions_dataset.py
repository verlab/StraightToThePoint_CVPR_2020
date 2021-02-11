import torch.utils.data as data
from PIL import Image
import random
import os

import numpy as np
import torch
from utils import convert_sentences_to_word_idxs

class CocoCaptionsDataset(data.Dataset):
    def __init__(self, root, annFile, word_map, img_transform=None, annotations_transform=None, num_sentences=5, max_words=45, neg_sentences_proportion=0.5, dataset_proportion=1., generate_negatives=True):
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.ids = self.ids[:int(dataset_proportion*len(self.ids))]

        idx = 0
        while idx < len(self.ids):
            ann_ids = self.coco.getAnnIds(imgIds=[self.ids[idx]])
            if len(self.coco.loadAnns(ann_ids)) == 0:
                del self.ids[idx]
            else: idx += 1

        self.ids = np.array(self.ids)
        self.num_imgs = len(self.ids)

        self.img_transform = img_transform
        self.annotations_transform = annotations_transform
        self.generate_negatives = generate_negatives
        self.neg_sentences_proportion = neg_sentences_proportion
        self.word_map = word_map
        self.num_sentences = num_sentences
        self.max_words = max_words
        self.num_negative_sentences = int(self.neg_sentences_proportion*self.num_sentences)
        self.num_positive_sentences = self.num_sentences - self.num_negative_sentences

    def get_captions(self, imgs_ids):
        ann_ids = self.coco.getAnnIds(imgIds=imgs_ids)
        anns = self.coco.loadAnns(ann_ids)
        captions = list(set([ann['caption'] for ann in anns])) # Removing duplicates

        return captions

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple ((image, annotations), target). target is a binary list telling if images and captions match.
        """
        is_negative_sample, index = divmod(index, self.num_imgs) # If index greater than max, generate a negative sample instead

        img_id = self.ids[index]
        
        if is_negative_sample: #If it is negative, we should get a random annotation to truly make it negative
            captions = self.get_captions(self.ids[random.sample(range(self.num_imgs-1), 2)])
        else:
            captions = self.get_captions([img_id])
            negative_captions = self.get_captions(self.ids[random.sample(range(self.num_imgs-1), 1)])
            captions.extend(negative_captions)

        random.shuffle(captions)
        
        while len(captions) < self.num_sentences:
            captions.append('UNK')
        captions = captions[:self.num_sentences]

        # Converting annotations to array of indexes
        converted_annotations, words_per_sentence = convert_sentences_to_word_idxs(captions, self.max_words, self.word_map)

        img_path = self.coco.loadImgs([img_id])[0]['file_name']
        
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.annotations_transform is not None:
            converted_annotations = self.annotations_transform(converted_annotations)

        return img_path, captions, img, converted_annotations, converted_annotations.size()[1], words_per_sentence, torch.tensor([1. if not is_negative_sample else -1.], dtype=torch.float32)

    def __len__(self):
        if self.generate_negatives:
            return len(self.ids)*2 # Returns twice the length because for every positive document match, we create its negative counterpart
        else:
            return len(self.ids)

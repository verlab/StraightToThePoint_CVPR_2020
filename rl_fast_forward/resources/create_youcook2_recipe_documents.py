#!/bin/python3
import os
import json
import sys
import numpy as np
from tqdm import tqdm

file_path = os.path.abspath(__file__)
sys.path.append(f'{os.path.dirname(os.path.dirname(os.path.dirname(file_path)))}/semantic_encoding/_utils/')
from experiment_to_video_mapping import Experiment2VideoMapping

if __name__=='__main__':
    if not os.path.isdir(f'{os.path.dirname(file_path)}/YouCook2/recipes'):
        os.mkdir(f'{os.path.dirname(file_path)}/YouCook2/recipes')

    annotations = json.load(open(f'{os.path.dirname(file_path)}/YouCook2/youcookii_annotations_trainval.json'))
    experiments = Experiment2VideoMapping.get_dataset_experiments('YouCook2')
    for experiment in tqdm(experiments, desc='Creating recipe files'):
        recipe = [ann['sentence'] for ann in annotations['database'][experiment]['annotations']]
        np.savetxt(f'{os.path.dirname(file_path)}/YouCook2/recipes/recipe_{experiment}.txt', recipe, delimiter='\n', fmt='%s', encoding='utf-8')

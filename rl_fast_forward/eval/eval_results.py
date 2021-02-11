from experiment_to_video_mapping import Experiment2VideoMapping
import argparse
import json
import sys
import os
import cv2
import numpy as np

from tqdm import tqdm

file_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(file_path))) + '/semantic_encoding/_utils/')


def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Evaluation parameters')
    #
    parser.add_argument('-gt', '--ground_truth_filename', type=str, required=True, help="JSON File containing the ground truth frames.")
    parser.add_argument('-sf', '--selected_frames_filename', type=str, required=True, help='JSON with the selected frames.')

    return parser.parse_args(args)


def compute_f1_score(selected_frames, gt_frames, num_frames):

    # Prepare ground truth vec
    ground_truth = np.array([False]*num_frames, dtype=bool)
    ground_truth[gt_frames-1] = True

    # Prepare selected frames vec
    sf_binary = np.array([False]*num_frames, dtype=bool)
    sf_binary[selected_frames-1] = True

    # Compute Precision and Recall
    true_positives = sf_binary * ground_truth
    precision = np.sum(true_positives)/len(selected_frames)
    recall = np.sum(true_positives)/np.sum(ground_truth)

    f1_score = 2*(precision*recall)/(precision+recall)
    return precision, recall, f1_score


def compute_IoU(selected_frames, gt_segments, num_frames, fps):
    return 0


def compute_jaccard_similarity(selected_frames, gt_segments, num_frames, fps):
    return 0


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    gt_json = json.load(open(args.ground_truth_filename))
    sf_json = json.load(open(args.selected_frames_filename))

    output_save_basename = os.path.basename(args.selected_frames_filename).split('.')[0]
    json_save_filename = '{}_results.json'.format(output_save_basename)
    csv_save_filename = '{}_results.csv'.format(output_save_basename)

    scores_dict = {}
    recipe_id_tuple = []
    for video_id in tqdm(sf_json['data'].keys()):

        video_filename = f"{os.path.dirname(file_path)}/resources/YouCook2/raw_videos/{gt_json['data'][video_id]['split']}/{gt_json['data'][video_id]['recipe_id']}/{video_id}"

        experiment = Experiment2VideoMapping(video_id)

        gt_frames = np.array(gt_json['data'][video_id]['frames'], dtype=int)
        sf = np.array(sf_json['data'][video_id]['frames'], dtype=int)

        # COMPUTING F1 SCORE
        precision, recall, f1_score = compute_f1_score(sf, gt_frames, experiment.num_frames)

        # Putting everything in one piece
        scores_dict[video_id] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}
        recipe_id_tuple.append((video_id, gt_json['data'][video_id]['recipe_id']))

    f = open(csv_save_filename, 'w')
    print('RECIPE_ID, VIDEO_ID, PRECISION, RECALL, F1 SCORE')
    f.write('RECIPE_ID, VIDEO_ID, PRECISION, RECALL, F1 SCORE\n')
    for tup in sorted(recipe_id_tuple, key=lambda x: (x[1], x[0])):
        video_id = tup[0]
        print_str = '{}, {}, {:.3f}, {:.3f}, {:.3f}'.format(tup[1], video_id, scores_dict[video_id]['precision'], scores_dict[video_id]['recall'], scores_dict[video_id]['f1_score'])
        print(print_str)
        f.write('{}\n'.format(print_str))

    f.close()

    with open(json_save_filename, 'w') as f:
        json.dump(scores_dict, f, sort_keys=True)

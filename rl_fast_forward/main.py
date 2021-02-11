""" Deep RL Algorithms for OpenAI Gym environments
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/semantic_encoding/_utils/')

import argparse
import numpy as np
import cv2
import socket
import json
from tqdm import tqdm
from datetime import datetime as dt

from envs import VideoEnvironment
from REINFORCE.agent import Agent

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from experiment_to_video_mapping import Experiment2VideoMapping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args(args):
    """ Parse arguments from command line input
    """
    parser = argparse.ArgumentParser(description='Training parameters')
    #
    parser.add_argument('-n', '--nb_epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Batch size for feature extraction (in ResNet-50)")
    #
    parser.add_argument('--create_video', dest='create_video', action='store_true', help="Create the output video based on the test environment")
    parser.add_argument('--print_speedup', dest='print_speedup', action='store_true', help="Print the speed-up in the created video")
    #
    parser.add_argument('-s', '--semantic_encoder_model_filename', type=str, help='Semantic encoder model path filename')
    parser.add_argument('-mp', '--reinforce_model_filename', type=str, help='Policy Model path filename to perform fine tuning or testing only')
    parser.add_argument('-u', '--user_document_filename', type=str, default=None, help="Name (complete path) of the sentences document file if not using parameters -d and -e (if using a dataset such as YouCook2, the recipes are automatically loaded)")
    parser.add_argument('-i', '--input_video_filename', type=str, help="Name (complete path) of the input video file if not using parameters -d or -e (if using a dataset such as YouCook2, videos are automatically loaded)")
    parser.add_argument('-e', '--experiment', type=str, help="Name of the experiment video file [e.g. CWxjNRIKjA0, SkawoKeyNoQ]")
    parser.add_argument('-d', '--dataset', type=str, help="Name of the dataset that will be used for training [e.g., YouCook2]")
    #
    parser.add_argument('--is_test', dest='is_test', action='store_true', help="If set, performs only testing")
    parser.add_argument('--include_test', dest='include_test', action='store_true', help="If set, test after finishing the agent training")
    #
    parser.add_argument('-eb', '--entropy_beta', type=float, default=1e-2, help='Beta value for the entropy H')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5, help='Learning rate for any algorithm')
    parser.add_argument('-cr', '--critic_lr', type=float, default=1e-3, help='Learning rate for the Critic branch')
    parser.add_argument('-g', '--gamma', type=float, default=1.0, help='Discount factor for the Discounted Reward in REINFORCE')

    parser.set_defaults(render=False, include_test=False, is_test=False, create_video=False, print_speedup=False)
    return parser.parse_args(args)

def train(args, training_envs, test_envs):
    hostname = socket.gethostname()
    aux_env = list(training_envs.values())[0]

    agent = Agent(state_size=aux_env.state_size, action_size=aux_env.action_size, lr=args.learning_rate, critic_lr=args.critic_lr, gamma=args.gamma, entropy_beta=args.entropy_beta)
    model_path = 'models/{}_{}_reinforce.pth'.format(agent.creation_timestamp, args.dataset.lower())
    agent.train(envs=training_envs, dataset_name=args.dataset, n_epochs=args.nb_epochs, model_path=model_path)
    
    if args.include_test:
        if not os.path.isdir('results/{}'.format(args.dataset)):
            print('Folder "{}" does not exist. We are attempting creating it... '.format('results/{}'.format(args.dataset)))
            os.mkdir('results/{}'.format(args.dataset))
            print('Folder created!')

        agent.load_model(model_path.split('.')[0] + '_epoch{}.pth'.format(args.nb_epochs))
        json_sf = {'info': {'version': 'v1.1_{}'.format(dt.now().strftime('%Y%m%d_%H%M%S')), 'dataset': args.dataset}, 'data': {}}
        for exp_key, env in test_envs.items():
            print('\nTesting: {}'.format(exp_key))
            agent.test(env, args.dataset, exp_key)
            sf = env.selected_frames
            # print(sf)
            agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(args.dataset), suffix=exp_key)
            #pdb.set_trace()
            agent.writer.close()
            if args.experiment:
                output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, args.experiment)
            elif args.input_video_filename:
                video_basename = os.path.basename(args.input_video_filename).split('.')[0]
                output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, video_basename)
            else:
                output_filename = 'results/{}/{}_sf_rl_fast_forward_{}.npy'.format(args.dataset, agent.creation_timestamp, exp_key)
                
                json_sf['data'][exp_key] = {'file_name': os.path.abspath(output_filename), 'recipe_id': env.experiment.recipe_id, 'split': 'test', 'frames': list(np.array(sf, dtype=str))}

            np.save(output_filename, sf)
            print('Saved to: {}'.format(output_filename))

            if args.create_video:
                output_video_filename = os.path.splitext(output_filename)[0] + '.avi'
                
                fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
                input_video = cv2.VideoCapture(env.experiment.video_filename)
                output_video = cv2.VideoWriter(output_video_filename, fourcc, env.experiment.fps, (env.experiment.frame_width, env.experiment.frame_height))

                print('\nCreating output video at {}'.format(output_video_filename))
                skips = list(np.hstack([[0],np.array(env.selected_frames)[1:] - np.array(env.selected_frames)[:-1]]))
                for idx, selected_frame in enumerate(tqdm(env.selected_frames)):
                    input_video.set(1, selected_frame)
                    check, frame = input_video.read()
                    if check:
                        if args.print_speedup:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, '{}x'.format(skips[idx]),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
                        output_video.write(frame)
                    else:
                        print("Frame read error")

                print('\nDone!')

                input_video.release()
                output_video.release()

        if args.dataset in ['YouCook2']:
            with open('results/{}_{}_{}_ours_selected_frames.json'.format(agent.creation_timestamp, hostname, args.dataset.lower()), 'w') as f:
                json.dump(json_sf, f, sort_keys=True)
                print('\nJSON results file saved at: results/{}_{}_{}_ours_selected_frames.json'.format(agent.creation_timestamp, hostname, args.dataset.lower()))


def test(args):
    hostname = socket.gethostname()
    if args.experiment:
        env = VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, experiment_name=args.experiment, batch_size=args.batch_size)
    elif args.input_video_filename:        
        env = VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, args.input_video_filename, batch_size=args.batch_size)
    elif args.dataset:
        experiments = np.array(Experiment2VideoMapping.get_dataset_experiments(args.dataset))

        np.random.shuffle(experiments)
        
        test_set = [exp for exp in experiments if Experiment2VideoMapping(exp).split_type in ['validation', 'test']]
        
        test_envs = {exp_key: VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, experiment_name=exp_key, batch_size=args.batch_size) for exp_key in test_set}

        json_sf = {'info': {'version': 'v1.1_{}'.format(dt.now().strftime('%Y%m%d_%H%M%S')), 'dataset': args.dataset}, 'data': {}}

        print('Test set:\n{}'.format(' '.join(test_envs.keys())))

    
    agent = Agent(state_size=env.state_size, action_size=env.action_size, lr=args.learning_rate, critic_lr=args.critic_lr, gamma=args.gamma, entropy_beta=args.entropy_beta)
    agent.load_model(args.reinforce_model_filename)

    if args.experiment:
        print('\nTesting: {}'.format(args.experiment))
        agent.test(env, args.experiment, args.experiment)
        sf = env.selected_frames
        # print(sf)
        agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(args.experiment))
        agent.writer.close()
        output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, args.experiment)
        np.save(output_filename, sf)
        print('Saved to: {}'.format(output_filename))
    elif args.input_video_filename:
        video_basename = os.path.basename(args.input_video_filename).split('.')[0]
        print('\nTesting: {}'.format(video_basename))
        agent.test(env, video_basename, video_basename)
        sf = env.selected_frames
        # print(sf)
        agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(video_basename))
        agent.writer.close()
        output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, video_basename)
        np.save(output_filename, sf)
        print('Saved to: {}'.format(output_filename))
    else:
        if not os.path.isdir('results/{}'.format(args.dataset)):
            print('Folder "{}" does not exist. We are attempting creating it... '.format('results/{}'.format(args.dataset)))
            os.mkdir('results/{}'.format(args.dataset))
            print('Folder created!')

        for exp_key, env in test_envs.items():
            print('\nTesting: {}'.format(exp_key))
            agent.test(env, args.dataset, exp_key)
            sf = env.selected_frames
            # print(sf)
            agent.write_selected_frames_image_to_log(env, 0, 'Test_{}'.format(args.dataset), suffix=exp_key)
            #pdb.set_trace()
            agent.writer.close()
            if args.experiment:
                output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, args.experiment)
            elif args.input_video_filename:
                video_basename = os.path.basename(args.input_video_filename).split('.')[0]
                output_filename = 'results/{}_sf_rl_fast_forward_{}.npy'.format(agent.creation_timestamp, video_basename)
            else:
                output_filename = 'results/{}/{}_sf_rl_fast_forward_{}.npy'.format(args.dataset, agent.creation_timestamp, exp_key)

                json_sf['data'][exp_key] = {'file_name': os.path.abspath(output_filename), 'recipe_id': env.experiment.recipe_id, 'split': 'test', 'frames': list(np.array(sf, dtype=str))}

            np.save(output_filename, sf)
            print('Saved to: {}'.format(output_filename))

            if args.create_video:
                output_video_filename = os.path.splitext(output_filename)[0] + '.avi'
                
                fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
                input_video = cv2.VideoCapture(env.experiment.video_filename)
                output_video = cv2.VideoWriter(output_video_filename, fourcc, env.experiment.fps, (env.experiment.frame_width, env.experiment.frame_height))

                print('\nCreating output video at {}'.format(output_video_filename))
                skips = list(np.hstack([[0],np.array(env.selected_frames)[1:] - np.array(env.selected_frames)[:-1]]))
                for idx, selected_frame in enumerate(tqdm(env.selected_frames)):
                    input_video.set(1, selected_frame)
                    check, frame = input_video.read()
                    if check:
                        if args.print_speedup:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(frame, '{}x'.format(skips[idx]),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
                        output_video.write(frame)
                    else:
                        print("Frame read error")

                print('\nDone!')

                input_video.release()
                output_video.release()

        if args.dataset in ['YouCook2']:
            with open('results/{}_{}_{}_ours_selected_frames.json'.format(agent.creation_timestamp, hostname, args.dataset.lower()), 'w') as f:
                json.dump(json_sf, f, sort_keys=True)
                print('\nJSON results file saved at: results/{}_{}_{}_ours_selected_frames.json'.format(agent.creation_timestamp, hostname, args.dataset.lower()))
def main(args=None):
    np.random.seed(123)

    # Parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    if args.is_test:
        test(args)
        return

    if args.dataset: ## If a dataset is provided, run on it...
        experiments = np.array(Experiment2VideoMapping.get_dataset_experiments(args.dataset))

        np.random.shuffle(experiments)
        print(experiments)
        
        all_envs = {exp_key: VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, experiment_name=exp_key, batch_size=args.batch_size) for exp_key in experiments}
        training_envs = {exp_key: env for exp_key, env in all_envs.items() if env.experiment.split_type=='training'}
        test_envs = {exp_key: env for exp_key, env in all_envs.items() if env.experiment.split_type=='validation'}

        print('Training set:\n{}'.format(' '.join(training_envs.keys())))
        print('Test set:\n{}'.format(' '.join(test_envs.keys())))

        train(args, training_envs, test_envs)
        
    elif args.experiment:
        env = VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, experiment_name=args.experiment, batch_size=args.batch_size)

        training_envs = {args.experiment: env}
        test_envs = {args.experiment: env}

        args.dataset = args.experiment
        train(args, training_envs, test_envs)

    elif args.input_video_filename:
        video_basename = os.path.basename(args.input_video_filename).split('.')[0]
        
        env = VideoEnvironment(args.semantic_encoder_model_filename, args.user_document_filename, input_video_filename=args.input_video_filename, batch_size=args.batch_size)

        training_envs = {video_basename: env}
        test_envs = {video_basename: env}

        args.dataset = video_basename
        train(args, training_envs, test_envs)

if __name__ == "__main__":
    main()

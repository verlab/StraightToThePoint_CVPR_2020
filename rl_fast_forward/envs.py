import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torchvision.transforms as T
import torch.backends.cudnn as cudnn

import cv2
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/semantic_encoding/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/semantic_encoding/_utils/')
from utils import load_checkpoint
from extract_vdan_feats import extract_feats
from experiment_to_video_mapping import Experiment2VideoMapping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN   = [0.485, 0.456, 0.406]
IMAGENET_STD    = [0.229, 0.224, 0.225]

DOC_FEATS_SIZE = 128
IMG_FEATS_SIZE = 128

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

class VideoEnvironment():
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, semantic_encoder_model_filename, user_document_filename=None, experiment_name=None, input_video_filename=None, batch_size=32):

        self.batch_size = batch_size

        self.visual_feats = None
        self.user_feats = None

        self.acceleration = 1
        self.curr_frame_id = 0
        self.MAX_SKIP = 20
        self.MIN_SKIP = 1
        self.MAX_ACC = 5
        self.MIN_ACC = 1
        
        self.skip = np.random.randint(self.MIN_SKIP, self.MAX_SKIP)

        self.img_feats_size = IMG_FEATS_SIZE
        self.doc_feats_size = DOC_FEATS_SIZE
        self.action_size = 3
        self.state_size = self.doc_feats_size + self.img_feats_size 

        self.experiment_name = experiment_name
        self.semantic_encoder_model_name = os.path.basename(semantic_encoder_model_filename).split('.')[0]
        
        if self.experiment_name is not None:
            self.experiment = Experiment2VideoMapping(self.experiment_name)
            self.input_video_filename = self.experiment.video_filename

            if user_document_filename is None and self.experiment.dataset == 'YouCook2':
                user_document_filename = self.experiment.user_document_filename

            frame_feats_base_dir = 'resources/{}/VDAN/img_feats/{}'.format(self.experiment.dataset, self.semantic_encoder_model_name)
            doc_feats_base_dir = 'resources/{}/VDAN/doc_feats/{}'.format(self.experiment.dataset, self.semantic_encoder_model_name)
            user_doc_basename = os.path.basename(user_document_filename).split('.')[0]

            if not os.path.isdir(frame_feats_base_dir):
                os.mkdir(frame_feats_base_dir)
            if not os.path.isdir(doc_feats_base_dir):
                os.mkdir(doc_feats_base_dir)

            self.frames_feats_filename = '{}/{}_img_feats.npz'.format(frame_feats_base_dir, self.experiment.video_name)
            self.doc_feats_filename = '{}/{}/{}_doc_feats.npz'.format(doc_feats_base_dir, user_doc_basename, self.experiment.video_name)

        elif input_video_filename:
            self.input_video_filename = input_video_filename
            self.frames_feats_filename = '{}/{}_img_feats.npz'.format(os.path.dirname(input_video_filename), os.path.basename(input_video_filename).split('.')[0])
            self.doc_feats_filename = '{}/{}_{}_doc_feats.npz'.format(os.path.dirname(input_video_filename), os.path.basename(input_video_filename).split('.')[0], os.path.basename(user_document_filename).split('.')[0])
        else:
            print('ERROR: experiment name or input video filename needed!')
            exit(1)

        self.semantic_encoder_model_filename = semantic_encoder_model_filename
        self.user_document_filename = user_document_filename

        self.states = self.generate_environment_states()

        # Computing dot product
        print('[{}] Computing semantic features (Dot products)...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        self.dots_vec = np.zeros((self.num_frames,), dtype=np.float32)
        for idx in tqdm(range(self.num_frames)):
            self.dots_vec[idx] = np.dot(self.states[idx, :self.img_feats_size], self.states[idx, self.img_feats_size:].T)
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        self.total_reward = 0
        self.selected_frames = []
                    
    def load_semantic_encoder(self, model_filename):
        print('[{}] Loading saved model weights: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), model_filename))
        _, model, _, word_map, _, train_params = load_checkpoint(model_filename)
        model.to(device)
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        return model, word_map, train_params['max_words']
    
    def generate_environment_states(self):
        document = np.loadtxt(self.user_document_filename, delimiter='\n', dtype=str, encoding='utf-8')

        # Load features if they already exist
        if os.path.exists(self.frames_feats_filename) and os.path.exists(self.doc_feats_filename):
            doc_feats_dic = np.load(self.doc_feats_filename, allow_pickle=True)
            
            doc_feats = doc_feats_dic['features']
            saved_document = doc_feats_dic['document']
            semantic_encoder_name = str(doc_feats_dic['semantic_encoder_name'])
            
            ##### Checking if the features to be loaded are valid! #####
            # Check if the saved document is the same as the loaded one. If not, somebody changed it and it is not valid since features are different.
            # Also checking if the semantic model is the same as the one used to extract these features 
            if (len(document) == len(saved_document) and (document == saved_document).all() and semantic_encoder_name == os.path.basename(self.semantic_encoder_model_filename).split('.')[0]):
                print('\n\n[{}] Loading feats from {} and {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), os.path.basename(self.frames_feats_filename).split('.')[0], os.path.basename(self.doc_feats_filename).split('.')[0]))
                frames_feats = np.load(self.frames_feats_filename, allow_pickle=True)['features']
                self.num_frames = frames_feats.shape[0]

                states = np.concatenate([doc_feats, frames_feats], axis=1)
                print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
                return states
            else:
                print('\n[{}] Saved features cannot be used: the provided document or model has been modified...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        else:
            print('\n[{}] Frames and/or document feats not found...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


        ##### If we cannot load or use the features for any reason, keep on going and extract it again

        print('\n[{}] Extracting frames and document feats for video: {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), os.path.basename(self.input_video_filename)))
        
        semantic_encoder, word_map, max_words = self.load_semantic_encoder(self.semantic_encoder_model_filename)

        img_transform = T.Compose( [T.Resize((224,224)),
                                    T.ToTensor(),
                                    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        if self.input_video_filename is None:
            print('ERROR: Could not load the video for experiment {}'.format(self.experiment_name))
            exit(1)

        video = cv2.VideoCapture(self.input_video_filename)        

        self.num_frames = int(video.get(7))

        if not self.num_frames:
            print('ERROR: Could not load the video')
            exit(1)

        num_batches = int(np.ceil(self.num_frames/float(self.batch_size)))

        states = np.zeros((self.num_frames, (self.doc_feats_size + self.img_feats_size)), dtype=np.float32) #Removing location and budget vecs
        frames_feats = np.zeros((self.num_frames,self.img_feats_size), dtype=np.float32)
        doc_feats = np.zeros((self.num_frames,self.doc_feats_size), dtype=np.float32)
        for idx in tqdm(range(num_batches)):
            current_batch_size = min(self.num_frames - idx*self.batch_size, self.batch_size)

            X = np.zeros((current_batch_size, 3, 224, 224))
            for idx_j in range(current_batch_size):
                ret, frame = video.read()

                if not ret:
                    print('Error reading frame: {}'.format(idx*self.batch_size+idx_j))
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                x = img_transform(frame)

                X[idx_j, :, :, :] = x

            #pdb.set_trace()
            with torch.no_grad():
                imgs_feats, docs_feats, word_alphas, sentence_alphas = extract_feats(semantic_encoder, word_map, max_words, imgs=torch.from_numpy(X).float(), docs=np.array([document]*current_batch_size))

            doc_feats[idx*self.batch_size:idx*self.batch_size+current_batch_size,:] = docs_feats.detach().cpu().numpy()
            frames_feats[idx*self.batch_size:idx*self.batch_size+current_batch_size,:] = imgs_feats.detach().cpu().numpy()

            states[idx*self.batch_size:idx*self.batch_size+current_batch_size, :] = np.concatenate([docs_feats.detach().cpu().numpy(), imgs_feats.detach().cpu().numpy()], axis=1)
        
        #img_feats, document_feats, word_alphas, sentence_alphas = extract_feats(model, word_map, imgs=img.unsqueeze(0), docs=[document])
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        print('[{}] Saving feats...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        # Saving stuff
        if not os.path.isdir(os.path.dirname(self.doc_feats_filename)): # Creating unexistent directory if needed
            os.mkdir(os.path.dirname(self.doc_feats_filename))

        semantic_encoder_name = os.path.basename(self.semantic_encoder_model_filename).split('.')[0]
        np.savez_compressed(self.frames_feats_filename, features=frames_feats, semantic_encoder_name=semantic_encoder_name)
        np.savez_compressed(self.doc_feats_filename, features=doc_feats, document=document, semantic_encoder_name=semantic_encoder_name)
        
        print('[{}] Frames feats saved to {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.frames_feats_filename))
        print('[{}] Document feats saved to {}...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.doc_feats_filename))
        print('[{}] Done!\n'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

        return states
        
    def step(self, action):
        #pdb.set_trace()
        info = {'items': []}

        self.prev_frame_id = self.curr_frame_id

        #Check action
        #action = np.argmax(action)
        if action == 0: # Accelerate
            if self.skip + self.acceleration <= self.MAX_SKIP:
                self.skip += self.acceleration
            else:
                self.skip = self.MAX_SKIP
            if self.acceleration < self.MAX_ACC:
                self.acceleration += 1
        elif action == 2: # Decelerate
            if self.skip - self.acceleration >= self.MIN_SKIP:
                self.skip -= self.acceleration
            else:
                self.skip = self.MIN_SKIP
            if self.acceleration > self.MIN_ACC:
                self.acceleration -= 1
        
        self.curr_frame_id += int(self.skip)
        
        done = self.curr_frame_id >= self.states.shape[0] #or len(self.selected_frames) > self.desired_num_frames
        
        if done:
            self.curr_frame_id = self.states.shape[0]-1
            observation = None
            reward = self.get_semantic_reward()

            self.total_reward += reward            
            return observation, reward, done, info

        self.selected_frames.append(self.curr_frame_id)

        observation = self.states[self.curr_frame_id]

        reward = self.get_semantic_reward()

        return observation, reward, done, info
        
    def reset(self):
        self.skip = np.random.randint(self.MIN_SKIP, self.MAX_SKIP)
        self.acceleration = 1
        self.prev_frame_id = 0
        self.curr_frame_id = 0
        self.total_reward = 0
        self.selected_frames = []

        observation = self.states[self.curr_frame_id]

        return observation

    def get_semantic_reward(self):        
        return self.dots_vec[self.curr_frame_id]

    def get_num_selected_frames(self):
        return len(self.selected_frames)

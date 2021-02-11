import torch
import torch.nn as nn
import json
import cv2
import os
import numpy as np
import torchvision.transforms as T
from tqdm import tqdm
from semantic_encoding.models import VDAN
from PIL import Image
from semantic_encoding.utils import convert_sentences_to_word_idxs
from rl_fast_forward.REINFORCE.policy import Policy
from rl_fast_forward.REINFORCE.critic import Critic

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class JointModel(nn.Module):
    def __init__(self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers, hidden_feat_emb_size, final_feat_emb_size, sent_att_size, word_att_size, use_visual_shortcut=False, use_sentence_level_attention=False, use_word_level_attention=False, sent_rnn_dropout=0.25, word_rnn_dropout=0.25, dropout=0.5, pretrained_img_embedder=True, action_size=3):
        super(JointModel, self).__init__()

        self.vdan = VDAN(vocab_size=vocab_size,
                         doc_emb_size=doc_emb_size,  # ResNet-50 embedding size
                         sent_emb_size=sent_emb_size,
                         word_emb_size=word_emb_size,  # GloVe embeddings size
                         sent_rnn_layers=sent_rnn_layers,
                         word_rnn_layers=word_rnn_layers,
                         hidden_feat_emb_size=hidden_feat_emb_size,
                         final_feat_emb_size=final_feat_emb_size,
                         sent_att_size=sent_att_size,
                         word_att_size=word_att_size,
                         use_visual_shortcut=use_visual_shortcut,
                         use_sentence_level_attention=use_sentence_level_attention,
                         use_word_level_attention=use_word_level_attention,
                         sent_rnn_dropout=sent_rnn_dropout,
                         word_rnn_dropout=word_rnn_dropout,
                         dropout=dropout,
                         pretrained_img_embedder=pretrained_img_embedder)

        self.agent = Policy(state_size=final_feat_emb_size*2, action_size=action_size)
        self.critic = Critic(state_size=final_feat_emb_size*2)

    def foward(self, imgs, documents, sentences_per_document, words_per_sentence):
        """
        params:

        """
        imgs_embeddings, texts_embeddings, word_alphas, sentence_alphas = self.vdan(imgs, documents, sentences_per_document, words_per_sentence)

        action_probs = self.policy(torch.cat(imgs_embeddings, texts_embeddings))

        return action_probs, imgs_embeddings, texts_embeddings, word_alphas, sentence_alphas

    def fast_forward_video(self, video_filename, document, output_video_filename=None, max_words=10):
        word_map = json.load(open(f'{os.path.dirname(os.path.abspath(__file__))}/semantic_encoding/resources/glove6B_word_map.json'))
        img_transform = T.Compose([T.Resize((224,224)),
                                   T.ToTensor(),
                                   T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

        converted_sentences, words_per_sentence = convert_sentences_to_word_idxs(document, max_words, word_map)
        sentences_per_document = np.array([converted_sentences.shape[0]])

        transformed_document = torch.from_numpy(converted_sentences).unsqueeze(0).to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = torch.from_numpy(sentences_per_document).to(device)  # (batch_size)
        words_per_sentence = torch.from_numpy(words_per_sentence).unsqueeze(0).to(device)  # (batch_size, sentence_limit)

        video = cv2.VideoCapture(video_filename)
        if output_video_filename:
            fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
            fps = video.get(5)
            frame_width = int(video.get(3))
            frame_height = int(video.get(4))
            output_video = cv2.VideoWriter(output_video_filename, fourcc, fps, (frame_width, frame_height))

        num_frames = int(video.get(7))
        MIN_SKIP = 1
        MAX_SKIP = 20
        MAX_ACC = 5
        MIN_ACC = 1
        acceleration = 1
        skip = 1
        frame_idx = 0
        selected_frames = []

        pbar = tqdm(total=num_frames)
        while frame_idx < num_frames:
            i = 0
            while i < skip and frame_idx < num_frames:
                ret, frame = video.read()
                i += 1

            if not ret:
                print('Error reading frame: {}'.format(frame_idx))
                break
            
            transformed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed_frame = Image.fromarray(transformed_frame)
            transformed_frame = img_transform(transformed_frame).unsqueeze(0).to(device)

            if output_video_filename:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, '{}x'.format(skip), (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                output_video.write(frame)
                
            with torch.no_grad():
                img_embedding, text_embedding, word_alphas, sentence_alphas = self.vdan(transformed_frame, transformed_document, sentences_per_document, words_per_sentence)

                action_probs = self.agent(torch.cat([img_embedding, text_embedding],axis=1))
            
            action = torch.argmax(action_probs).item()

            if action == 0:  # Accelerate
                if skip + acceleration <= MAX_SKIP:
                    skip += acceleration
                else:
                    skip = MAX_SKIP
                if acceleration < MAX_ACC:
                    acceleration += 1
            elif action == 2:  # Decelerate
                if skip - acceleration >= MIN_SKIP:
                    skip -= acceleration
                else:
                    skip = MIN_SKIP
                if acceleration > MIN_ACC:
                    acceleration -= 1

            frame_idx += skip
            selected_frames.append(frame_idx+1)
            pbar.update(skip)

        pbar.close()

        return selected_frames

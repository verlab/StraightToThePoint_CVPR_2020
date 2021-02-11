dependencies = ['torch', 'torchvision', 'numpy']

from semantic_encoding.utils import init_weights
from semantic_fast_forward import JointModel
import torch
import numpy as np

# from tqdm import tqdm
# from pycocotools.coco import COCO
# from datetime import datetime
# import torch.nn as nn
# import torch.hub
# import nltk


def SemanticFastForward_RL(pretrained=False, progress=False, sent_emb_size=1024, hidden_feat_emb_size=512, final_feat_emb_size=128, sent_att_size=2048, word_att_size=1024, use_visual_shortcut=True, use_sentence_level_attention=True, use_word_level_attention=True, word_embeddings=None, fine_tune_word_embeddings=False, fine_tune_resnet=False, action_size=3):
    if not word_embeddings:
        word_embeddings = np.random.random((400002, 300)).astype(np.float32)

    model = JointModel(vocab_size=400002,
                       doc_emb_size=2048,  # ResNet-50 embedding size
                       sent_emb_size=sent_emb_size,
                       word_emb_size=300,  # GloVe embeddings size
                       sent_rnn_layers=1,
                       word_rnn_layers=1,
                       hidden_feat_emb_size=hidden_feat_emb_size,
                       final_feat_emb_size=final_feat_emb_size,
                       sent_att_size=sent_att_size,
                       word_att_size=word_att_size,
                       use_visual_shortcut=use_visual_shortcut,
                       use_sentence_level_attention=use_sentence_level_attention,
                       use_word_level_attention=use_word_level_attention,
                       sent_rnn_dropout=0.25,
                       word_rnn_dropout=0.25,
                       dropout=0.5,
                       pretrained_img_embedder=True,
                       action_size=action_size)

    # Init word embeddings layer with pretrained embeddings
    model.vdan.text_embedder.doc_embedder.sent_embedder.init_pretrained_embeddings(torch.from_numpy(word_embeddings))
    model.vdan.text_embedder.doc_embedder.sent_embedder.allow_word_embeddings_finetunening(fine_tune_word_embeddings)  # Make it available to finetune the word embeddings
    model.vdan.img_embedder.fine_tune(fine_tune_resnet)  # Freeze/Unfreeze ResNet-50 layers. We didn't use it in our paper. But, feel free to try ;)

    model.vdan.apply(init_weights)  # Apply function "init_weights" to all FC layers of our model.

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url('https://github.com/verlab/StraightToThePoint_CVPR_2020/releases/download/v1.0.0/sff_rl_vdan_model-5b50b542.pth',
                                                        progress=progress)

        model.vdan.load_state_dict(state_dict['vdan_state_dict'])
        model.agent.load_state_dict(state_dict['agent_state_dict'])

    return model

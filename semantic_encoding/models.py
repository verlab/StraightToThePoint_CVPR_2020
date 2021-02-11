import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from torchvision.models import resnet50

RESNET50_FEATS_SIZE = 2048


class VDAN(nn.Module):
    def __init__(self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers, hidden_feat_emb_size, final_feat_emb_size, sent_att_size, word_att_size, use_visual_shortcut=False, use_sentence_level_attention=False, use_word_level_attention=False, sent_rnn_dropout=0.25, word_rnn_dropout=0.25, dropout=0.5, pretrained_img_embedder=True):
        super(VDAN, self).__init__()

        self.text_embedder = TextSubnetwork(vocab_size,
                                            doc_emb_size,
                                            sent_emb_size,
                                            word_emb_size,
                                            sent_rnn_layers,
                                            word_rnn_layers,
                                            sent_att_size,
                                            word_att_size,
                                            use_visual_shortcut=use_visual_shortcut,
                                            use_sentence_level_attention=use_sentence_level_attention,
                                            use_word_level_attention=use_word_level_attention,
                                            sent_rnn_dropout=sent_rnn_dropout,
                                            word_rnn_dropout=word_rnn_dropout)

        self.img_embedder = ImageSubnetwork(final_feat_emb_size, pretrained_img_embedder)

        self.dropout = nn.Dropout(dropout)  # A dropout layer
        self.relu = F.relu  # Applying ReLU
        self.norm = F.normalize  # Applying normalization

        # Image Subnetwork
        self.img_fc1 = nn.Linear(RESNET50_FEATS_SIZE, hidden_feat_emb_size)  # 2048 img_feats size (output size of Resnet50)
        self.img_fc2 = nn.Linear(hidden_feat_emb_size, final_feat_emb_size)  # Linear(256,128)
        self.img_bnorm = nn.BatchNorm1d(final_feat_emb_size)  # Use Batch Norm this trick makes the training faster

        # Text Subnetwork
        self.text_fc1 = nn.Linear(2*doc_emb_size, hidden_feat_emb_size)
        self.text_fc2 = nn.Linear(hidden_feat_emb_size, final_feat_emb_size)  # Linear(256,128)
        self.text_bnorm = nn.BatchNorm1d(final_feat_emb_size)  # Use Batch Norm this trick makes the training faster

    def l2norm(self, X):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
        X = torch.div(X, norm)
        return X

    def forward(self, img, documents, sentences_per_document, words_per_sentence):

        # Image
        img_embedding, resnet_output = self.get_img_embedding(img)

        # Text
        text_embedding, word_alphas, sentence_alphas = self.get_text_embedding(documents, sentences_per_document, words_per_sentence, resnet_output=resnet_output)

        return img_embedding, text_embedding, word_alphas, sentence_alphas

    def get_img_embedding(self, img):
        resnet_output = self.img_embedder(img).squeeze(3).squeeze(2)
        # resnet_output = self.dropout(resnet_output) # UNCOMMENT FOR DROPOUT
        img_hidden_feat = F.relu(self.img_fc1(resnet_output))
        # img_hidden_feat = self.dropout(img_hidden_feat) # UNCOMMNET FOR DROPOUT
        img_hidden_feat = self.img_fc2(img_hidden_feat)
        # img_hidden_feat = F.relu(img_hidden_feat) # UNCOMMENT FOR RELU (ONLY POSITIVE DOT PRODUCTS)

        img_hidden_feat = self.img_bnorm(img_hidden_feat)
        #img_hidden_feat = F.relu(img_hidden_feat)
        img_embedding = self.l2norm(img_hidden_feat)

        return img_embedding, resnet_output

    def get_text_embedding(self, documents, sentences_per_document, words_per_sentence, resnet_output=None):
        doc_embeddings, word_alphas, sentence_alphas = self.text_embedder(documents, sentences_per_document, words_per_sentence, resnet_output)
        # doc_embeddings = self.dropout(doc_embeddings) # UNCOMMENT FOR DROPOUT
        text_hidden_feat = F.relu(self.text_fc1(doc_embeddings))
        # text_hidden_feat = self.dropout(text_hidden_feat) # UNCOMMNET FOR DROPOUT
        text_hidden_feat = self.text_fc2(text_hidden_feat)
        # text_hidden_feat = F.relu(text_hidden_feat) # UNCOMMENT FOR RELU (ONLY POSITIVE DOT PRODUCTS)

        text_hidden_feat = self.text_bnorm(text_hidden_feat)
        #text_hidden_feat = F.relu(text_hidden_feat)
        text_embedding = self.l2norm(text_hidden_feat)

        return text_embedding, word_alphas, sentence_alphas


class ImageSubnetwork(nn.Module):
    def __init__(self, final_feat_emb_size, pretrained_img_embedder):
        super(ImageSubnetwork, self).__init__()
        resnet50_model = resnet50(pretrained=pretrained_img_embedder)  # pretrained ImageNet ResNet-50

        # Removing the FC layer to add ours!
        modules = list(resnet50_model.children())[:-1]
        self.resnet = nn.Sequential(*modules)

    def forward(self, img):
        img_feats = self.resnet(img)

        return img_feats

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for (all layers.) #convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = fine_tune


class TextSubnetwork(nn.Module):
    def __init__(self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers,
                 sent_att_size, word_att_size, use_visual_shortcut, use_sentence_level_attention, use_word_level_attention, sent_rnn_dropout=0.25, word_rnn_dropout=0.25):
        super(TextSubnetwork, self).__init__()

        self.doc_embedder = DocumentEmbedder(vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers, sent_att_size, word_att_size, use_visual_shortcut, use_sentence_level_attention, use_word_level_attention, sent_rnn_dropout, word_rnn_dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence, resnet_output):
        doc_embeddings, word_alphas, sentence_alphas = self.doc_embedder(documents, sentences_per_document, words_per_sentence, resnet_output)

        return doc_embeddings, word_alphas, sentence_alphas


class DocumentEmbedder(nn.Module):
    def __init__(self, vocab_size, doc_emb_size, sent_emb_size, word_emb_size, sent_rnn_layers, word_rnn_layers,
                 sent_att_size, word_att_size, use_visual_shortcut, use_sentence_level_attention, use_word_level_attention, sent_rnn_dropout, word_rnn_dropout):
        super(DocumentEmbedder, self).__init__()
        self.use_sentence_level_attention = use_sentence_level_attention
        self.use_word_level_attention = use_word_level_attention
        self.use_visual_shortcut = use_visual_shortcut

        self.sent_embedder = SentenceEmbedder(vocab_size, sent_emb_size, word_emb_size, word_rnn_layers, word_att_size, use_word_level_attention, word_rnn_dropout)

        self.embedder = nn.GRU(input_size=2*sent_emb_size,
                               hidden_size=doc_emb_size,
                               num_layers=sent_rnn_layers,
                               # dropout=sent_rnn_dropout,
                               bidirectional=True,
                               batch_first=True)

        self.sentence_attention = nn.Linear(2 * doc_emb_size, sent_att_size)
        self.sentence_context_vector = nn.Linear(sent_att_size, 1,
                                                 bias=False)  # this performs a dot product with the linear layer's 1D parameter vector, which is the sentence context vector
        #self.dropout = nn.Dropout(dropout)

    def forward(self, documents, sentences_per_document, words_per_sentence, resnet_output):

        # Re-arrange as sentences by removing sentence-pads (DOCUMENTS -> SENTENCES)
        packed_sentences = pack_padded_sequence(documents,
                                                lengths=sentences_per_document.tolist(),
                                                batch_first=True,
                                                enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentences (n_sentences, word_pad_len)

        # Re-arrange sentence lengths in the same way (DOCUMENTS -> SENTENCES)
        packed_words_per_sentence = pack_padded_sequence(words_per_sentence,
                                                         lengths=sentences_per_document.tolist(),
                                                         batch_first=True,
                                                         enforce_sorted=False)  # a PackedSequence object, where 'data' is the flattened sentence lengths (n_sentences)

        # Find sentence embeddings by applying the word-level attention module
        sent_embeddings, word_alphas = self.sent_embedder(packed_sentences.data, packed_words_per_sentence.data)

        # Apply the sentence-level RNN over the sentence embeddings (PyTorch automatically applies it on the PackedSequence)
        if self.use_visual_shortcut:
            outputs, ht = self.embedder(PackedSequence(data=sent_embeddings,
                                                       batch_sizes=packed_sentences.batch_sizes,
                                                       sorted_indices=packed_sentences.sorted_indices,
                                                       unsorted_indices=packed_sentences.unsorted_indices),
                                        resnet_output.repeat(2, 1).view(-1, resnet_output.size()[0], resnet_output.size()[1]))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size) === ResNet50 output is used as a h_0 vector to provide visual information
        else:
            outputs, ht = self.embedder(PackedSequence(data=sent_embeddings,
                                                       batch_sizes=packed_sentences.batch_sizes,
                                                       sorted_indices=packed_sentences.sorted_indices,
                                                       unsorted_indices=packed_sentences.unsorted_indices))  # a PackedSequence object, where 'data' is the output of the RNN (n_sentences, 2 * sentence_rnn_size)

        # Use code bellow to apply attention mechanism
        # Similarly re-arrange sentence-level RNN outputs as documents by re-padding with 0s (SENTENCES -> DOCUMENTS)
        if self.use_sentence_level_attention:

            # Find attention vectors by applying the attention linear layer on the output of the RNN
            att_s = self.sentence_attention(outputs.data)
            att_s = torch.tanh(att_s)
            att_s = self.sentence_context_vector(att_s).squeeze(1)
            max_value = att_s.max()
            att_s = torch.exp(att_s - max_value)

            att_s, _ = pad_packed_sequence(PackedSequence(data=att_s,
                                                          batch_sizes=packed_sentences.batch_sizes,
                                                          sorted_indices=packed_sentences.sorted_indices,
                                                          unsorted_indices=packed_sentences.unsorted_indices),
                                           batch_first=True)

            sentence_alphas = att_s / torch.sum(att_s, dim=1, keepdim=True)

            documents, _ = pad_packed_sequence(outputs, batch_first=True)

            documents = documents * sentence_alphas.unsqueeze(2)
            documents = documents.sum(dim=1)

            if self.use_word_level_attention:
                word_alphas, _ = pad_packed_sequence(PackedSequence(data=word_alphas,
                                                                    batch_sizes=packed_sentences.batch_sizes,
                                                                    sorted_indices=packed_sentences.sorted_indices,
                                                                    unsorted_indices=packed_sentences.unsorted_indices),
                                                     batch_first=True)

            return documents, word_alphas, sentence_alphas
        else:
            return torch.cat([ht[0], ht[1]], dim=1), word_alphas, None


class SentenceEmbedder(nn.Module):
    def __init__(self, vocab_size, sent_emb_size, word_emb_size, word_rnn_layers, word_att_size, use_word_level_attention, word_rnn_dropout):
        super(SentenceEmbedder, self).__init__()
        self.use_word_level_attention = use_word_level_attention

        self.word_embedding_layer = nn.Embedding(vocab_size, word_emb_size)

        self.embedder = nn.GRU(input_size=word_emb_size,
                               hidden_size=sent_emb_size,
                               num_layers=word_rnn_layers,
                               # dropout=word_rnn_dropout,
                               bidirectional=True,
                               batch_first=True)

        self.word_att = nn.Linear(2 * sent_emb_size, word_att_size)
        self.word_context_vector = nn.Linear(word_att_size, 1, bias=False)

    def init_pretrained_embeddings(self, embeddings):
        """
        This function fills the Embedding layer with pretrained embeddings.
        """
        self.word_embedding_layer.weight = nn.Parameter(embeddings)

        for p in self.word_embedding_layer.parameters():
            p.requires_grad = False

    def allow_word_embeddings_finetunening(self, allow=False):
        """
        This function permits to freeze or not the word_embeddings training (Default: frozen).
        """
        for p in self.word_embedding_layer.parameters():
            p.requires_grad = allow

    def forward(self, sentences, words_per_sentence):

        embedded_sentences_words = self.word_embedding_layer(sentences)

        packed_words = pack_padded_sequence(embedded_sentences_words,
                                            lengths=words_per_sentence.tolist(),
                                            batch_first=True,
                                            enforce_sorted=False)

        outputs, ht = self.embedder(packed_words)

        # Use code bellow to apply attention mechanism
        if self.use_word_level_attention:
            att_w = self.word_att(outputs.data)
            att_w = torch.tanh(att_w)
            att_w = self.word_context_vector(att_w).squeeze(1)

            max_value = att_w.max()
            att_w = torch.exp(att_w - max_value)

            att_w, _ = pad_packed_sequence(PackedSequence(data=att_w,
                                                          batch_sizes=outputs.batch_sizes,
                                                          sorted_indices=outputs.sorted_indices,
                                                          unsorted_indices=outputs.unsorted_indices),
                                           batch_first=True)

            word_alphas = att_w / torch.sum(att_w, dim=1, keepdim=True)

            sentences, _ = pad_packed_sequence(outputs, batch_first=True)

            sentences = sentences * word_alphas.unsqueeze(2)
            sentences = sentences.sum(dim=1)

            return sentences, word_alphas
        else:
            return torch.cat([ht[0], ht[1]], dim=1), None

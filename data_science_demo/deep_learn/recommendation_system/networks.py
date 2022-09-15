import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, BertModel, BertConfig

"""
############################################################
# Networds
############################################################
TODO:
    4. How to embedding impress video

############################################################
"""


class YTRSEmbedding(nn.Module):
    def __init__(self, video_feature_dim, embed_size, corpus_size,
                 gru=True, ranking_model=False):
        super().__init__()

        ##################################################
        # Candidate generate model
        ##################################################
        # This nn.LeakyReLU() is import (by exp)
        self.embed_video = nn.Sequential(
            nn.Linear(video_feature_dim, embed_size), nn.ReLU())

        # Fusion features
        self.fusion_mode = 'average'
        self.gru_watch, self.gru_collect = None, None
        if gru:
            self.fusion_mode = 'gru'
            self.gru_watch = nn.GRU(embed_size, embed_size, batch_first=True)
            self.gru_collect = nn.GRU(embed_size, embed_size, batch_first=True)

        self.embed_language = nn.Embedding(corpus_size['language'], 64)

        ##################################################
        # Ranking model
        ##################################################
        if ranking_model:
            self.embed_cefr = nn.Embedding(corpus_size['c'], 64)
            self.embed_category = nn.Embedding(corpus_size['ca'], 64)
            self.embed_channel = nn.Embedding(corpus_size['ch'], 64)

        self.pad = {'video': torch.zeros([video_feature_dim])}

    def embed_samples(self, samples):
        embed_samples = self.embed_video(samples)
        return embed_samples

    def candidate(self, _videos, t_videos, languages):
        avg_embed_video = self._embedding_videos(watched_videos,
                                                 self.gru_watch)
        avg_embed_c_videos = self._embedding_videos(t_videos,
                                                          self.gru_collect)
        embed_language = self._embedding_language(languages)
        return (avg_embed_video, avg_embed_c_videos, embed_language)

    def ranking(self, watched_videos, language, category):
        avg_embed_video = self._embed_watched_videos(watched_videos)
        embed_language = self._embed_language(language)
        embed_category = self._embed_category(category)
        return (avg_embed_video, embed_language, embed_category)

    # TODO 4: How to embedding impress video
    def _embedding_videos(self, videos, fusion_model=None):
        embed_videos = self.embed_video(videos)
        # Fusion
        if self.fusion_mode == 'gru':
            _, avg_embed_video = fusion_model(embed_videos)
        else:
            avg_embed_video = torch.mean(embed_videos, 1)
        return avg_embed_video.squeeze(dim=0)

    def _embedding_language(self, language):
        embed_language = self.embed_language(language).squeeze(dim=1)
        return embed_language

    def _embed_watched_videos(self, watched_videos):
        batch_size, num_videos = watched_videos.shape[:2]
        # Watched videos (SBERT features)
        watched_videos = watched_videos.reshape(batch_size*num_videos, -1)
        embed_watched_videos = self.embed_video(watched_videos)
        embed_watched_videos = \
            embed_watched_videos.reshape(batch_size, num_videos, -1)
        avg_embed_video = torch.mean(embed_watched_videos, 1)
        return avg_embed_video

    def _embed_cefr(self, cefr):
        embed_cefr = self.embed_cefr(cefr).squeeze(dim=1)
        return embed_cefr

    def _embed_category(self, category):
        embed_category = self.embed_category(category).squeeze(dim=1)
        return embed_category

    def _embed_channel(self, channel):
        embed_channel = self.embed_channel(channel).squeeze(dim=1)
        return embed_channel


class YTRSCandidateGenerate(nn.Module):
    def __init__(self, input_size, in_channel, output_size, n_layer):
        super().__init__()
        self.candidate_generate_model = self._build_model(
            input_size, in_channel, output_size, n_layer)

    def forward(self, avg_embed_video, avg_embed_collect_video,
                example_age, gender, embed_language, embed_pos_samples,
                embed_neg_samples):
        # Concatenate all features
        user_feature = torch.cat([
            avg_embed_video, avg_embed_collect_video, example_age,
            example_age**2, gender, embed_language], dim=1)

        # Candidate generate
        user_features = \
            self.candidate_generate_model(user_feature).unsqueeze(dim=2)

        # Logistic regression (dimension is equal to number of samples)
        predict_pos = torch.matmul(embed_pos_samples,
                                   user_features).squeeze(dim=2)
        predict_neg = torch.stack(
            [torch.matmul(embed_neg_samples, user_feature)
             for user_feature in user_features.squeeze(dim=2)], dim=0)
        return predict_pos, predict_neg

    def loss(self, predict_pos, predict_neg, num_positive, w_pos=10, w_neg=0.1):
        batch_size = predict_pos.shape[0]
        loss = 0
        for pred_p, pred_n, num in zip(predict_pos, predict_neg, num_positive):
            predict = torch.cat([pred_p[:num], pred_n])
            predict = F.softmax(predict, dim=0)
            target = torch.zeros_like(predict)
            target[:num] = 1
            # Weighted loss
            weight = torch.zeros_like(predict)
            weight[:num], weight[num:] = w_pos, w_neg
            loss += F.binary_cross_entropy(predict, target, weight=weight)

        loss /= batch_size
        return loss

    def predict(self, avg_embed_video, avg_embed_collect_video,
                example_age, gender, embed_language, embed_samples):
        # Concatenate all features
        user_feature = torch.cat([
            avg_embed_video, avg_embed_collect_video, example_age,
            example_age**2, gender, embed_language], dim=1)

        # Candidate generate
        user_feature = self.candidate_generate_model(user_feature).unsqueeze(dim=2)
        # Logistic regression (dimension is equal to number of samples)
        predict = torch.matmul(embed_samples, user_feature).squeeze(dim=2)
        return predict

    def _build_model(self, input_size, in_channel, output_size, n_layer):
        dim = in_channel
        # First Layer
        layers = [nn.Linear(input_size, dim), nn.BatchNorm1d(dim),
                  nn.LeakyReLU()]
        # Hidden layers
        for _ in range(n_layer-2):
            layers += [nn.Linear(dim, dim//2), nn.BatchNorm1d(dim//2),
                       nn.LeakyReLU()]
            dim = dim // 2
        # Last layer
        layers += [nn.Linear(dim, output_size), nn.ReLU()]
        return nn.Sequential(*layers)


class YTRSRanking(nn.Module):
    def __init__(self, input_size, in_channel, output_size, n_layer):
        super().__init__()
        self.ranking_model = self._build_model(
            input_size, in_channel, output_size, n_layer)

        self.logistic_regression = nn.Sequential(nn.Linear(output_size, 1),
                                                 nn.Sigmoid())

    def forward(self, embed_impress_video, avg_embed_video, embed_language,
                last_watch_time, embed_cefr, embed_category, video_duration,
                video_created_time):
        # Time since last watch
        time = last_watch_time
        time_since_last_watch = torch.cat([
            torch.sqrt(time), time, torch.square(time)])

        # Impression video duration
        length = video_duration
        video_duration = torch.cat([
            torch.sqrt(length), length, torch.square(length)])

        # Impression video created time
        time = video_created_time
        video_created_time = torch.cat([
            torch.sqrt(time), time, torch.square(time)])

        # Number ob previous impressions (maybe we don't record this)

        # Concatenate all features
        feature = torch.cat([
            embed_impress_video, avg_embed_video,
            embed_language, time_since_last_watch,
            embed_cefr, embed_category, video_duration, video_created_time
            ], dim=1)

        # Ranking
        feature = self.ranking_model(feature)
        probability = self.logistic_regression(feature)  # one impression video
        return probability

    def _build_model(self, input_size, in_channel, output_size, n_layer):
        dim = in_channel
        # First Layer
        layers = [nn.BatchNorm1d(input_size),
                  nn.Linear(input_size, dim), nn.LeakyReLU()]
        # Hidden layers
        for _ in range(n_layer-2):
            layers += [nn.Linear(dim, dim//2), nn.LeakyReLU()]
            dim = dim // 2
        # Last layer
        layers += [nn.Linear(dim, output_size), nn.LeakyReLU()]
        return nn.Sequential(*layers)


class YoutubeDeepRecSys(nn.Module):
    def __init__(self, corpus_size, embed_size, input_size, in_channel,
                 output_size, n_layer, criterion=None):
        super().__init__()
        self.criterion = criterion
        self.embed_video = nn.Embedding(corpus_size['v'], embed_size)
        self.embed_sample = nn.Embedding(corpus_size['v'], output_size)

        self.embed_level = nn.Embedding(corpus_size['l'], 64)
        self.embed_location = nn.Embedding(corpus_size['lo'], 64)
        self.fusion_layer = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, in_channel), nn.ReLU())
        self.candidate_generate = self._build_model(in_channel, in_channel,
                                                    output_size, n_layer)

    def forward(self, watched_videos, embed_searched_keywords,
                embed_collect_words, level, location, gender, age, platform,
                example_age, samples, prob=False):

        # Watched videos
        embed_watched_videos = self.embed_video(watched_videos)
        avg_embed_video = torch.mean(embed_watched_videos, 1)
        # Search keywords
        avg_embed_keyword = torch.mean(embed_searched_keywords, 1)
        # Collect words
        avg_embed_collect = torch.mean(embed_collect_words, 1)
        # Level
        embed_level = self.embed_location(level)
        avg_embed_level = torch.mean(embed_level, 1)
        # Location
        embed_location = self.embed_location(location).squeeze(dim=1)

        # Concatenate all features
        user_feature = torch.cat([
            avg_embed_video, avg_embed_keyword, avg_embed_collect,
            avg_embed_level, embed_location, gender, age, platform,
            example_age, example_age**2], dim=1)
        # Fusion all features
        user_feature = self.fusion_layer(user_feature)
        # Candidate generate
        user_feature = self.candidate_generate(user_feature)
        # Sample embedding
        sample_features = self.embed_sample(samples)
        return user_feature, sample_features

    def loss(self, anchor_feature, sample_features, postive_idx):
        return self.criterion(anchor_feature, sample_features, postive_idx)

    def _build_model(self, input_size, in_channel, output_size, n_layer):
        dim = in_channel
        # First Layer
        layers = [nn.BatchNorm1d(input_size),
                  nn.Linear(input_size, dim), nn.ReLU()]
        # Hidden layers
        for _ in range(n_layer-2):
            layers += [nn.Linear(dim, dim//2), nn.ReLU()]
            dim = dim // 2
        # Last layer
        layers += [nn.Linear(dim, output_size), nn.ReLU()]
        return nn.Sequential(*layers)

    def predict(self, anchor_feature, sample_features, confidence_only=False,
                top_k=None, threshold=None):
        """
            Args:
                confidence_only (bool) : only return confidences in each sample
                                         (using in evaluation)
                top_k           (int)  : top k tags
                threshold       (float): return tags which confidence larger
                                         than threshold
            Return:
                confidences (torch.FloatTensor): confidence in each sample
                                                 dimension: (#users, #samples)
                samples     (list of tensor)   : predicted samples
        """
        similarity = torch.matmul(anchor_feature, torch.t(sample_features))
        confidences = F.relu(similarity)
        if confidence_only: return confidences

        # Ranking by confidence
        confidences, samples = torch.sort(confidences, descending=True)
        if top_k:
            samples = samples[:, :top_k]
            confidences = confidences[:, :top_k]
        if threshold:
            indexes = confidences >= threshold
            samples = [sample[idx] for sample, idx in zip(samples, indexes)]
        return samples


class MeanRepresents(nn.Module):
    def __init__(self, multilingual=None):
        super().__init__()
        self.sbert = SentenceBERT(multilingual)

    def forward(self, video_list, corpus):
        corpus_embeddings = self.sbert(corpus)
        corpus_dict = dict(zip(video_list, corpus_embeddings))
        return corpus_embeddings, corpus_dict

    def extract_user_feature(self, watched_videos, corpus_dict):
        users_represent = []
        for video_id in watched_videos:
            if video_id in corpus_dict:
                users_represent.append(corpus_dict[video_id])

        if len(users_represent) == 0:
            return None
        users_represent = torch.stack(users_represent, dim=0)
        users_represent = torch.mean(users_represent, dim=0)
        return users_represent

    def recommend(self, user_represents, corpus_embeddings):
        sim_mtx = self.similarity(user_represents, corpus_embeddings)
        rank_idx = torch.argsort(sim_mtx, dim=1, descending=True)
        recommended = [self.corpus[idx] for idx in rank_idx]
        return recommended

    def similarity(self, qurry_features, corpus_features):
        """
            Args:
                qurry_features  (tensor): shape is (num_querry, feature_dim)
                corpus_features (tensor): shape is (len_corpus, feature_dim)
            Return:
                sim             (tensor): similarity between querry and corpus
                                          with shape (num_querry, len_corpus)
        """
        norm_q = torch.norm(qurry_features, dim=1, keepdim=True)
        norm_c = torch.norm(corpus_features, dim=1, keepdim=True)
        qurry_features /= norm_q
        corpus_features /= norm_c
        sim = torch.mm(qurry_features, corpus_features.T)
        return sim


class SentenceBERT():
    """ Sentence BERT
        Args:
            xs  (list): sentences ["doc1", "doc2", ...]
        Returns:
        sentence_embeds (tensor): senmtence embeddings
    """
    def __init__(self, multilingual=None):
        model = 'distiluse-base-multilingual-cased' if multilingual else\
                'distilbert-base-nli-stsb-mean-tokens'
        self.model = SentenceTransformer(model)
        self.embed_size = 512 if multilingual else 786

    def __call__(self, xs):
        sentence_embeds = self.model.encode(xs)
        sentence_embeds = torch.FloatTensor(sentence_embeds)
        return sentence_embeds


class BERT():
    """ BERT
        Args:
            xs  (list): words [word1, word2, ...]
        Returns:
            word_embeds (np.array): word embeddings
    """
    def __init__(self):
        bert_type = "bert-base-uncased"
        bert_config = BertConfig.from_pretrained(bert_type,
                                                 output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_type)
        self.model = BertModel.from_pretrained(bert_type, config=bert_config)
        self.model.eval()

        self.layers = [-2]

    def __call__(self, xs):
        word_embed_list = []
        for i, x in enumerate(xs, 1):
            word_embed_list.append(self.bert_embed(x))

            if i % 100 == 0:
                print(f"[{i}/{len(xs)}]", end='\r')

        word_embeds = np.stack(word_embed_list)
        return word_embeds

    def bert_embed(self, text):
        # Get tokens from words
        sentence = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(sentence)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Single sentence
        segments_ids = [1] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids])

        # Forward
        with torch.no_grad():
            hidden_states = self.model(tokens_tensor, segments_tensors)[2]

        # Extract hidden state features by layers
        word_embed = []
        for layer in self.layers:
            embed = hidden_states[layer][0, 1:-1, :]
            embed = torch.sum(embed, dim=0)
            word_embed.append(embed)
        word_embed = torch.stack(word_embed)
        word_embed = torch.sum(word_embed, dim=0)
        return word_embed


if __name__ == '__main__':
    input_size, in_channel, output_size, n_layer = 000, 000, 000, 000
    model = YTRSRanking(input_size, in_channel, output_size, n_layer)

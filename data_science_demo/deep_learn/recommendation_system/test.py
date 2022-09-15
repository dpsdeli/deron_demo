from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import os
from os.path import join as PJ
import torch
from torch.utils.data import DataLoader

from dataset import VTRSDataset
from data_toolkit.utils import config, env_argument_parser
from data_toolkit.utils import Messager, GoogleStorage
from data_toolkit.utils import message_config, storage_config
from evaluation import precision_k, recall_k, f1_score_k
from networks import (MeanRepresents, SentenceBERT,
                      YTRSEmbedding, YTRSCandidateGenerate, YTRSRanking)
from utils import download_dataset

load_dotenv()


def testing_candidate_generate(embed_model, candidate_model, samples, video_feature_dict, dataset, dataloader, at_k, device, distribution=None):
    display_iter = len(dataloader) // 20 or 1
    embed_model.eval(), candidate_model.eval()
    # Extract sample features
    print("Extract sample features")
    info_names = ['_times', 'z_subtitle', 'j_subtitle']
    sample_info = dataset._extract_info(samples.tolist(), info_names)
    sample_features = dataset._extract_features(
            samples.tolist(), dataset.feature_pad['video'])
    sample_info = torch.FloatTensor(sample_info)
    sample_features = torch.cat((sample_features, sample_info), 1)

    # Testing
    print("Testing")
    true_dist = None
    y_trues, y_preds = [], []
    for it, (positive_samples, batch_inputs,
             num_positive) in enumerate(dataloader, 1):
        watched_video_features, collect_video_features = batch_inputs[:2]
        example_age, gender, language, _ = batch_inputs[2:]

        # Drop to device
        sample_features = sample_features.to(device).detach()
        watched_video_features = watched_video_features.to(device).detach()
        collect_video_features = collect_video_features.to(device).detach()
        language = language.to(device).detach()
        example_age = example_age.to(device).detach()
        gender = gender.to(device).detach()

        # Embedding
        embed_samples = embed_model.embed_samples(sample_features)
        avg_embed_video, avg_embed_collect_video, embed_language = \
            embed_model.candidate(
                watched_video_features, collect_video_features, language)

        # Candidate generate
        predict = candidate_model.predict(
            avg_embed_video, avg_embed_collect_video, example_age, gender,
            embed_language, embed_samples)

        # Ranking videos
        batch_index = torch.argsort(predict, dim=1, descending=True)
        for index, label, num in zip(batch_index,
                                     positive_samples, num_positive):
            y_preds.append(samples[index].tolist()[:at_k])
            y_trues.append(label[:num].tolist())

        if it % display_iter == 0:
            print(f"it: [{it+1:03d}/{len(dataloader):03d}]", end='\r')
    eval_results = {'precision': precision_k(y_trues, y_preds, k=at_k),
                    'recall': recall_k(y_trues, y_preds, k=at_k),
                    'f1_score': f1_score_k(y_trues, y_preds, k=at_k)}

    display_distribution(y_trues, y_preds, true_dist, distribution)
    return eval_results


def display_distribution(y_trues, y_preds, true_dist, distribution):
    if distribution:
        # The distribution of top 50 recommended videos
        print("(Distribution Predict:")
        y_preds = np.asarray(y_preds)[:, :50].reshape(-1)
        counts = _calculate_distrubution(y_preds)
        counts = '%, '.join([str(c)[:6] for c in counts[-10:].tolist()])
        print(f" {counts}%")

        if not true_dist:
            temp = []
            for y_true in y_trues:
                temp += y_true
            counts = _calculate_distrubution(temp)
            true_dist = '%, '.join([str(c)[:6] for c in counts[-10:].tolist()])
        print(" Distribution Label:")
        print(f" {true_dist}%")
        print(")")
    return true_dist


def _calculate_distrubution(y):
    _, counts = np.unique(y, return_counts=True)
    counts = np.sort(counts[-100:]) / np.sum(counts[-100:])
    counts *= 100
    return counts


def load_video_features(source, target, storage):
    if not os.path.exists(target):
        storage.download(source=source, target=target)
    video_feature_dict = torch.load(target)
    return video_feature_dict


if __name__ == '__main__':
    torch.manual_seed(2020)
    start_time = datetime.now()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ##################################################
    # Environment setting
    ##################################################
    storage_directory = "./deep_learn/recommendation_system/"
    dataset_name = 'Dataset'

    # Parse arguments
    parser = env_argument_parser()
    parser.add_argument('--exp', '-e', type=str, default='exp3')
    parser.add_argument('--model_weights', '-w', type=str, default='')
    parser.add_argument('--k', type=int, default=100, help="Evaluate @K")
    parser.add_argument('--testset', type=int, default=1,
                        help="1: testset, 0: valset, 2: trainset")
    parser.add_argument('--download', type=int, default=0,
                        help="Whether download data from storage")
    args = parser.parse_args()

    # Google storage
    storage_config = storage_config(args, project_id=os.environ['PROJECT'])
    storage = GoogleStorage(storage_config)
    # Messager
    msg_config = message_config(args)
    message = Messager(msg_config)
    # Config
    config_path = f"./deep_learn/recommendation_system/configs/{args.exp}.yaml"
    config = config(config_path)
    exp_name = config['exp_name']

    # Path
    data_root = config['data_root']
    current_directory = "./deep_learn/recommendation_system"
    local_path = {
        'exp_dir': PJ(current_directory, f"results/{exp_name}"),
        'model_dir': PJ(current_directory, f"results/{exp_name}/model"),
        'log_dir': PJ(current_directory, "results/logs"),
        'data_root': config['data_root'],
        'sbert_feature': PJ(current_directory, "features/title_sbert_features.pt")}

    storage_directory = "vt-datascientist/deep_learn/recommendation_system"
    storage_path = {
        'model_dir': f"{storage_directory}/results/{exp_name}/model",
        'log_dir': f"{storage_directory}/results/logs",
        'data_root': "vt-datascientist/dataset/VTRecSysDataset/",
        'sbert_feature': f"{storage_directory}/features/title_sbert_features.pt"}

    # Show experiment info
    message(f"> EXP: {exp_name} | Model: {config['model']} | Testing")
    message(f"Weights: {args.model_weights}")

    ##################################################
    # Data loader
    ##################################################
    # Load video title features from SBERT
    video_feature_dict = load_video_features(
        storage_path['sbert_feature'],
        local_path['sbert_feature'], storage)
    input_size, corpus_size = config['input_size'], config['corpus_size']
    feature_pad = {'v': torch.zeros([input_size['v']]),
                   'k': torch.zeros([input_size['k']])}

    message("> Data loader")

    # Download dataset from google storage
    if args.download:
        message("Download dataset")
        download_dataset(storage, storage_path['data_root'],
                         local_path['data_root'])
        message(f"Save in {local_path['data_root']}")
    # Dataset
    test_file = {1: 'test_file', 0: 'val_file', 2: 'train_file'}
    test_file = config[test_file[args.testset]]

    testset = VTRSDataset(
        config['data_root'], test_file,
        n_day=config['N_day'], t_max=config['t_max'],
        feature_pad=feature_pad, video_feature_dict=video_feature_dict,
        num_watched_videos=config['num_watched_videos'],
        num_searched_keywords=config['num_searched_keywords'],
        num_negative_samples=config['num_negative_samples'],
        num_positive_samples=config['num_positive_samples'],
        in_train=False)
    # Dataloader
    testloader = DataLoader(testset, config['test_size'],
                            shuffle=False, num_workers=config['num_workers'])

    message(f"Number of videos in corpus: {testset.corpus_size}")
    message(f"Test users: {len(testset)}")
    message(f"Number of batch: {len(testloader)}/epoch")

    ##################################################
    # Build model
    ##################################################
    message("> Build model")
    if config['model'] == 'MeanRepresents':
        model = MeanRepresents(corpus=None)
    elif config['model'] == 'YoutubeDeepRecSys':
        input_size, corpus_size = config['input_size'], config['corpus_size']
        embed_model = YTRSEmbedding(
            input_size['video'], input_size['keyword'],
            corpus_size['location'], corpus_size['platform'],
            corpus_size['language'], corpus_size['cefr'],
            corpus_size['category'], config['embed_size'])
        candidate_model = YTRSCandidateGenerate(
            input_size['candidate_cat'], config['in_channel'],
            config['output_size'], config['n_layer'])
        ranking_model = YTRSRanking(
            input_size['ranking_cat'], config['in_channel'],
            config['output_size'], config['n_layer'])
        sbert = SentenceBERT(multilingual=True)

        embed_model.to(device)
        candidate_model.to(device)
        ranking_model.to(device)

    # Load SBERT features
    if config.get('SBERT_feature', None):
        tag_feature_path = PJ(current_directory,
                              f"features/{config['SBERT_feature']['tag']}")
        video_feature_path = PJ(current_directory,
                                f"features/{config['SBERT_feature']['video']}")
        tag_features_matrix = torch.load(tag_feature_path)
        video_features_dict = torch.load(video_feature_path)

    model_path = PJ(local_path['model_dir'], args.model_weights)
    # Download model weight
    if not os.path.exists(model_path):
        source = f"{storage_path['model_dir']}/{args.model_weights}"
        storage.download(source=source, target=model_path)

    state_dict = torch.load(model_path, map_location=device)
    embed_model.load_state_dict(state_dict['embed_model'])
    candidate_model.load_state_dict(state_dict['candidate_model'])
    ranking_model.load_state_dict(state_dict['ranking_model'])

    # Load SBERT features
    if config.get('SBERT_feature', None):
        tag_feature_path = PJ(current_directory,
                              f"features/{config['SBERT_feature']['tag']}")
        video_feature_path = PJ(current_directory,
                                f"features/{config['SBERT_feature']['video']}")
        tag_features_matrix = torch.load(tag_feature_path)
        video_features_dict = torch.load(video_feature_path)

    ########################################
    # Evaluate
    ########################################
    message("> Evaluation")

    embed_model.eval(), candidate_model.eval(), ranking_model.eval()
    torch.set_grad_enabled(False)

    samples = testset.video_corpus
    eval_results = testing_candidate_generate(
        embed_model, candidate_model, sbert, samples, video_feature_dict,
        testloader, args.k, device, distribution=True)

    eval_msg = f"Precision@{args.k}: {eval_results['precision']:.6f} | " +\
               f"Recall@{args.k}: {eval_results['recall']:.6f} | " +\
               f"F1_Score@{args.k}: {eval_results['f1_score']:.6f}\n"
    message(eval_msg)

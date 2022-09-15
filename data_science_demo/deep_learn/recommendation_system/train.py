from datetime import date, datetime
from dotenv import load_dotenv
import numpy as np
import os
from os.path import join as PJ
import shutil
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, lr_scheduler

from dataset import VTRSDataset
from data_toolkit.utils import config, check_directory, env_argument_parser
from data_toolkit.utils import Messager, GoogleStorage
from data_toolkit.utils import message_config, storage_config
from networks import (MeanRepresents, YTRSEmbedding, YTRSCandidateGenerate)
from test import testing_candidate_generate

'''
######################################################################

######################################################################
TODO:
    1. Change n day (decide label) in training

Hint:
    Run extract_feature.py first to get video title features
######################################################################
'''


# Environment setting
def build_data_path(data_root, exp_name, local_root, storage_root):
    local_path = {
        'exp_dir': PJ(local_root, f"results/{exp_name}"),
        'model_dir': PJ(local_root, f"results/{exp_name}/model"),
        'log_dir': PJ(local_root, "results/logs"),
        'data_root': data_root,
        'sbert_feature': PJ(local_root, "features/title_sbert_features.pt")}
    storage_path = {
        'model_dir': f"{storage_root}/results/{exp_name}/model",
        'log_dir': f"{storage_root}/results/logs",
        'data_root': "./dataset/VTRecSysDataset/",
        'sbert_feature': f"{storage_root}/features/title_sbert_features.pt"}
    return local_path, storage_path


# Data loader
def download_dataset(storage, source_directory, target_directory):
    file_list = storage.list_dir(prefix=source_directory)
    for file_path in file_list:
        if file_path == source_directory:
            continue
        name = file_path.replace(source_directory, '')
        storage.download(source=file_path, target=PJ(target_directory, name))


def load_video_features(source, target, storage):
    if not os.path.exists(target):
        storage.download(source=source, target=target)
    video_feature_dict = torch.load(target)
    return video_feature_dict


def dataset_and_dataloader(config, dataset, data_file, feature_pad, video_feature_dict, set_type):
    shuffle, batch_size, in_train = \
        (True, 'batch_size', True) if set_type == 'train' else \
        (False, 'test_size', False)

    data_set = VTRSDataset(
        config['data_root'], data_file,
        n_day=config['N_day'], t_max=config['t_max'],
        feature_pad=feature_pad, video_feature_dict=video_feature_dict,
        num_watched_videos=config['num_watched_videos'],
        num_negative_samples=config['num_negative_samples'],
        num_positive_samples=config['num_positive_samples'],
        num_collect_videos=config['num_collect_videos'],
        in_train=in_train, entire_video_corpus=config['entire_corpus'])

    dataloader = DataLoader(data_set, config[batch_size],
                            shuffle=shuffle, num_workers=config['num_workers'])
    return data_set, dataloader


# Optimizer
def trainable_parameters(config, embed_model, candidate_model):
    trainable, parameters = config['train'], []

    parameters = (parameters + list(embed_model.parameters())
                  if trainable['embed_model'] else parameters)
    parameters = (parameters + list(candidate_model.parameters())
                  if trainable['candidate_model'] else parameters)
    return parameters


def download_last_train_results(source_dir, target_dir, optimizer_file):
    # List all files in google storage
    model_weight_list = storage.list_dir(prefix=source_dir)
    # Pick last training model weight
    model_weight_list.remove(f"{source_dir}/{optimizer_file}")
    last_model_path = model_weight_list[-1]
    last_model_name = last_model_path.split('/')[-1]

    # Download model and optimizer
    storage.download(source=f"{source_dir}/{last_model_name}",
                     target=PJ(target_dir, last_model_name))
    storage.download(source=f"{source_dir}/{optimizer_file}",
                     target=PJ(target_dir, optimizer_file))


def resume_training(embed_model, candidate_model, optimizer, model_directory, optimizer_file, device):
    # Last saved model file (by file name)
    model_weights = os.listdir(model_directory)
    model_weights.remove(optimizer_file)
    model_weights.sort()
    last_model_name = model_weights[-1]
    # Load model and optimizer
    last_model_path = PJ(model_directory, last_model_name)
    optimizer_path = PJ(model_directory, optimizer_file)
    state_dict = torch.load(last_model_path, map_location=device)
    embed_model.load_state_dict(state_dict['embed_model'])
    candidate_model.load_state_dict(state_dict['candidate_model'])
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    # Calculate last epoch and iteration
    last_epoch, iteration = last_model_name.split('_')
    last_epoch, iteration = int(last_epoch), int(iteration[:-3])
    return iteration, last_epoch, last_model_name


# Training
def training_candidate_generate(embed_model, candidate_model, trainset, dataloader, optimizer, scheduler, writer, _config):
    log_iter, epoch, max_epoch, iteration = (
        _config["log_iter"], _config['epoch'], _config['max_epoch'],
        _config['iteration'])

    num_batch = len(dataloader)
    batch_loss = 0.
    for it, (positive_sample_features, positive_samples,
             batch_inputs, num_positive) in enumerate(dataloader, 1):
        iteration += 1
        optimizer.zero_grad()

        watched_video_features, collect_video_features = batch_inputs[:2]
        example_age, gender, language, _ = batch_inputs[2:]

        # Negative sampling
        negative_sample_features = _negative_sampling(trainset, num_positive,
                                                      positive_samples)

        # Drop to device
        positive_sample_features = positive_sample_features.to(device).detach()
        negative_sample_features = negative_sample_features.to(device).detach()
        watched_video_features = watched_video_features.to(device).detach()
        collect_video_features = collect_video_features.to(device).detach()
        language = language.to(device).detach()
        example_age = example_age.to(device).detach()
        gender = gender.to(device).detach()

        # Embedding
        embed_pos_samples = embed_model.embed_samples(positive_sample_features)
        embed_neg_samples = embed_model.embed_samples(negative_sample_features)
        avg_embed_video, avg_embed_collect_video, embed_language = \
            embed_model.candidate(
                watched_video_features, collect_video_features, language)

        # Candidate generate
        predict_pos, predict_neg = candidate_model(
            avg_embed_video, avg_embed_collect_video, example_age, gender,
            embed_language, embed_pos_samples, embed_neg_samples)

        # Backpropagation
        loss = candidate_model.loss(predict_pos, predict_neg, num_positive,
                                    w_pos=config['weight_postive'],
                                    w_neg=config['weight_negative'])
        loss.backward()
        optimizer.step()

        # save trainig log
        batch_loss += loss.item()
        if it % log_iter == 0:
            batch_loss /= log_iter
            writer.add_scalar("loss", batch_loss, iteration)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], iteration)
            print(f"Epoch: [{epoch+1}/{max_epoch}] " +
                  f"Iter: {it}/{num_batch} loss: {batch_loss:.6f}")
            batch_loss = 0.

    scheduler.step()
    return iteration


def _negative_sampling(dataset, num_positive, positive_samples):
    # Positive set
    postive_set = []
    for num, samples in zip(num_positive, positive_samples):
        postive_set += samples[:num].tolist()
    postive_set = list(set(postive_set))

    # Sampling and remove positive
    num = dataset.num_negative_samples + len(postive_set)
    idx = torch.randperm(dataset.corpus_size)[:num]
    negative_samples = dataset.video_corpus[idx].tolist()
    for sample in np.intersect1d(postive_set, negative_samples):
        negative_samples.remove(sample)
    negative_samples = negative_samples[:dataset.num_negative_samples]

    # Extract negative sample features
    info_names = ['_times', 'z_subtitle', 'j_subtitle']
    sample_info = dataset._extract_info(negative_samples, info_names)
    negative_sample_features = dataset._extract_features(
        negative_samples, dataset.feature_pad['video'])
    sample_info = torch.FloatTensor(sample_info)
    negative_sample_features = torch.cat(
        (negative_sample_features, sample_info), 1)
    return negative_sample_features


# Save model and training log
def save_and_upload(models, optimizer, save_directory, model_name, optimizer_file, storage=None, target_directory=None, upload=False):
    # Models
    source_model = PJ(save_directory, model_name)
    model_dict = {'embed_model': models[0].state_dict(),
                  'candidate_model': models[1].state_dict()}
    torch.save(model_dict, source_model)
    # Optimizer
    source_optimizer = PJ(save_directory, optimizer_file)
    torch.save(optimizer.state_dict(), source_optimizer)

    if upload:
        storage.upload(source_model,
                       f"{storage_path['model_dir']}/{model_name}")
        storage.upload(source_optimizer,
                       f"{storage_path['model_dir']}/{optimizer_file}")


def upload_log(storage, log_dir, storage_log_dir):
    names = os.listdir(log_dir)
    names.sort()
    success = storage.upload(PJ(log_dir, names[-1]),
                             f"{storage_log_dir}/{names[-1]}")
    return success


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ##################################################
    # Environment setting
    ##################################################
    # Parse arguments
    parser = env_argument_parser()
    parser.add_argument('--exp', '-e', type=str, default='exp5')
    parser.add_argument('--k', type=int, default=100, help="Evaluate @K")
    parser.add_argument('--download', type=int, default=0,
                        help="Whether download data from storage")
    args = parser.parse_args()

    # Google storage
    storage_config = storage_config(args, project_id=os.environ['P'])
    storage = GoogleStorage(storage_config)
    # Messager
    msg_config = message_config(args)
    message = Messager(msg_config)
    # Config
    config_path = f"./deep_learn/recommendation_system/configs/{args.exp}.yaml"
    config = config(config_path)
    exp_name = config['exp_name']
    # TODO 1: Change n day (decide label) in training
    n_day = config['N_day']

    # Path
    data_root = config['data_root']
    current_directory = "./deep_learn/recommendation_system"
    storage_directory = "./deep_learn/recommendation_system"
    local_path, storage_path = build_data_path(
        data_root, exp_name, current_directory, storage_directory)
    optimizer_file = 'optimizer.pt'

    # Create directory
    check_directory(*[
        local_path['model_dir'], local_path['log_dir'], data_root,
        PJ(data_root, 'index'), PJ(data_root, 'data_list'),
        PJ(current_directory, 'features')])

    # Saving config file
    file = f"{exp_name}_{date.today().strftime('%Y%m%d')}.yaml"
    shutil.copy(config_path, PJ(local_path['exp_dir'], file))

    # Tensorboad
    writer = SummaryWriter(PJ(local_path['log_dir'], exp_name))

    # Show experiment info
    message(f"> EXP: {exp_name} | Model: {config['model']}")
    start_time = datetime.now()

    ##################################################
    # Data loader
    ##################################################
    message("> Data loader")
    # Download dataset from google storage
    if args.download:
        message("Download dataset from storage")
        download_dataset(storage, storage_path['data_root'],
                         local_path['data_root'])
        message(f"Save in {local_path['data_root']}")

    # Load video title SBERT features
    video_feature_dict = load_video_features(storage_path['sbert_feature'],
                                             local_path['sbert_feature'],
                                             storage)
    size = video_feature_dict[1].shape[0]
    feature_pad = {'vi': torch.zeros([size])}

    # Dataset and data loader
    trainset, trainloader = dataset_and_dataloader(
        config, VTRSDataset, config['train_file'], feature_pad,
        video_feature_dict, set_type='train')
    valset, valloader = dataset_and_dataloader(
        config, VTRSDataset, config['val_file'], feature_pad,
        video_feature_dict, set_type='val')

    message(f"Number of videos in corpus: {trainset.corpus_size}")
    message(f"Train users: {len(trainset)} | Val users: {len(valset)}")
    message(f"Number of batch: {len(trainloader)}/epoch")

    ##################################################
    # Build model with loss function
    ##################################################
    message("> Build model")
    if config['model'] == 'MeanRepresents':
        model = MeanRepresents(corpus=None)
    elif config['model'] == 'YoutubeDeepRecSys':
        input_size, corpus_size = config['input_size'], config['corpus_size']
        embed_model = YTRSEmbedding(input_size['video'], config['embed_size'],
                                    corpus_size, gru=config['fusion_gru'])
        candidate_model = YTRSCandidateGenerate(
            input_size['candidate_cat'], config['in_channel'],
            config['output_size'], config['n_layer'])

        embed_model.to(device)
        candidate_model.to(device)

    ##################################################
    # Optimizer
    ##################################################
    message("> Create optimizer")
    parameters = trainable_parameters(config, embed_model, candidate_model)
    if config['optimizer'] == 'SGD':
        optimizer = SGD(parameters, lr=config["lr"])
    elif config['optimizer'] == 'Adam':
        optimizer = Adam(parameters, lr=config["lr"])
    else:
        message(f"Not support optimizer: {config['optimizer']}")
        raise ValueError(f"Not support optimizer: {config['optimizer']}")

    iteration, last_epoch = 0, 0
    # Resume training process
    if config['resume']:
        message("Resume training")
        if optimizer_file not in os.listdir(local_path['model_dir']):
            message(f"Download from {storage_path['model_dir']}")
            download_last_train_results(storage_path['model_dir'],
                                        local_path['model_dir'],
                                        optimizer_file)
        iteration, last_epoch, last_model_name = \
            resume_training(embed_model, candidate_model, optimizer,
                            local_path['model_dir'], optimizer_file, device)
        message(f"Loading model {last_model_name} and optimizer successed!")

    # Learning rate decay scheduler
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, config["step_size"], config["gamma"],
        last_epoch=last_epoch-1)

    ##################################################
    # Start training
    ##################################################
    best = {'epoch': -1, 'performance': 0, 'weight': None}
    for epoch in range(last_epoch, config["max_epoch"]):
        message(f"> Training [{(epoch+1)}/{config['max_epoch']}]")
        _config = {'epoch': epoch,
                   'iteration': iteration,
                   'max_epoch': config["max_epoch"],
                   'log_iter': config["log_iter"]}
        iteration = training_candidate_generate(
            embed_model, candidate_model, trainset, trainloader, optimizer,
            scheduler, writer, _config)

        ##################################################
        # Evaluation
        ##################################################
        message("> Evaluation (validation set)")
        embed_model.eval(), candidate_model.eval()
        torch.set_grad_enabled(False)

        samples = valset.video_corpus
        eval_results = testing_candidate_generate(
            embed_model, candidate_model, samples, video_feature_dict,
            valset, valloader, args.k, device, distribution=True)

        eval_msg = f"Epoch: [{epoch + 1}] " +\
                   f"Best: {best['performance']:.6f} ({best['epoch']})\n" +\
                   f"Precision@{args.k}: {eval_results['precision']:.6f} | " +\
                   f"Recall@{args.k}: {eval_results['recall']:.6f} | " +\
                   f"F1_Score@{args.k}: {eval_results['f1_score']:.6f}"
        message(eval_msg)

        # Record results in log
        writer.add_scalar(f"performance/Precision@{args.k}/val",
                          eval_results['precision'], epoch+1)
        writer.add_scalar(f"performance/Recall@{args.k}/val",
                          eval_results['recall'], epoch+1)
        writer.add_scalar(f"performance/F1-Score@{args.k}/val",
                          eval_results['f1_score'], epoch+1)

        # Save and upload model and optimizer
        recall = eval_results['recall']
        if recall > best['performance'] or best['performance'] == 0:
            best['epoch'], best['performance'] = epoch + 1, recall

            models = [embed_model, candidate_model]
            save_and_upload(models, optimizer, local_path['model_dir'],
                            f'{epoch+1:04d}_{iteration:06d}.pt',
                            optimizer_file, storage, storage_path['model_dir'],
                            upload=True)
            message("Saved and uploaded model and optimizer")
        # Save and upload log
        success = upload_log(
            storage,  log_dir=PJ(local_path['log_dir'], exp_name),
            storage_log_dir=f"{storage_path['log_dir']}/{exp_name}")
        message("Saved and uploaded log\n")

        torch.set_grad_enabled(True)
        embed_model.train(), candidate_model.train()

    models = [embed_model, candidate_model]
    save_and_upload(models, optimizer, local_path['model_dir'],
                    f'{epoch+1:04d}_{iteration:06d}.pt',
                    optimizer_file, storage, storage_path['model_dir'],
                    upload=True)
    message("Training done. Saved and uploaded model and optimizer")

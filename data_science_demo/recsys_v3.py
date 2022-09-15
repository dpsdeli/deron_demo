from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv
import os
from os.path import join as PJ
import pytz
import torch
from torch.utils.data import DataLoader

from data_toolkit.utils import (
    env_argument_parser, message_config, storage_config, Messager,
    GoogleStorage, check_directory, google_datastore,
    pubsub_publish, config, load_torch_data)
from data_toolkit.task_manager import TaskManager
from deep_learn.recommendation_system.dataset import VTRSDatasetInference
from deep_learn.recommendation_system.networks import (YTRSEmbedding,
                                                       YTRSCandidateGenerate)
from data_filter import WebVideoFilter, AutoImportVideoFilter, ProVideoFilter

load_dotenv()

"""
###########################################################################
# Personal Recommendation system
###########################################################################
    # Check exists first
    ## Data
    1. Check have ./dataset/RecSysDataset_infer/index/***.csv
       (7 index files)
    ## Model
    1. Check have ./model/modeler/title_sbert_features.pt
    2. Check have ./model/modeler/YTRS16/***.pt
       (model weight)

    ----------------------------------------------------------------------
    ## Pub/sub data format:
        {"userId": 55555555555,
         "videoList":[
            {"language": "a", "videoIds" : [...]},
            {"language": "b", "videoIds" : [...]},
            {"language": "c", "videoIds" : [...]}],
         "videoSource": "pro"
        }

    ----------------------------------------------------------------------
    Run
    > Recommendation with top z videos, only recommend new videos which
      edited within x months, recommend users who watched during y weeks,
      , download dataset, and filter and record videos for Web and App.
        python3 -m recsys_proj.recsys_model_v3_3_0 \
        --exp exp6 -w 0018_003204.pt --download 1 \
        --duration 6 --top_k 10 --unused_weeks 3 \
        --platform "App Web" \
        --in_slack 0 --in_cmd 1

    ----------------------------------------------------------------------
    TODO:
    1. Add pub/sub into pubsub_manager
###########################################################################
"""


# Environment setting
def argument_parser():
    parser = env_argument_parser()
    # Model information and trained weights
    parser.add_argument('--exp', '-e', type=str, default='exp3')
    parser.add_argument('--model_weights', '-w', type=str, default='')
    # Data related
    parser.add_argument('--download', type=int, default=0,
                        help="Whether download data from storage")
    # Recommendation result related
    parser.add_argument('--platform', default='App', type=str,
                        help="Web, App, or Auto import")
    parser.add_argument('--top_k', default=250, type=int,
                        help="Number of recommended videos")
    parser.add_argument('--duration', default=6,
                        help="Only recommend videos which created within `duration` months")
    parser.add_argument('--unused_weeks', default=3, type=int,
                        help="Definition in unused user (no recommend)")
    parser.add_argument('--max_users', default=None,
                        help="Maximum number of user will be recommend")
    return parser


def build_data_path(model_weights, local_model_dir, storage_model_dir):
    local_path = {
        'model_dir': PJ(local_model_dir, "YTRS16"),
        'model': PJ(local_model_dir, f"YTRS16/{model_weights}"),
        'sbert_feature': PJ(local_model_dir, "title_sbert_features.pt"),
        'data_root': PJ(local_model_dir, "dataset/VTRecSysDataset_infer")}

    storage_path = {
        'model': f"{storage_model_dir}/YTRS16/{model_weights}",
        'sbert_feature': f"{storage_model_dir}/title_sbert_features.pt",
        'data_root': "./dataset/VTRecSysDataset_infer/"}

    return local_path, storage_path


def download_dataset(storage, source_directory, target_directory):
    file_list = storage.list_dir(prefix=source_directory)
    for file_path in file_list:
        if file_path == source_directory:
            continue
        name = file_path.replace(source_directory, '')
        storage.download(source=file_path, target=PJ(target_directory, name))


def download_model_weights(local_path, storage_path):
    try:
        storage.download(source=storage_path['model'],
                         target=local_path['model'])
    except Exception as e:
        message(f"[Error]: Missing model weights!\n{e}", tracker)
        exit()


def load_models(model_weights_path, embed_model, candidate_model, device):
    state_dict = torch.load(model_weights_path, map_location=device)
    embed_model.load_state_dict(state_dict['embed_model'])
    candidate_model.load_state_dict(state_dict['candidate_model'])


def extract_video_features(videos, video_feature_dict, padding):
    video_features = [video_feature_dict.get(video_id, padding)
                      for video_id in videos.tolist()]
    video_features = torch.stack(video_features, dim=0)
    return video_features


class RecommendationSystem:
    def __init__(self, embed_model, candidate_model, samples, dataset, dataloader, platforms, datastore_kind, top_k, pubsub_topic, device, message):
        self.message = message
        # Model
        self.embed_model = embed_model
        self.candidate_model = candidate_model
        self.device = device
        # Data
        self.samples = samples
        self.dataloader = dataloader
        # Result
        self.platforms = platforms
        self.datastore_kind = datastore_kind
        self.pubsub_topic = pubsub_topic
        self.top_k = top_k
        self.num_candidate = 1000

        info_names = ['_times', 'z_subtitle', 'a_subtitle']
        self.sample_features = self._extract_sample_features(
            samples, info_names, dataset)
        languages = ['OA', 'OB', 'OC']
        self.filter_dict = self._build_video_filters(languages, platforms)

    def inference(self, dataloader):
        recommentation_dict = {}
        for it, (batch_users, batch_inputs) in enumerate(dataloader):
            if len(recommentation_dict) >= 1000:
                self.message(f"Inference [{it + 1} / {len(dataloader)}]")
            #############################################
            # Inference
            #############################################
            watched_video_features, collect_video_features = batch_inputs[:2]
            example_age, gender, language, last_time = batch_inputs[2:]

            # Drop to device
            data = [self.sample_features, watched_video_features,
                    collect_video_features, language, example_age, gender]
            (sample_features, watched_video_features, collect_video_features,
             language, example_age, gender) = self._drop_to(self.device, *data)

            # Embedding
            embed_samples = embed_model.embed_samples(sample_features)
            avg_embed_video, avg_embed_collect_video, embed_language = \
                embed_model.candidate(
                    watched_video_features, collect_video_features, language)

            # Candidate generate
            predicts = candidate_model.predict(
                avg_embed_video, avg_embed_collect_video, example_age, gender,
                embed_language, embed_samples)

            # Ranking videos
            rec_dict = self._ranking_videos(batch_users, predicts,
                                            self.samples)
            recommentation_dict = {**recommentation_dict, **rec_dict}

            #############################################
            # Filter and upload recommendation results
            #############################################
            # Upload results every 1000 users or last batch
            if len(recommentation_dict) >= 1000 or it >= len(dataloader) - 1:
                self.message("Filter and upload recommendation results")
                self.upload_recommemndation_results(recommentation_dict)
                recommentation_dict = {}

    def upload_recommemndation_results(self, recommentation_dict):
        update_time = datetime.now().astimezone(tw_timezone)
        for platform, video_filter in self.filter_dict.items():
            # Filter
            top_k = self.top_k[platform]
            filtered_recommend_dict = self._filter_and_convert_format(
                recommentation_dict, video_filter, platform,
                update_time, top_k)

            # Upload to datastore
            kind = self.datastore_kind[platform]
            google_datastore('upload', kind, filtered_recommend_dict,
                             to_datastore=args.to_datastore)

            # Publish to pub/sub
            if platform == 'App':
                for user_id, recommend_dict in filtered_recommend_dict.items():
                    recommend_dict.pop('entity', None)
                    self._upload_to_pubsub(
                        user_id, recommend_dict, self.pubsub_topic,
                        source='OOO')

    def _extract_sample_features(self, samples, info_names, dataset):
        sample_info = dataset._extract_info(samples.tolist(), info_names)
        sample_features = dataset._extract_features(
                samples.tolist(), dataset.feature_pad['video'])
        sample_info = torch.FloatTensor(sample_info)
        sample_features = torch.cat((sample_features, sample_info), 1)
        return sample_features

    def _build_video_filters(self, languages, platforms):
        filter_dict = {}
        for platform in platforms:
            if platform == 'W':
                condition = {'interval_month': 00000, 'duration_sec': 00000}
                video_filter = WebVideoFilter(condition)
                filter_dict[platform] = video_filter
            elif platform == 'A':
                video_filter = {}
                for lang in languages:
                    condition = {'interval_month': 3333, 'language': lang}
                    video_filter[lang] = ProVideoFilter(condition)
            elif platform == 'AB':
                condition = {'interval_month': 111111}
                video_filter = AutoImportVideoFilter(condition)
                filter_dict[platform] = video_filter
            else:
                self.message(f"[Error] Not support {platform} (A/AB//W)")
                exit()
            filter_dict[platform] = video_filter
        return filter_dict

    def _drop_to(self, device, *args):
        data = []
        for arg in args:
            data.append(arg.to(device).detach())
        return data

    def _ranking_videos(self, users, predicts, samples):
        batch_size = users.shape[0]
        batch_samples = samples.repeat(batch_size, 1)
        batch_index = torch.argsort(predicts, dim=1, descending=True)
        ranked_samples = torch.gather(batch_samples, 1, batch_index)
        recommentation_dict = {
            user.item(): ranked_sample.tolist()[:self.num_candidate]
            for user, ranked_sample in zip(users, ranked_samples)}
        return recommentation_dict

    def _filter_and_convert_format(self, recommentation_dict, video_filter, platform, update_time, top_k=20):
        # Filter
        if platform == 'A':
            recommend_dict = defaultdict(dict)
            for user, videos in recommentation_dict.items():
                for lang, pro_video_filter in video_filter.items():
                    videos = pro_video_filter(videos, data_type='list')[:top_k]
                    recommend_dict[user][f'v_{lang}'] = videos
                recommend_dict[user]['entity'] = update_time
            recommend_dict = dict(recommend_dict)
        else:
            recommend_dict = video_filter(recommentation_dict)
            # This is for change column name in datastore
            name = 'content_based' if platform == 'W' else 'AB'
            for user, v in recommentation_dict.items():
                recommend_dict[user] = {'v': v[:top_k],
                                        '_update_time': update_time,
                                        name: v[:top_k]}
        return recommend_dict

    def _upload_to_pubsub(self, r_id, recommend_dict, topic, source):
        lang_dict = {'videos_z': 'z',
                     'videos_e': 'e',
                     'videos_j': 'j'}
        # Data format
        data = {'r_Id': r_id, 'videoList': [], 'videoSource': source}
        for lang, video_list in recommend_dict.items():
            data['videoList'].append({'language': lang_dict.get(lang),
                                      'videoIds': video_list})
        try:
            pubsub_publish(data, topic)
        except Exception as e:
            message(e)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ######################################################################
    # Setting
    ######################################################################
    projectId = os.environ['P']
    tw_timezone = pytz.timezone('Asia/Taipei')

    parser = argument_parser()
    args = parser.parse_args()
    batch_size = 512
    num_workers = 2
    platforms = args.platform.split()
    top_k = {'A': args.top_k, 'W': args.top_k, 'AB': 500}
    duration = args.duration
    n_day = t_max = datetime.now().astimezone(tw_timezone).strftime("%Y-%m-%d")
    max_users = args.max_users

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

    # Path
    local_directory = "./_proj/model/modeler"
    storage_directory = "model/modeler"
    local_path, storage_path = build_data_path(
        args.model_weights, local_directory, storage_directory)

    # Pub/sub setting (for daily push)
    pubsub_topic = "updateList"
    # Google storage setting
    datastore_kind = {
        'W': '_w',
        'A': '_p',
        'AB': '_ab'}

    # Create directory
    check_directory(*[
        local_path['data_root'],
        local_path['model_dir'],
        PJ(local_path['data_root'], 'data_list'),
        PJ(local_path['data_root'], 'index')])

    task, program = "Recommendation system", "recsys_v3.py"
    message(f"> [{projectId}] [{task}]:\n{program}")
    # Tracing this program
    tracker = TaskManager(task, program)
    tracker.tracing()
    start_time = datetime.now()

    # Show inference information
    message(f"_Model   : {config['model']} | Inference_", tracker)
    message(f"_Weights : {args.model_weights}_", tracker)
    message(f"_Platform: {', '.join(platforms)}_", tracker)

    ######################################################################
    # Data loader
    ######################################################################
    # Load video title features from SBERT
    video_feature_dict = load_torch_data(
        storage_path['sbert_feature'], local_path['sbert_feature'],
        storage, download=True, message=message)
    size = video_feature_dict[1].shape[0]
    feature_pad = {'v': torch.zeros([size])}

    message("Data loader", paragraph=True)

    # Download dataset from google storage
    if args.download:
        source, target = storage_path['data_root'], local_path['data_root']
        message(f"Download dataset `{source}` to `{target}`")
        download_dataset(
            storage, storage_path['data_root'], local_path['data_root'])

    # Dataset
    if 'AB' in platforms:
        config['entire_corpus'] = True
    inferset = VTRSDatasetInference(
        local_path['data_root'], 'infer_list.txt', n_day=n_day, t_max=t_max,
        feature_pad=feature_pad, video_feature_dict=video_feature_dict,
        num_watched_videos=config['num_watched_videos'],
        num_collect_videos=config['num_collect_videos'],
        entire_video_corpus=config['entire_corpus'],
        week=args.unused_weeks)
    if max_users:
        inferset.user_list = inferset.user_list[-int(max_users):]

    # Dataloader
    inferloader = DataLoader(inferset, batch_size,
                             shuffle=True, num_workers=num_workers)
    # Video corpus
    samples = inferset.video_corpus
    condition = {'interval_month': duration, 'duration_sec': 3600}
    video_filter = WebVideoFilter(condition)
    if 'AB' in platforms:
        video_filter = AutoImportVideoFilter(condition)
    samples = video_filter(samples.tolist(), data_type='list')
    samples = torch.LongTensor(samples)

    message(f"Number of videos in corpus: {len(samples)}", tracker)
    message(f"Number of users need recommentation: {len(inferset)}", tracker)

    ##################################################
    # Build and load trained model
    ##################################################
    message("Build model", paragraph=True)
    input_size, corpus_size = config['input_size'], config['corpus_size']
    embed_model = YTRSEmbedding(input_size['video'], config['embed_size'],
                                corpus_size, gru=config['fusion_gru'])
    candidate_model = YTRSCandidateGenerate(
        input_size['candidate_cat'], config['in_channel'],
        config['output_size'], config['n_layer'])

    embed_model.to(device)
    candidate_model.to(device)

    download_model_weights(local_path, storage_path)
    load_models(local_path['model'], embed_model, candidate_model, device)

    ######################################################################
    # Inference
    ######################################################################
    message("Inference", paragraph=True)
    embed_model.eval(), candidate_model.eval()
    torch.set_grad_enabled(False)

    recommendation = RecommendationSystem(
        embed_model, candidate_model, samples, inferset, inferloader,
        platforms, datastore_kind, top_k, pubsub_topic, device, message)
    recommendation.inference(inferloader)

    ######################################################################
    # End task
    ######################################################################
    end_time = datetime.now()
    message(f"> [{projectId}] [{task}] - Done! ({end_time-start_time})")
    tracker.done(end_time-start_time)

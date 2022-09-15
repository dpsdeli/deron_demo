from datetime import datetime, timedelta
import numpy as np
from os.path import join as PJ
import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetPrototype(Dataset):
    def __init__(self, data_root, data_file, n_day, t_max, feature_pad,
                 video_feature_dict, in_infer=False, entire_video_corpus=True):
        date_format = "%Y-%m-%d"
        self.n_day = datetime.strptime(n_day, date_format)
        self.t_max = datetime.strptime(t_max, date_format) + timedelta(days=1)
        self.in_infer = in_infer

        path = {'user_': PJ(data_root, 'user_.csv'),
                'videos': PJ(data_root, 'videos.csv'),
                'history': PJ(data_root, 'history.csv'),
                'video_info': PJ(data_root, 'video_info.csv')}

        ######################################################################
        # Data, users and videos
        ######################################################################
        # Load data set (train/val/test)
        with open(PJ(data_root, f"data_list/{data_file}"), 'r') as f:
            user_list = list(map(int, f.read().split('\n')))

        # User information
        user_info = self._load_data(
            path['user_'], user_list, dict_key='_id')
        user_list = list(user_info.keys())

        # Watch history (users who have input and label in train and eval)
        watch_history = self._load_data(
            path['history'], user_list, time_column='hd_at')
        if not in_infer:
            user_list, watch_history = self._users_with_input_and_label(
                user_list, watch_history, self.n_day)

        # User collected videos (old -> new, created_at within n_day)
        df = self._load_data(path['videos'], user_list,
                             time_column='d_at', day=self.n_day)
        users_have_videos = list(set(df['user_id'].tolist()))
        df = df.groupby(['r_id'])
        collect_videos_dict = {
            user: df.get_group(user)['video_id'].tolist()[::-1]
            for user in users_have_videos}

        # Video information (2 months and 10 mins)
        video_info = self._load_and_process_video_info(
            path['video_info'], norm_created_day=60, norm_duration=600,
            dict_key='video_id')

        # Video watched times (during past 30 days)
        default_watch_times = 0
        watch_times_dict = self._video_watch_times(watch_history, days=30)
        for video_id in video_info.keys():
            video_info[video_id]['watch_times'] = \
                watch_times_dict.get(video_id, default_watch_times)

        self.user_list = user_list
        self.user_info = user_info
        self.watch_history = watch_history
        self.collect_videos_dict = collect_videos_dict
        self.video_info = video_info
        self.video_feature_dict = video_feature_dict

        ######################################################################
        # Samples in evaluation and negative sampling
        ######################################################################
        if entire_video_corpus:
            video_corpus = list(self.video_info.keys())
        else:
            video_corpus = [v for v, info in self.video_info.items()
                            if info['ja_subtitle'] or info['zhTW_subtitle']]
        video_corpus.sort()
        self.video_corpus = torch.LongTensor(video_corpus)
        self.corpus_size = len(video_corpus)

        ##############################
        # Padding
        ##############################
        self.pad = {'video': 0, 'keyword': '', 'video_info': [0, 0, 0]}
        self.feature_pad = feature_pad
        self.default_video_info = {
            'watch_times': default_watch_times,
            'zhTW_subtitle': False, 'ja_subtitle': False}

    def __len__(self):
        return len(self.user_list)

    def _load_and_process_video_info(self, path, norm_created_day, norm_duration, dict_key):
        df = pd.read_csv(path)
        # Transform format
        df['at'] = df['t_at'].astype('datetime64[ns]')

        # Normalize
        df['_at'] = [max(t.days, 0)
                            for t in self.t_max - df['_at']]
        condition = df['_at'] >= norm_created_day
        df.loc[condition, '_at'] = norm_created_day
        df['_at'] /= norm_created_day
        df['du'] /= norm_duration

        # In dictionary
        df = df.set_index(dict_key)
        df = df.to_dict('index')
        return df

    def _load_data(self, path, user_list, time_column=None, dict_key=None, day=None):
        df = pd.read_csv(path)
        condition = df['id'].isin(user_list)
        df = df[condition]

        # As datetime format
        if time_column:
            df[time_column] = df[time_column].astype('datetime64[ns]')
        # To dictionary
        if dict_key:
            df = df.set_index(dict_key)
            df = df.to_dict('index')
        # Keep data which time_column within `day`
        if day:
            condition = df[time_column] < day
            df = df[condition]

        return df

    def _video_watch_times(self, watch_history, days=30):
        day = self.n_day - timedelta(days=days)
        condition = ((watch_history['_at'] > day) &
                     (watch_history['_at'] < self.n_day))
        watch_times = watch_history[condition]
        videos, times = np.unique(watch_times['video_id'].tolist(),
                                  return_counts=True)

        # Normalization
        idx = np.argsort(-times)
        videos, times = videos[idx], times[idx]
        times = times / times[0]
        watch_times_dict = dict(zip(videos, times))
        return watch_times_dict

    def _remove_user(self, data_list, df, condition, column_name='column_a'):
        sub_df = df[condition]
        data_list = list(set(
            pd.unique(sub_df[column_name])).intersection(data_list))
        data_list.sort()
        return data_list

    def _users_with_input_and_label(self, user_list, watch_history, n_day):
        #  with input
        condition = watch_history['at'] < n_day
        users_have_input = pd.unique(watch_history[condition]['_id'])
        user_list = list(set(users_have_input).intersection(_list))

        #  with label
        condition = watch_history['_at'] >= n_day
        users_have_label = pd.unique(watch_history[condition]['r_id'])
        user_list = list(set(users_have_label).intersection(user_list))

        # Filter corresponding watch hiostory
        condition = watch_history[r_id'].isin(user_list)
        watch_history = watch_history[condition]
        return user_list, watch_history

    def _extract_info(self, videos, info_names):
        video_info = []
        for video in videos:
            info = self.video_info.get(video, self.default_video_info)
            info = [info[name] for name in info_names]
            video_info.append(info)

        return video_info

    def _extract_features(self, videos, default):
        video_features = [self.video_feature_dict.get(video_id, default)
                          for video_id in videos]
        video_features = torch.stack(video_features, dim=0)
        return video_features

    def _padding(self, data, pad, num, float_type=True):
        data = data[:num]
        num_lack = num - len(data)
        data = data + [pad]*num_lack if num_lack > 0 else data
        if float_type:
            return torch.FloatTensor(data)
        return data

    def _padding_tensor(self, data, tensor_pad, num):
        data = data[:num, :]
        num_data = data.shape[0]
        num_lack = num - num_data
        pad = tensor_pad.repeat((num_lack, 1))
        data = torch.cat((data, pad), dim=0)
        return data

    def _get_input(self, user, watch_history):
        # Watched videos
        watched_videos = watch_history['key'].tolist()[::-1]

        # User collected videos
        collect_videos = self.collect_videos_dict.get(user, [self.pad['key']])

        # User information
        gender = torch.FloatTensor([self.user_info[user]['a']])
        language = torch.LongTensor([self.user_info[user]['b']])

        # Number of days away from last watched (the x days since n_day)
        last_time = (self.n_day - watch_history['_at'].iloc[0]).days
        last_time = torch.FloatTensor([last_time])

        # Example age
        day = 0 if self.in_infer else (self.t_max - self.n_day).days
        example_age = torch.FloatTensor([day])

        return (watched_videos, collect_videos, gender, language, last_time,
                example_age)

    def _preprocessing(self, watched_videos, collect_videos, info_names):
        watched_videos_info = self._extract_info(watched_videos, info_names)
        collect_videos_info = self._extract_info(collect_videos, info_names)

        feature_pad = self.feature_pad['v']
        watched_features = self._extract_features(watched_videos, feature_pad)
        collect_features = self._extract_features(collect_videos, feature_pad)

        # Padding
        pad = self.pad['v_info']
        watched_videos_info = self._padding(watched_videos_info, pad,
                                            self.num_watched_videos)
        collect_videos_info = self._padding(collect_videos_info, pad,
                                            self.num_collect_videos)
        pad = self.feature_pad['v']
        watched_features = self._padding_tensor(watched_features, pad,
                                                self.num_watched_videos)
        collect_features = self._padding_tensor(collect_features, pad,
                                                self.num_collect_videos)
        # Concatenate information as features
        watched_features = torch.cat((watched_features, watched_videos_info), 1)
        collect_features = torch.cat((collect_features, collect_videos_info), 1)

        return watched_features, collect_features


class VTRSDataset(DatasetPrototype):
    """ Recommendation System Dataset """
    def __init__(self, data_root, data_file, n_day, t_max,
                 feature_pad, video_feature_dict,
                 num_watched_videos=000, num_collect_videos=000,
                 num_negative_samples=000, num_positive_samples=000,
                 in_train=True, entire_video_corpus=True):
        super().__init__(data_root, data_file, n_day, t_max, feature_pad,
                         video_feature_dict, False, entire_video_corpus)
        self.data_root = data_root
        self.in_train = in_train

        # The labels in watch history is after n_day (include)
        self.num_watched_videos = num_watched_videos
        self.num_collect_videos = num_collect_videos
        self.num_negative_samples = num_negative_samples
        self.num_positive_samples = num_positive_samples

    def __getitem__(self, index):
        # Get this user's data
        user = self.user_list[index]
        condition = self.watch_history['_id'] == user
        watch_history = self.watch_history[condition]

        #################################################
        # Label
        #################################################
        label_condition = watch_history['d_at'] >= self.n_day
        # Positive samples (list: old -> new)
        labels = watch_history[label_condition]
        positive_samples = labels.video_id.tolist()[::-1]
        num_positive = len(positive_samples)

        positive_samples = self._padding(
            positive_samples, self.pad['video'], self.num_positive_samples,
            float_type=False)
        num_positive = min(num_positive, self.num_positive_samples)

        #################################################
        # Input
        #################################################
        watch_history = watch_history[~label_condition]
        (watched_videos, collect_videos, gender, language,
            last_time, example_age) = self._get_input(user, watch_history)

        #################################################
        # Preprocessing
        #################################################
        info_names = ['_times', 'z_subtitle', 'j_subtitle']
        watched_features, collect_features = self._preprocessing(
            watched_videos, collect_videos, info_names)

        input_list = (watched_features, collect_features,
                      example_age, gender, language, last_time)

        if self.in_train:
            # Concatenate sample information as features
            positive_info = self._extract_info(positive_samples, info_names)
            positive_features = self._extract_features(positive_samples,
                                                       self.feature_pad['video'])
            positive_info = torch.FloatTensor(positive_info)
            positive_features = torch.cat((positive_features, positive_info), 1)

            return (positive_features, torch.LongTensor(positive_samples),
                    input_list, num_positive)

        return torch.LongTensor(positive_samples), input_list, num_positive


class VTRSDatasetInference(DatasetPrototype):
    """Recommendation System Dataset for inference """
    def __init__(self, data_root, data_file, n_day, t_max,
                 feature_pad, video_feature_dict,
                 num_watched_videos=000, num_collect_videos=000,
                 entire_video_corpus=True, week=000):
        super().__init__(data_root, data_file, n_day, t_max, feature_pad,
                         video_feature_dict, True, entire_video_corpus)

        self.data_root = data_root

        self.num_watched_videos = num_watched_videos
        self.num_collect_videos = num_collect_videos

        # Only keep user who watched videos during 3 weeks
        day = self.n_day - timedelta(days=7*week)
        condition = self.watch_history['_at'] > day
        active_users = list(set(self.watch_history[condition]['_id']))
        self.user_list = np.intersect1d(self.user_list, active_users,
                                        assume_unique=True).tolist()
        self.user_list.sort()

    def __getitem__(self, index):
        # Get this user's data
        user = self.user_list[index]
        condition = self.watch_history['_id'] == user
        watch_history = self.watch_history[condition]

        #################################################
        # Input
        #################################################
        (watched_videos, collect_videos, gender, language,
            last_time, example_age) = self._get_input(user, watch_history)

        #################################################
        # Preprocessing
        #################################################
        info_names = ['wtimes', 'z_subtitle', 'j_subtitle']
        watched_features, collect_features = self._preprocessing(
            watched_videos, collect_videos, info_names)

        input_list = (watched_features, collect_features,
                      example_age, gender, language, last_time)

        return user, input_list


if __name__ == '__main__':
    # """
    print("Testing RSCDataset")
    from torch.utils.data import DataLoader
    data_root = "./deep_learn/dataset"
    data_file = "train_list.txt"
    num_watched_videos = 000
    num_positive_samples = 000
    num_negative_samples = 000
    feature_pad = {'video': torch.zeros([512]),
                   'keyword': torch.zeros([512])}
    video_feature_dict = torch.load("./deep_learn/recommendation_system/aaa.pt")

    dataset = VTRSDataset(
        data_root, data_file,
        n_day='2020-10-15', t_max='2020-10-31',
        feature_pad=feature_pad, video_feature_dict=video_feature_dict,
        num_watched_videos=num_watched_videos,
        num_negative_samples=num_negative_samples,
        num_positive_samples=num_positive_samples,
        in_train=True)

    dataloader = DataLoader(dataset, 2, shuffle=False, num_workers=0)
    for it, (positive_features, positive_samples, input_list, num_positive) in enumerate(dataloader, 1):
        print(positive_features.shape)
        print(positive_samples)
        print("watchd videos\n", input_list[0].shape)
        print("collect videos\n", input_list[1].shape)
        print(input_list[2:])
        print(num_positive)
        exit()
    # """


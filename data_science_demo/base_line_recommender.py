# install packages
import numpy as np
import pandas as pd

from lightfm import LightFM
from scipy import sparse
from spotlight.sequence.implicit import ImplicitSequenceModel
from spotlight.cross_validation import user_based_train_test_split
from spotlight.interactions import Interactions
from tqdm import tqdm
from utils import get_data_from_db_v2


# Calculate long / short term trending data
def calculate_trending_data() -> dict:
    """
    Function to calculate long / short term trending data

    Returns:
        trending_dict(dict): result of trending data
    """

    sql_query_string_long_term_data_rating = """sql string"""

    sql_query_string_short_term_data_rating = """sql string"""

    _long_term_rating_df = get_data_from_db_v2(
        query=sql_query_string_long_term_data_rating, database='DB_A')
    _short_term_rating_df = get_data_from_db_v2(
        query=sql_query_string_short_term_data_rating, database='DB_A')

    _trending_df = _long_term_rating_df.merge(
        _short_term_rating_df, how='left', on=['column_a', 'column_b'])
    _trending_df = _trending_df[['column_a', 'column_b', 'column_c']]
    _trending_df = _trending_df.sort_values(
        by=['column_a', 'column_b'], ascending=False)

    trending_dict = {}
    for i in range(1, 30):
        _temp_list = _trending_df.loc[(
            _trending_df.column_a == i)].column_b.tolist()
        trending_dict[i] = _temp_list

    return trending_dict


def data_preprocessing() -> pd.DataFrame:

    sql_query_rating_string = """sql string"""
    sql_query_string_ = """sql string"""

    _rating_df = get_data_from_db_v2(
        query=sql_query_rating_string, database='DB_A')
    c_df = get_data_from_db_v2(query=sql_query_string_, database='DB_A')

    _rating_df = _rating_df.loc[_rating_df['column_f'].isin(
        _rating_df['column_f'].unique().tolist())]

    return _rating_df, c_df


def create_interaction_matrix(df, user_col: str, item_col: str, rating_col: str, norm=False, threshold=None):
    """Function to create an dictionary based on their index and number in interaction dataset
       from interaction matrix dataframe of transactional type interactions.

    Args:
        df: Pandas DataFrame containing user-item interactions.
        user_col: column name containing user's identifier.
        item_col: column name containing item's identifier.
        rating col: column name containing user feedback on interaction with a given item
        norm(optional): True if a normalization of ratings is needed
        threshold(required if norm = True): value above which the rating is favorable
    Returns:
        _interactions_df: Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
        _interactions_dict: Dictionary type output containing interaction_index as key and user_id as value
    """

    _interactions_df = df.groupby([user_col, item_col])[rating_col].sum(
    ).unstack().reset_index().fillna(0).set_index(user_col)
    if norm:
        _interactions_df = _interactions_df.applymap(
            lambda x: 1 if x > threshold else 0)
    user_id = list(_interactions_df.index)
    _interactions_dict = {}
    counter = 0
    for i in user_id:
        _interactions_dict[i] = counter
        counter += 1

    return _interactions_df, _interactions_dict


def create_dict(df, id_col: str, name_col: str) -> dict:
    '''Function to create an item dictionary based on their item_id and item name.

    Args:
        df: Pandas dataframe with Item information
        id_col: Column name containing unique identifier for an item
        name_col: Column name containing name of the item
    Returns:
        hero_course_dict: Dictionary type output containing item_id as key and item_name as value
    '''
    _c_dict = {}
    for i in range(df.shape[0]):
        _c_dict[(df.loc[i, id_col])] = df.loc[i, name_col]

    return _c_dict


def run_mf_algorithm(interactions, n_components=100, loss='warp', k=15, epoch=30, n_jobs=4):
    ''' Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run
        - n_jobs = number of cores used for execution
    Expected Output  -
        Model - Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components=n_components, loss=loss, k=k)
    model.fit(x, epochs=epoch, num_threads=n_jobs)
    return model


def sample_recommendation_user(model, interactions, user_id, user_dict,
                               item_dict, threshold=0, nrec_items=10, show=True):
    '''
    Function to produce user recommendations
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output -
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))

    known_items = list(pd.Series(interactions.loc[user_id, :]
                                 [interactions.loc[user_id, :] > threshold].index)
                       .sort_values(ascending=False))

    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        counter = 1
        for i in scores:
            counter += 1

    return return_score_list


def lightfm_recommender_algorithm() -> dict:
    """
    """

    # lightmf recommendation
    interactions_df, interactions_dict = create_interaction_matrix(df=rating_df,
                                                                   user_col='column_a',
                                                                   item_col='column_c',
                                                                   rating_col='column_d')
    _c_dict = create_dict(interactions_df, 'column_a', 'column_b')

    print(f'factorization machine algorithm training...')
    # Build FM Model
    _mf_model = run_mf_algorithm(interactions=interactions_df,
                                           n_components=1024,
                                           loss='bpr',
                                           epoch=125,
                                           n_jobs=4)

    mf_user_dict = {}
    for value in interactions_df["hero_user_id"].unique():
        user_rec_list = sample_recommendation_user(model=_mf_model,
                                                   interactions=interactions_df,
                                                   user_id=value,
                                                   user_dict=interactions_dict,
                                                   item_dict=_c_dict,
                                                   threshold=3,
                                                   nrec_items=60,
                                                   show=False)
        mf_user_dict[value] = user_rec_list

    return mf_user_dict


def recommend_next_movies(user_ids, model, n_movies=75):

    pred = model.predict(sequences=np.array(user_ids))
    indices = np.argpartition(pred, -n_movies)[-n_movies:]
    best_movie_ids = indices[np.argsort(pred[indices])]
    return [movie_id for movie_id in best_movie_ids]


def sequence_model_training() -> dict:

    train_data_df = train_data_df[[
        "column_a", "column_b", "column_c", "column_d"]]
    train_data_df["c_timestamp"] = pd.to_datetime(
        train_data_df['created_date'])

    training_dataset = Interactions(
        user_ids=train_data_df['column_a'].values,
        item_ids=train_data_df['column_b'].values,
        ratings=train_data_df['column_c'].values,
        timestamps=train_data_df["column_d"].values
    )

    train, test = user_based_train_test_split(training_dataset)
    train = train.to_sequence(max_sequence_length=30, min_sequence_length=1)
    test = test.to_sequence(max_sequence_length=30, min_sequence_length=1)

    sequence_model = ImplicitSequenceModel(embedding_dim=768, n_iter=150,
                                           representation='pooling', learning_rate=0.045,
                                           loss='bpr')
    sequence_model.fit(train, verbose=True)
    sequence_model.fit(training_dataset.to_sequence(), verbose=True)

    #
    recommendation_dict = {}

    for user in recommendation_dict._id.unique():
        recommendation_dict.setdefault(user, [])

    for user in tqdm(recommendation_dict._id.unique()):
        _sequence_list = data_b_df.loc[(data_b_df._id == user)].e_id.tolist()
        recommend_c_list = recommend_next_movies(
            user_ids=_sequence_list, model=sequence_model)
        recommendation_dict[user] = recommend_c_list

    return recommendation_dict


# execute data preprocessing before model training
trending_dict = calculate_trending_data()
data_a_df, data_b_df = data_preprocessing()

# execute sequence models for recommending items
recommendation_a_dict = sequence_model_training()
recommendation_b_dict = lightfm_recommender_algorithm()

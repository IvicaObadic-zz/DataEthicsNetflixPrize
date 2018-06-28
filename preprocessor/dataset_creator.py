import os
import util
import descriptive
import random
import pandas as pd

MIN_RATINGS_THRESHOLD = 2

MIN_MOVIE_RATING_COUNTS = 100
PERCENTAGE_OF_SAMPLES_TO_TAKE = 0.05

def sample_from_file(filename, movie_id):
    # Samples data from a file with filename supplied as input parameter to this function.

    movie_file = open(filename, mode='r')
    file_content = movie_file.readlines()[1:]

    data_to_take = [observation for observation in file_content if random.random() <= PERCENTAGE_OF_SAMPLES_TO_TAKE]
    data_to_take = [preference.strip().split(',') for preference in data_to_take]

    for d in data_to_take:
        d.insert(1, movie_id)

    return [tuple(preference) for preference in data_to_take]


def create_initial_dataset():
    '''
    Creates the initial dataset by processing the original movie files.
    (user, movie, rating, timestamp) tuples are extracted only for movies which have at least 100 ratings
    :return:
    '''
    movie_rating_counts = descriptive.movie_rating_count_descriptives(plot_hist=False)
    print('Movie rating counts are read. Starting to create the dataset...')
    data_as_list = []
    count = 0
    for filename in os.listdir(util.TRAINING_DATA_ROOT_PATH):
        if count % 100 == 0:
            print(str(count) + ' files are processed')
        movie_id = util.extract_movie_id_from_filename(filename)
        if movie_rating_counts[movie_id] < MIN_MOVIE_RATING_COUNTS:
            continue  # We want to have for each movie at least 5 ratings.

        movie_abs_filename = os.path.abspath(os.path.join(util.TRAINING_DATA_ROOT_PATH, filename))
        movie_data = sample_from_file(movie_abs_filename, movie_id)
        for preference in movie_data:
            data_as_list.append(preference)
        count = count + 1
    ## Writing the data in a csv file.
    print('Creating csv file for the dataset')
    print('Num of observations = ' + str(len(data_as_list)))
    dataset = pd.DataFrame(data_as_list, columns=['user_id', 'movie_id', 'rating', 'timestamp'])
    dataset['user_id'] = dataset['user_id'].astype(int)
    dataset['rating'] = dataset['rating'].astype(int)
    dataset.to_csv(util.PREPROCESSED_DATA_PATH + 'dataset.csv')


def clean_sampled_dataset(filename = util.PREPROCESSED_DATA_PATH + 'dataset.csv'):
    '''
    Cleanes the dataset by removing instances of movies which have less than 3 ratings in the sampled data

    :param filename: the dataset file
    :return:
    '''
    data = pd.read_csv(filename, sep=",")
    movie_counts = data['movie_id'].value_counts()
    movie_counts_greater_than_two = movie_counts[movie_counts > MIN_RATINGS_THRESHOLD]
    data = data[data['movie_id'].isin(movie_counts_greater_than_two.index)]
    print('Dataset size after removing not frequent movies: '+str(data.shape))

    user_counts = data['user_id'].value_counts()
    user_counts_greater_than_two = user_counts[user_counts > MIN_RATINGS_THRESHOLD]
    data = data[data['user_id'].isin(user_counts_greater_than_two.index)]
    print('Dataset size after removing not frequent users: ' + str(data.shape))

    data = data[['user_id', 'movie_id', 'rating', 'timestamp']]
    data.to_csv(filename)

if __name__ == '__main__':
    create_initial_dataset()
    clean_sampled_dataset()

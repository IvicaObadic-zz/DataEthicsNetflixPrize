import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util


def movie_rating_count_descriptives(plot_hist=True):
    '''
    Counts the number of ratings from each movie file of the raw movie dataset.

    Usage example:
    movie_rating_counts = movie_rating_count_descriptives(plot_hist=False)
    counts = np.array(list(movie_rating_counts.values()))
    print('Number of movies with rating count greater than 100: ' + str(len(counts[counts >= 100])))

    :param plot_hist: indicates whether histogram of the movie ratings count should be plotted.
    :return: map of number of ratings per movie
    '''

    movie_rating_counts = {}
    for filename in os.listdir(util.TRAINING_DATA_ROOT_PATH):
        movie_abs_filename = os.path.abspath(os.path.join(util.TRAINING_DATA_ROOT_PATH, filename))
        movie_file = open(movie_abs_filename, mode='r')
        num_ratings = len(movie_file.readlines()) - 1

        movie_id = util.extract_movie_id_from_filename(filename)
        movie_rating_counts[movie_id] = num_ratings

        movie_file.close()

    if plot_hist:
        counts = movie_rating_counts.values()
        plt.hist(counts, bins=100)
        plt.xlim([3, 232944])
        plt.xticks(np.arange(0, 232944, step=10000))
        plt.xlabel('Number of ratings per movie')
        plt.ylabel('Frequency')
        plt.show()

    return movie_rating_counts

def movie_count_user_count(filename):
    '''
    Generates descriptive statistics and writes them in a file.

    :param filename: the filename of the dataset
    :return:
    '''
    data = pd.read_csv(filename, sep=',')
    print(data['movie_id'].value_counts(ascending=True))
    print(data['user_id'].value_counts(ascending=True))

if __name__ == '__main__':
    movie_rating_count_descriptives(plot_hist=False)
    movie_count_user_count(util.PREPROCESSED_DATA_PATH + 'dataset.csv')

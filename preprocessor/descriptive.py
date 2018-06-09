import os
import matplotlib.pyplot as plt
import numpy as np
import util
from scipy import stats
from surprise import SVD


def movie_rating_count_descriptives(plot_hist=True):
    # Counts the number of ratings for each movie of the dataset.
    # Plots histogram from the count data.
    # Returns min and max of ratings per movie

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

# Example of usage of the file
# movie_rating_counts = movie_rating_count_descriptives(plot_hist=False)
# counts = np.array(list(movie_rating_counts.values()))
# print('Number of movies with rating count greater than 100: ' + str(len(counts[counts >= 100])))

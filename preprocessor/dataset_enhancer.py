from imdb import IMDb
import util
import pandas as pd

NUM_MOST_RATED_MOVIES_TO_TAKE = 20

def enhance_dataset_with_movie_year_and_titles():
    '''
    Enhances the sample dataset of (user_id, movie_id, rating, timestamp)
    with details about movie titles and year of release for each movie.

    The merged dataset is written as csv file in the ${util.PREPROCESSED_DATA_PATH} folder under the name
    'dataset_with_movie_details.csv'.
    Another file called 'movie_details.csv' is created which contains the movies occurring in the sampled and cleaned dataset
    along with their titles.
    The movie titles are afterwards used in crawling the IMDB in order to get movie genres based on the movie title.
    :return:
    '''
    data = pd.read_csv(util.PREPROCESSED_DATA_PATH + 'dataset.csv', sep=',')
    movies_details = pd.read_csv(
        util.DATA_ROOT_PATH + 'movie_titles.txt',
        sep=',',
        header=None,
        names=['movie_id', 'year_of_release', 'movie_title'], encoding='windows-1252')

    movies_details = movies_details.dropna()
    movies_details[['year_of_release']] = movies_details[['year_of_release']].astype('int32')
    merged_data = pd.merge(data, movies_details, on='movie_id')[['user_id', 'movie_id', 'rating', 'timestamp', 'year_of_release', 'movie_title']]

    print(merged_data.head(5))
    print('Enhanced movie data is created. Writing in file')
    merged_data.to_csv(util.PREPROCESSED_DATA_PATH + 'dataset_with_movie_details.csv', sep=',')

    # Creating the movie_details.csv file which contains details only about the movies occurring in the sampled dataset.
    movies_of_interest = pd.merge(movies_details, merged_data, on='movie_id')[['movie_id', 'movie_title_x']]
    movies_of_interest.columns = ['movie_id', 'movie_title']
    movies_of_interest = movies_of_interest.drop_duplicates(subset = 'movie_id')
    print('"movie_titles.txt" file is subsetted based on movies in the sample dataset. Writing it in file')
    print(movies_of_interest.head(3))
    movies_of_interest.to_csv(util.PREPROCESSED_DATA_PATH + 'movie_details.csv', sep=',')

def crawl_imdb():
    '''
    Crawls the IMDB database to get movie genres based on the title of the movie.
    The movie genres are then written in a separate file called movie_details_enhanced.csv
    When genres about 20 movies are collected, they are flushed in the file in order to don't have to crawl from beggining
    if some unexpected failure happens
    The crawling is done in two steps:
        1) Request is sent to IMDB to get the IMDB movie id based on the movie title
        2) Second request is sent to IMDB to obtain movie details (genre, authors) based on the IMDB movie id.
    :return:
    '''
    data = pd.read_csv(util.PREPROCESSED_DATA_PATH + 'movie_details.csv', sep=',')

    imdb = IMDb()

    already_processed_movies = pd.read_csv(util.PREPROCESSED_DATA_PATH + 'movie_details_enhanced.csv', sep=',')
    already_processed_movies.columns = ['movie_id', 'title', 'genres']
    already_processed_movies = set(already_processed_movies['movie_id'])
    print(already_processed_movies)

    file_to_write = open(util.PREPROCESSED_DATA_PATH + 'movie_details_enhanced.csv', mode='a+', encoding='windows-1252')

    num_movies_processed = 0
    for i in range(0, data.shape[0]):
        movie_id = data.iloc[i, 1]
        title = data.iloc[i, 2]

        if movie_id in already_processed_movies:
            continue

        if num_movies_processed % 20 == 0 and num_movies_processed > 0:
            file_to_write.flush()
            print('Obtained movie genres for ' + str(num_movies_processed) + ' movies from IMDB.')

        imdb_res = imdb.search_movie(title, results=1)
        movie_genres = None
        if len(imdb_res) > 0:
            imdb_movie_basic_info = imdb_res[0]
            movie_details = imdb.get_movie(movieID=imdb_movie_basic_info.getID())
            if 'genres' in movie_details.keys():
                movie_genre = movie_details['genres']
                movie_genres = ';'.join(movie_genre)

        file_to_write.write(str(movie_id) + ',' + str(title) + ',' + str(movie_genres) + '\n')
        num_movies_processed = num_movies_processed + 1

    file_to_write.close()

def merge_dataset_with_movie_genres(include_k_most_rated_movies = True):
    '''
    Merges the files 'dataset_with_movie_details.csv' with file 'movie_details_enhanced.csv'
    in order to combine the crawled movie genres from IMDB and create the final dataset that will be used by
    Factorization Machines algorithm.
    The file 'dataset_enhanced.csv' which contains the final dataset used for further analysis
    consists of the following columns:
        user_id, movie_id, rating, timestamp, year_of_release, Genre 1, ..., Genre N (Genres are binary columns)

    :param include_k_most_rated_movies: whether the final mf dataset should consist only of the K most rated movies.
    :return:
    '''

    # Drop the movies from the sampled dataset for which genres are not found in IMDB (genres = 'None').
    movie_details_enhanced = pd.read_csv(util.PREPROCESSED_DATA_PATH + 'movie_details_enhanced.csv',
                                         sep=',',
                                         encoding='windows-1252',
                                         header=None)
    movie_details_enhanced.columns = ['movie_id', 'movie_title', 'genres']
    movie_details_enhanced = movie_details_enhanced[movie_details_enhanced.genres != 'None']
    print('Number of movies for which genres are found in IMDB: ' + str(movie_details_enhanced.shape[0]))


    dataset_with_movie_details = pd.read_csv(util.PREPROCESSED_DATA_PATH + 'dataset_with_movie_details.csv', sep=',')
    dataset_with_movie_details['timestamp'] = dataset_with_movie_details['timestamp'].str[:4].astype('int32')
    dataset_with_movie_details['years_diff'] = 0.00001 + dataset_with_movie_details['timestamp'] - \
                                               dataset_with_movie_details['year_of_release']
    # Subset by using only the k most rated movies in the dataset.
    if include_k_most_rated_movies:
        k_most_rated_movies = dataset_with_movie_details['movie_id'].value_counts()[0:NUM_MOST_RATED_MOVIES_TO_TAKE]
        movie_details_enhanced = movie_details_enhanced[movie_details_enhanced['movie_id'].isin(k_most_rated_movies.index)]

    #Create binary columns of genres for each movie and merge with the movie details dataframe
    genres_as_binary_matrix = movie_details_enhanced['genres'].str.get_dummies(';')
    # print(genres_as_binary_matrix.sum().divide(genres_as_binary_matrix.sum().sum() * 1.0))
    movie_details_enhanced = pd.concat([movie_details_enhanced, genres_as_binary_matrix], axis=1)

    # Merge the genres in the sampled dataset for each movie
    merged_data = pd.merge(dataset_with_movie_details, movie_details_enhanced, on='movie_id')
    print('Number of instances in the merged dataset for matrix factorization algorithm: ' + str(merged_data.shape[0]))

    # Subset the columns of interest
    genre_column_names = [genre for genre in genres_as_binary_matrix.columns]
    merged_data = merged_data[['user_id', 'movie_id', 'rating', 'years_diff'] + genre_column_names]
    merged_data.to_csv(util.PREPROCESSED_DATA_PATH + 'fm_dataset.csv', sep=',')

if __name__ == '__main__':
    enhance_dataset_with_movie_year_and_titles()
    crawl_imdb()
    merge_dataset_with_movie_genres()

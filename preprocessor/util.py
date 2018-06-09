DATA_ROOT_PATH = '../data/'
TRAINING_DATA_ROOT_PATH = DATA_ROOT_PATH + "training_set"
PREPROCESSED_DATA_PATH = DATA_ROOT_PATH + 'preprocessed/'

def extract_movie_id_from_filename(filename):
    # Extracts the movie id from the name of the file
    if (len(filename) != 14):
        raise ValueError('Filename must have length of 14')
    return int(filename[3:10])

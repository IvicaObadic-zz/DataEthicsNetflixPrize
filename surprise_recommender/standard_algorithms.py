from surprise import Reader, Dataset
from surprise.model_selection import KFold, GridSearchCV
from surprise import SVD, SVDpp, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, BaselineOnly, NormalPredictor, CoClustering, NMF
from surprise import model_selection
from surprise import accuracy
import pandas as pd
import numpy as np

# Read options for 1M and 5M datasets
#reader1M = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)
#data = Dataset.load_from_file('../data/preprocessed/dataset_cleaned.csv', reader=reader1M)

# Read options for 163K dataset
reader163K = Reader(line_format='user item rating', sep=',',skip_lines=1)
data163K = pd.read_csv("../data/preprocessed/fm_dataset.csv")
data163K = data163K[['user_id','movie_id','rating']]
data = Dataset.load_from_df(data163K, reader=reader163K)

def try_recom_algorithm(data, algo, filename, n_splits=5):
    """
    Function that tries out the standard recommendation algorithms supported by Surprise library,
    without hyperparameter tuning
    :param data: input data containing user, item, rating and timestamp(opt)
    :param algo: the recom. algorithm to be used
    :param filename: name of the file the results should be saved into
    :param n_splits: number of folds for the cross validation
    :return:
    """
    print("\nWorking on "+ filename+"\n")
    kf = KFold(n_splits=n_splits)
    i=1

    file = open("../results_surprise_163K/"+filename+".txt","w+")
    avg_rmse = 0
    avg_mae = 0
    for trainset, testset in kf.split(data):
        print("Fold "+str(i))
        file.write("Fold %d\n\n" % (i))
        i+=1

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error and Mean Absolute Error
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)
        file.write("RMSE: %f\n" % (rmse))
        file.write("MAE: %f\n\n" % (mae))

        avg_rmse += rmse
        avg_mae += mae

    # Calculate average RMSE and MAE
    avg_rmse = avg_rmse / n_splits
    avg_mae = avg_mae / n_splits
    file.write("Avg. RMSE: %f\n" % (avg_rmse))
    file.write("Avg. MAE: %f\n" % (avg_mae))
    file.close()


try_recom_algorithm(data, SlopeOne(), "slope_one")
try_recom_algorithm(data, NMF(), "nmf")
try_recom_algorithm(data, NormalPredictor(), "random")
try_recom_algorithm(data, BaselineOnly(), "baseline")
try_recom_algorithm(data, CoClustering(), "co_clustering")
try_recom_algorithm(data, SVD(), "svd")
try_recom_algorithm(data, SVDpp(), "svd_pp")
# NMF biased (use baselines)
try_recom_algorithm(data, NMF(biased = True), 'nmf_biased')

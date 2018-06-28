from sklearn.model_selection import cross_val_score
from surprise import Reader, Dataset
from surprise.model_selection import KFold, GridSearchCV
from surprise import SVD, SVDpp, SlopeOne, KNNBasic, KNNBaseline, KNNWithMeans, BaselineOnly, NormalPredictor, CoClustering, NMF
from surprise import model_selection
from surprise import accuracy
import pandas as pd

# Define the format
reader = Reader(line_format='user item rating timestamp', sep=',',skip_lines=1)
# Load the data from the file using the reader format
data = Dataset.load_from_file('../data/preprocessed/dataset_cleaned.csv', reader=reader)


def try_recom_algorithm_grid(data, algo, filename, grid_options, n_splits=5):
    print("\nWorking on " + filename + "\n")
    file = open("../results_surprise/" + filename + ".txt", "w+")

    gs = GridSearchCV(algo, grid_options, measures=['rmse', 'mae'], cv=n_splits)
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])
    file.write("RMSE: %f" % (gs.best_score['rmse']))

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])
    file.write("Best params:")
    file.write(str(gs.best_params['rmse']))
    file.close()

"""
# svd (defaults: lr_all = 0.005, reg_all = 0.02, n_factors = 100, n_epochs = 20
#svd_grid = { 'lr_all': [0.003, 0.007],'reg_all': [0.01, 0.05],'n_factors':[10,15]}
#try_recom_algorithm_grid(data, SVD, "svd_grid", svd_grid)

# svd ++ (defaults: lr_all = 0.007, reg_all = 0.02, n_factors = 20, n_epochs = 20
svd_pp_grid = { 'lr_all': [0.005, 0.009],'reg_all': [0.01, 0.05],'n_factors':[10,15]}
try_recom_algorithm_grid(data, SVDpp, "svd_pp_grid", svd_pp_grid)

# baseline only (als)
baseline_als = {'bsl_options': {'reg_i': [5, 15], 'reg_u':[10,25],'n_epochs':[10,20]}}
try_recom_algorithm_grid(data, BaselineOnly, "baseline_grid_als",baseline_als)

# baseline only (sdg)
baseline_sgd = {'bsl_options': {'method':['sgd'],'learning_rate': [.0005, 0.05],'reg':[0.01, 0.05],'n_epochs':[20,25]}}
try_recom_algorithm_grid(data, BaselineOnly, "baseline_grid_sgd",baseline_sgd)
"""

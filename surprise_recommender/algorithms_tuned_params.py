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
data = Dataset.load_from_file('../data/preprocessed/data_surprise.csv', reader=reader)

#bsl_options = {'method': 'als',
 #              'reg_u': 12,
 #              'reg_i':5
 #              }
#try_recom_algorithm(data, BaselineOnly(bsl_options=bsl_options), "baseline_opt_2", n_splits=5)

#bsl_options = {'method': 'als',
 #              'reg_u': 15,
  #             'reg_i':10
   #            }
#try_recom_algorithm(data, BaselineOnly(bsl_options=bsl_options), "baseline_opt_3", n_splits=5)

#bsl_options = {'method': 'als',
 #              'reg_u': 20,
  #             'reg_i':5
   #            }
#try_recom_algorithm(data, BaselineOnly(bsl_options=bsl_options), "baseline_opt_4", n_splits=5)


def try_recom_algorithm_grid(data, algo, filename, grid_options, n_splits=5):
    print("\nWorking on " + filename + "\n")
    file = open("../results/" + filename + ".txt", "w+")

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

# svd
svd_grid = { 'lr_all': [0.002, 0.007],'reg_all': [0.3, 0.6],'n_factors':[15,20]}
try_recom_algorithm_grid(data, SVD, "svd_grid", svd_grid)

"""
# baseline only (als)
baseline_als = {'bsl_options': {'reg_i': [5, 15], 'reg_u':[10,25],'n_epochs':[10,20]}}
try_recom_algorithm_grid(data, BaselineOnly, "baseline_grid_als",baseline_als)

# baseline only (sdg)
baseline_sgd = {'bsl_options': {'learning_rate': [.0005, 0.05],'reg':[0.01, 0.05],'n_epochs':[20,25]}}
try_recom_algorithm_grid(data, BaselineOnly, "baseline_grid_sgd",baseline_sgd)

"""
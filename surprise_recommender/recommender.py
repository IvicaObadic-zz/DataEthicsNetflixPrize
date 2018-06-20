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


# Use the  SVD algorithm.
#algo = SVD()
# Run 5-fold cross-validation and print results.
#model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Use SVD ++
#algo = SVDpp()
#model_selection.cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def try_recom_algorithm(data, algo, filename, n_splits=5):
    print("Working on "+ filename+"\n")
    kf = KFold(n_splits=n_splits)
    i=1

    file = open("../results/"+filename+".txt","w+")
    avg_rmse = 0
    avg_mae = 0
    for trainset, testset in kf.split(data):
        print("Fold "+str(i))
        file.write("Fold %d\n\n" % (i))
        i+=1

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        rmse = accuracy.rmse(predictions, verbose=True)
        mae = accuracy.mae(predictions, verbose=True)
        file.write("RMSE: %f\n" % (rmse))
        file.write("MAE: %f\n\n" % (mae))

        avg_rmse += rmse
        avg_mae += mae

    avg_rmse = avg_rmse / n_splits
    avg_mae = avg_mae / n_splits
    file.write("Avg. RMSE: %f\n" % (avg_rmse))
    file.write("Avg. MAE: %f\n" % (avg_mae))
    file.close()


#try_recom_algorithm(data, SlopeOne(), "slope_one", n_splits=5)
#try_recom_algorithm(data, KNNBaseline(), "knn_baseline", n_splits=5)
#try_recom_algorithm(data, KNNBasic(), "knn_basic", n_splits=5)
#try_recom_algorithm(data, KNNWithMeans(), "knn_means", n_splits=5)
#try_recom_algorithm(data, NMF(), "nmf", n_splits=5)
#try_recom_algorithm(data, NormalPredictor(), "random", n_splits=5)
#try_recom_algorithm(data, BaselineOnly(), "baseline", n_splits=5)
#try_recom_algorithm(data, CoClustering(), "co_clustering", n_splits=5)
#try_recom_algorithm(data, SVD(), "svd", n_splits=5)
#try_recom_algorithm(data, SVDpp(), "svd_pp", n_splits=3)

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


# Grid search for SVD
"""
file = open("../results/svd_grid.txt","w+")
# Grid search
param_grid = {'lr_all': [0.003, 0.006], 'n_epochs':[15,25],
              'reg_all': [0.3, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])
file.write("RMSE: %f" % (gs.best_score['rmse']))


# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
file.write("Parameters: %f" % (gs.best_params['rmse']))
file.close()

"""
"""
# Grid search for baseline

file = open("../results/baselne_grid.txt","w+")
# Grid search
param_grid = {'bsl_options': {'reg_i': [5, 15], 'reg_u':[10,25]}}
gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])
file.write("RMSE: %f" % (gs.best_score['rmse']))


# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
file.close()

"""

# Grid search for baseline

file = open("../results/baselne_grid_sgd.txt","w+")
# Grid search
param_grid = {'bsl_options': {'method':['als','sgd'],'learning_rate': [.0005,0.05]}}
gs = GridSearchCV(BaselineOnly, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])
file.write("RMSE: %f" % (gs.best_score['rmse']))


# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])
file.close()
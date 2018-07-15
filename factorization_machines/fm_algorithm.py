from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import numpy as np

data = pd.read_csv('../data/preprocessed/' + 'fm_dataset.csv', sep=',')
ratings = np.array(data['rating'], dtype=float)
columns_to_remove = ['Unnamed: 0', 'timestamp', 'rating']
data = data[data.columns.difference(columns_to_remove)]

data['movie_id'] = data['movie_id'].astype(str)
data['user_id'] = data['user_id'].astype(str)
print(data.head(3))

print('Encoding the data in proper format for the factorization machines algorithm')
dict_vectorizer = DictVectorizer()
X = dict_vectorizer.fit_transform(data.T.to_dict().values())
print('Encoding finished, starting the cross validation for factorization machines algorithm')

kf = KFold(n_splits=5)
splits = kf.split(X)
file_to_write = open('fm_results.txt', mode='w')

fm_num_iterations = [10, 25, 50]
num_factors = [10, 25, 20]

for fm_iteration in fm_num_iterations:
    for factor in num_factors:
        file_to_write.write('Results for configuration: Iterations = {}, num_factors = {}'
                            .format(fm_iteration, factor))
        file_to_write.write('\n')
        fm = pylibfm.FM(num_factors=factor, num_iter = fm_iteration, verbose=True, task="regression", initial_learning_rate=0.001, learning_rate_schedule="optimal")

        fold = 1
        mean_rmse = 0.0
        for train_index, test_index in splits:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = ratings[train_index], ratings[test_index]

            fm.fit(X_train, y_train)
            predictions = fm.predict(X_test)

            mse = mean_squared_error(predictions, y_test)
            rmse = math.sqrt(mse)
            print('RMSE for fold {} is {}'.format(fold, rmse))
            file_to_write.write('RMSE for fold {} is {}'.format(fold, rmse))
            file_to_write.write('\n')
            file_to_write.flush()
            fold = fold + 1
            mean_rmse = mean_rmse + rmse


        print('Mean RMSE on Cross validation = {}'.format(mean_rmse / kf.get_n_splits()))
        file_to_write.write('Mean RMSE on Cross validation = {}'.format(mean_rmse / kf.get_n_splits()))
        file_to_write.write('\n')
        file_to_write.write('###########################\n')
        file_to_write.flush()

file_to_write.close()
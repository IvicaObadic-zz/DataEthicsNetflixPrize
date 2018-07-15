# DataEthicsNetflixPrize

Team: Ivica Obadic & Angela Josifovska

Recommendation system as part of univeristy project on the Netflix dataset implemented in Python 3. 

Different recommendation systems algorithms are explored and compared with the novel Factorization Machines method. 

Libraries used for implementation of the algorithms:
* Surprise: http://surpriselib.com/
* pyfm: https://github.com/coreylynch/pyFM
* IMDbpy: https://github.com/alberanid/imdbpy

------------------------------

## Project structure
* **data** - folder containing the preprocessed datasets used for the analysis. As the datasets were too large for upload on GitHub, they have to be manually copied in this folder, before running the scripts. If one wants to generate the preprocessed datasets, the original movie files of the netflix dataset should be copied to the folder training_set under this data directory.
* **preprocessor**
    * ```dataset_creator.py```
    * ```dataset_enchanser.py```
    * ```descriptive.py```
* **factorization_machines**:
    * ```fm_algorithm.py``` - 
    * ```fm_results.txt``` - 
* **surprise_recommender**:
    * ```standard_algorithms.py``` - This script reads a dataset and tries out all the algorithms from Surprise library with default configurations and saves the results in separate text files. Algorithms used: Normal Predictor (random), Slope One, Co-clustering, Baseline Only, SVD, SVD++, NMF.
    * ```algorithms_tuned_params.py ``` - This script uses the algorithms that had the best performance in the default setting and tries to find the best configuration by tuning the parameters with Grid search. The algorithms used are: Baseline Only, SVD and SVDpp. For each of them, we define a dictionary of grid options - possible value range for each of the parameters. The results (RMSE & best param.configurations) are saved again in separate .txt files.
* **results_surprise_5M**: Folder containing the results of recommendation algorithms implemented in the Surprise library, implemented on a data subset containing 5 million ratings. The results are saved in separate .txt files for each algorithm and show the average RMSE and MSE of 5-Fold cross-validation for both default and tuned version of each algorithm. For the tuned version, the best parameter configuration is also given.
* **results_surprise_1M**: Similarly, this folder contaings the results of all algorithms from the Surprise library, but for a different subset of the original data, containing around 1 million ratings. [ADD SUBSETING 1M]
* **results_surprise_163K**: Results for a subset of around 163 000 ratings [ADD SUBSETTING]

## How to run the project
 
1. Put the CSV files containing the datasets into the **data** folder.
2. To run the algorithms from Surprise library with default configuration on the 163 K dataset, just run the ```standard_algorithms.py``` script. The results from each fold are printed in the console and the final results are saved in .txt files. To try the algorithms on the other two datasets, the name of the dataset sould be changed.
3. To tune the best algorithms (SVD, SVD++, Baseline Only) and test them on the 163K dataset, just run the  ```algorithms_tuned_params.py ``` script. Again, the results are both printed in console and saved in .txt files. To try the algorithms on the other two datasets, the name of the dataset sould be changed.
4. To try the Factorization machines algorithm - 

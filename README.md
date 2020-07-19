# Aprendizado de Máquina - 1.2020

This repository is dedicated to the development of the practical work of Machine Learning.

## Original dataset

The dataset used for this work is available at: https://crawdad.org/dartmouth/campus/20090909

David Kotz, Tristan Henderson, Ilya Abyzov, Jihwang Yeo, CRAWDAD dataset dartmouth/campus (v. 2009‑09‑09), https://doi.org/10.15783/C7F59T, Sep 2009.

The used portion of the dataset was the Syslog data of October of 2001, available at the directory syslog-v3.3, files from 20011001.log.gz to 20011031.log.gz

## Generated Dataset

The generated dataset is available at the [dataset-2001-10.csv](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/dataset-2001-10.csv) file.

![dataset_plot](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/plots/final/dataset_plot.png)

## Scripts

- [build_dataset.py](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/build_dataset.py)

This script processes the data from the Dartmouth Syslog dataset of October 2001. Generating the dataset-2001-10.csv file.

- [plot_dataset.py](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/plot_dataset.py)

This script plots graphs related to the generated dataset.

- [batch_learn.py](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/batch_learn.py)

This scripts uses batch regressors using the dataset file as input to predict the number of users in APs, generating a output file with the evaluated values.

**Models supported:** K-Nearest-Neighbors, Decision Tree, Random Forest.

**Validation method:** Time Split Validation

**Metrics:** Mean Average Error and Mean Squared Error

**Usage:** ```python3 batch_learn.py [-h] -m MODEL_OPTION [-a] [-f FILEPATH] [-p] [-s SPLIT_NUM] [-rc] [-br]```

```
Required Arguments:
  -m MODEL_OPTION, --model MODEL_OPTION
                        Batch Learning Regression Model. [KNN, DT, RF or DUMMY]
Optional Arguments:
  -h, --help            Show help message and exit
  -a, --alltargets      Run model with all APs, instead of one AP.
  -f FILEPATH, --datasetfile FILEPATH
                        Pass dataset file path
  -p, --plot            Display plot. Only available with one label.
  -s SPLIT_NUM, --splits SPLIT_NUM
                        Number of splits to make in the Time Split Validation. Default = 4 Splits
  -rc, --regressorchain
                        Use regressor chain for multioutput.
  -br, --multioutput    Use multioutput learner for multioutput.

```

- [unified_stream_learning.py](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/unified_stream_learning.py)

This scripts uses stream learning regressors using the dataset file as input to predict the number of users in APs, generating a output file with the evaluated values.

**Models supported:** K-Nearest Neighbors, Hoeffding Adaptive Tree, Adaptive Random Forest.

**Validation method:** Evaluate Prequential, Holdout

**Metrics:** Mean Average Error and Mean Squared Error

**Usage:** ```python3 unified_stream_learning.py [-h] -m MODEL_OPTION [-a] [-f FILEPATH] [-s SAMPLES] [-p] [-rc] [-ht]```

```
Required Arguments:
  -m MODEL_OPTION, --model MODEL_OPTION
                        Stream Learning Regression Model. [KNN, HAT or ARF]
Optional Arguments:
  -h, --help            Show help message and exit
  -a, --alltargets      Run model with all APs, instead of one AP.
  -f FILEPATH, --datasetfile FILEPATH
                        Pass dataset file path
  -s SAMPLES, --samples SAMPLES
                        Samples to process. Default: all samples
  -p, --plot            Display plot. Only available with one label.
  -rc, --regressorchain
                        Use regressor chain for multioutput. Default: Multi Output Learner
  -ht, --holdout        Use holdout instead of prequential

```
- [plot_result.py](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/results/plot_result.py)

This scripts plots the results.

**Usage**: ```python3 plot_result.py [-h] [-rc] [-s SAMPLES]```

```
Optional arguments:
  -h, --help            show this help message and exit
  -rc, --regressorchain
                        Plot regressor chain evaluation. Default = Multi Output Learner.
  -s SAMPLES, --samples SAMPLES
                        Number of samples to plot.
```

## Plot Examples

- **Dummy Regressor with Time Split Validation**

![Dummy Time Split Validation](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/plots/time-split-dummy.png)

- **KNN Regressor with Time Split Validation**

![KNN Time Split Validation](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/plots/time-split-knn.png)

- **Result of KNN, ARF and HAT using Multi Output Learner**

![KNN, ARF and HAT result](https://github.com/juanlucasvieira/Aprendizado-Maquina.1.2020/blob/master/plots/final/result_br.png)

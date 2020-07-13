# import pandas as pd
from skmultiflow.data import FileStream
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.tree import DecisionTreeRegressor

# from sklearn.model_selection import cross_validate, cross_val_predict
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import numpy as np
# from numpy import absolute, mean, std

from skmultiflow.trees import HoeffdingAdaptiveTreeRegressor
from skmultiflow.lazy import KNNRegressor
from skmultiflow.meta import AdaptiveRandomForestRegressor

from skmultiflow.evaluation import EvaluatePrequential

from skmultiflow.meta.multi_output_learner import MultiOutputLearner

import argparse

parser = argparse.ArgumentParser(description='Stream Learning Regression Script')
requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument('-m', '--model', action='store', dest='model_option',
                    help='Stream Learning Regression Model. [KNN, HAT or ARF]', required=True)
parser.add_argument('-a', '--alltargets', action='store_true', default=False, dest='all_ap',
                    help='Run model with all labels (APs)')
parser.add_argument('-f', '--datasetfile', action='store', default="dataset-2001-10.csv", dest='filepath',
                    help='Pass dataset file path')
parser.add_argument('-s', '--samples', action='store', default=44641, type=int, dest='samples',
                    help='Samples to process. Default: all samples')
parser.add_argument('-p', '--plot', action='store_true', default=False, dest='show_plot',
                    help='Display plot. Only available with one label.')

args = parser.parse_args()

###########
# DATASET #
###########
if args.all_ap:
        stream = FileStream(args.filepath, target_idx=4, n_targets=387) # OCTOBER , # ALL APS FROM COLUMN 4 to 387
        #stream = FileStream("dataset-2001-10.csv", target_idx=4, n_targets=387) # OCTOBER , # ALL APS FROM COLUMN 4 to 387
else:
        stream = FileStream(args.filepath, target_idx=52) # OCTOBER , # AP -> AcadBldg18AP2
        #stream = FileStream("dataset-2001-10.csv"

#max_samples = 44641
max_samples = args.samples

model = None

model_name = args.model_option.upper()

if model_name == "KNN":
        model = KNNRegressor(n_neighbors=5, max_window_size=1000)
        print("Chosen regressor:", "K-Nearest Neighbors")
elif model_name == "HAT":
        model = HoeffdingAdaptiveTreeRegressor()
        print("Chosen regressor:", "Hoeffding Adaptive Tree")
elif model_name == "ARF":
        model = AdaptiveRandomForestRegressor(random_state=123456)
        print("Chosen regressor:", "Adaptive Random Forest")
else: 
        print("Invalid Model Specified. Expected: KNN, HAT or ARF")
        parser.print_usage()
        exit()


if not args.all_ap:
        #evaluator = EvaluatePrequential(output_file=model_name+"_eval_one_label.txt",show_plot=args.show_plot, pretrain_size=200, max_samples=max_samples, metrics=['true_vs_predicted','mean_square_error','mean_absolute_error'])
        evaluator = EvaluatePrequential(output_file=model_name+"_eval_one_label_v2.txt",show_plot=args.show_plot, n_wait=60, pretrain_size=60, batch_size=60, max_samples=max_samples, metrics=['true_vs_predicted','mean_square_error','mean_absolute_error','running_time','model_size'])
        evaluator.evaluate(stream=stream, model=model, model_names=[model_name])
else:
        # For Multi-AP approach
        multiOutputModel = MultiOutputLearner(base_estimator=model)

        evaluator = EvaluatePrequential(output_file=model_name+"_eval_all_labels_v2.txt", n_wait=60, pretrain_size=60, batch_size=60, max_samples=max_samples, metrics=['average_mean_square_error','average_mean_absolute_error','running_time','model_size'])
        #evaluator = EvaluatePrequential(output_file=model_name+"_eval_all_labels.txt", pretrain_size=200, max_samples=max_samples, metrics=['average_mean_square_error','average_mean_absolute_error'])
        evaluator.evaluate(stream=stream, model=multiOutputModel, model_names=['MultiOutput'+model_name])
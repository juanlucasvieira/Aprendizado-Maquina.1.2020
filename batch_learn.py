import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy import absolute, mean, std

def plot_real_vs_predicted2(total_size,total_measured,index_test, measured_test, index_predicted, predicted_values):
    plt.rcParams['figure.figsize'] = 12, 3
    plt.xlim(0,total_size)
    plt.plot(np.arange(index_predicted[0],index_predicted[-1]+1),total_measured.iloc[index_predicted[0]:index_predicted[-1]+1],label='Measured (Test)')
    plt.plot(index_test,measured_test,label='Measured (Train)')
    plt.plot(index_predicted,predicted_values,label='Predicted')
    plt.legend()
    plt.show()

def plot_timesplit(plt, total_size,total_measured,index_test, measured_test, index_predicted, predicted_values):
    plt.rcParams['figure.figsize'] = 12, 3
    plt.xlim(0,total_size)
    ax = plt.subplot(411)
    plt.plot(np.arange(index_predicted[0],index_predicted[-1]+1),total_measured.iloc[index_predicted[0]:index_predicted[-1]+1],label='Measured (Test)')
    plt.plot(index_test,measured_test,label='Measured (Train)')
    plt.plot(index_predicted,predicted_values,label='Predicted')
    plt.legend()
    plt.show()

def plot_parcial(plt, i, n_splits, axList, total_size,total_measured,index_test, measured_test, index_predicted, predicted_values):
    plt.rcParams['figure.figsize'] = 12, n_splits
    plt.xlim(0,total_size)
    if len(axList) > 0:
        axList.append(plt.subplot(n_splits, 1, i+1, sharex=axList[i-1],sharey=axList[i-1]))
    else:
        axList.append(plt.subplot(n_splits, 1, i+1))
    if i != n_splits-1:
        plt.setp(axList[i].get_xticklabels(), visible=False)
    plt.plot(index_test,measured_test,label='Measured (Train)')
    plt.plot(np.arange(index_predicted[0],index_predicted[-1]+1),total_measured.iloc[index_predicted[0]:index_predicted[-1]+1],label='Measured (Test)')
    plt.plot(index_predicted,predicted_values,label='Predicted')
    if i == 0:
        plt.legend(ncol=2)
    

# def plot_real_vs_predicted(index_measured, measured_values, index_predicted, predicted_values):
#     plt.rcParams['figure.figsize'] = 12, 5
#     plt.plot(index_measured,measured_values,label='Measured')
#     plt.plot(index_predicted,predicted_values,label='Predicted')
#     plt.legend()
#     plt.show()

def printRealVsEvaluted(measured, predicted):
    print("Real","Predicted")
    real = measured.tolist()
    for i in range(1,len(predicted)):
        print(real[i],predicted[i])

def evaluateMSEandMAE(measured, predicted):
    mse = mean_squared_error(measured, predicted)
    mae = mean_absolute_error(measured, predicted)
    print("Evaluation - ","MSE:",mse,"| MAE:", mae)
    return mse, mae

def holdOutEvaluation(model, X, y):
    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33) 
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    evaluateMSEandMAE(y_test,predicted)

def kFoldCV(model, X, y):
    # K Fold Cross Validation
    result = cross_validate(model, X, y, cv=10, scoring='neg_mean_squared_error')
    print("10 FCV - MSE ",mean(absolute(result['test_score'])))
    result = cross_validate(model, X, y, cv=10, scoring='neg_mean_absolute_error')
    print("10 FCV - MAE ",mean(absolute(result['test_score'])))

def timeSeriesSplitCV(model, X, y, splits, plotGraph=False):
    # Time Series Split Cross Validation
    tscv = TimeSeriesSplit(n_splits=splits)
    mse_values = []
    mae_values = []
    i=0
    axList = []
    for train_index, test_index in tscv.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # print(y_train)
        model.fit(X_train,y_train)
        predicted = model.predict(X_test)
        mse, mae = evaluateMSEandMAE(y_test,predicted)
        mse_values.append(mse)
        mae_values.append(mae)
        #print(train_index)
        #print(y_test)

        if plotGraph:
                plot_parcial(plt,i,splits,axList,len(y),y,train_index,y_train,test_index,predicted)
                #plot_real_vs_predicted2(len(y),y,train_index,y_train,test_index,predicted)
                #plot_real_vs_predicted(np.arange(len(y)),y,test_index,predicted)
                i=i+1
    
    print("Time Series Evaluation:")
    print("MSE VALUES",mse_values,"\nMSE MEAN", mean(mse_values))
    print("MAE VALUES",mae_values,"\nMAE MEAN", mean(mae_values))

    if plotGraph:    
        plt.show()
        # for i in range(0,len(predicted[0])):
        #     plot_real_vs_predicted(np.arange(len(y.iloc[:,i])),y.iloc[:,i],test_index,predicted[:,i])


parser = argparse.ArgumentParser(description='Batch Learning Regression Script')
requiredArgs = parser.add_argument_group('required arguments')
requiredArgs.add_argument('-m', '--model', action='store', dest='model_option',
                    help='Stream Learning Regression Model. [KNN, DT, RF or DUMMY]', required=True)
parser.add_argument('-a', '--alltargets', action='store_true', default=False, dest='all_ap',
                    help='Run model with all labels (APs)')
parser.add_argument('-f', '--datasetfile', action='store', default="dataset-2001-10.csv", dest='filepath',
                    help='Pass dataset file path')
parser.add_argument('-p', '--plot', action='store_true', default=False, dest='show_plot',
                    help='Display plot. Only available with one label.')
parser.add_argument('-s', '--splits', action='store', default=4, type=int, dest='split_num',
                    help='Number of splits to make in the Time Split Cross Validation.')

args = parser.parse_args()

##########
# MODELS #
##########

model = None
model_name = args.model_option.upper()

if args.show_plot and args.all_ap:
        print("Plotting is not available with multi-label regression. Please remove -p or -a option.")
        exit()

if model_name == "KNN":
        model = KNeighborsRegressor(n_neighbors=5)
        print("Chosen regressor:", "K-Nearest Neighbors")
elif model_name == "DT":
        model = DecisionTreeRegressor(random_state=0)
        print("Chosen regressor:", "Decision Tree")
elif model_name == "RF":
        model = RandomForestRegressor(max_depth=10, random_state=0) # Default: 100 trees
        print("Chosen regressor:", "Random Forest")
elif model_name == "DUMMY":
        model = DummyRegressor(strategy="mean")
else: 
        print("Invalid Model Specified. Expected: KNN, DT or RF")
        parser.print_usage()
        exit()

###########
# DATASET #
###########
dataframe = pd.read_csv(args.filepath, header=0) 
#dataframe = pd.read_csv("dataset-2001-10.txt", header=0) # OCTOBER 
#dataframe = pd.read_csv("dataset-2001-oct-nov.txt", header=0) # OCTOBER AND NOVEMBER


feature_cols = ['day', 'hour', 'minute', 'weekday'] # FEATURES
#feature_cols = ['hour','minute','weekday']

X = dataframe.loc[:,feature_cols]
if not args.all_ap:
        print("One label")
        y = dataframe.AcadBldg18AP2 # Only one AP first as target
else:
        print("Multi-label")
        y = dataframe.loc[:,"AcadBldg10AP1":"SocBldg9AP1"] # ALL APS

#print(dataframe.head(-1))

timeSeriesSplitCV(model,X,y,args.split_num,args.show_plot)

#real = y.tolist()

# Scatter plot
# fig, ax = plt.subplots()
# ax.scatter(real, predicted)
# ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import numpy as np
from numpy import absolute, mean, std
import pandas as pd
import datetime
import argparse
import matplotlib.ticker as ticker
from sklearn.preprocessing import minmax_scale

def plot_results(index, knn_values, hat_values, arf_values, titletxt, samples):
    plt.rcParams['figure.figsize'] = 10, 6
    #ax.set_xticklabels('Monday', 'Tuesday')
    
    plt.xlim(0,len(knn.id))
    ax1 = plt.subplot(311)
    ax1.set_xmargin(0.01)

    ax1.xaxis.set_tick_params(color='white',labelrotation=90, which="minor", labelsize=0, labelcolor='white')

    # ax1.set_ylim(0,3)

    ax1.text(1.01,0.4, "K-Nearest\nNeighbors", size=10, ha="left", transform=ax1.transAxes)
    ax1.set_title(titletxt)
    plt.plot(index, knn["current_amse_[MultiOutputKNN]"].iloc[:len(index)], label="KNN AMSE (Current)")
    plt.plot(index, knn["current_amae_[MultiOutputKNN]"].iloc[:len(index)], label="KNN AMAE (Current)")
    # plt.plot(index, knn["mean_amse_[MultiOutputKNN]"], label="KNN AMSE (Mean)")

    ax2 = plt.subplot(312, sharex=ax1, sharey=ax1)

    ax2.xaxis.set_tick_params(color='white',labelrotation=90, which="minor", labelsize=0, labelcolor='white')
    ax2.text(1.01,0.35, "Hoeffding\nAdaptive\nTree", size=10, ha="left", transform=ax2.transAxes)

    plt.plot(index, hat["current_amse_[MultiOutputHAT]"].iloc[:len(index)], label="HAT AMSE (Current)")
    plt.plot(index, hat["current_amae_[MultiOutputHAT]"].iloc[:len(index)], label="HAT AMAE (Current)")
    # plt.plot(index, hat["mean_amse_[MultiOutputHAT]"], label="HAT AMSE (Mean)")
    ax3 = plt.subplot(313, sharex=ax1, sharey=ax2)
    
    ax3.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU)))
    ax3.xaxis.set_minor_locator(dates.HourLocator(byhour=12))
    ax3.xaxis.set_major_formatter(ticker.NullFormatter())
    ax3.xaxis.set_minor_formatter(dates.DateFormatter('%a %d'))
    ax3.xaxis.set_tick_params(color='white',labelrotation=90, which="minor")
    
    ax3.set_xmargin(.0)
    # ax3.set_ylabel("Adaptive Random Forest")
    ax3.text(1.01,0.35, "Adaptive\nRandom\nForest", size=10, ha="left", transform=ax3.transAxes)
    # ax3.text(.5, .5, "Adaptive Random Forest")

    plt.plot(index, arf["current_amse_[MultiOutputARF]"].iloc[:len(index)], label="Average Mean Squared Error (Current)")
    plt.plot(index, arf["current_amae_[MultiOutputARF]"].iloc[:len(index)], label="Average Mean Absolute Error (Current)")
    # plt.plot(index, arf["mean_amse_[MultiOutputARF]"], label="ARF AMSE (Mean)")
    print(knn.loc[knn['id'] == samples,["mean_amse_[MultiOutputKNN]"]].to_string(index=False,header=False))
    print(titletxt, "AMSE -","KNN:",knn.loc[knn['id'] == samples,["mean_amse_[MultiOutputKNN]"]].to_string(index=False,header=False),"| HAT:",hat.loc[hat['id'] == samples,["mean_amse_[MultiOutputHAT]"]].to_string(index=False,header=False),"| ARF:",arf.loc[arf['id'] == samples,["mean_amse_[MultiOutputARF]"]].to_string(index=False,header=False))
    print(titletxt, "AMAE -","KNN:",knn.loc[knn['id'] == samples,["mean_amae_[MultiOutputKNN]"]].to_string(index=False,header=False),"| HAT:",hat.loc[hat['id'] == samples,["mean_amae_[MultiOutputHAT]"]].to_string(index=False,header=False),"| ARF:",arf.loc[arf['id'] == samples,["mean_amae_[MultiOutputARF]"]].to_string(index=False,header=False))

    plt.legend(ncol=2)
    plt.show()
    # plt.setp(ax.get_xmajorticklabels(), visible=False)
    # plt.savefig("plot_result_"+titletxt+".pdf",bbox_inches='tight')
    plt.clf()

parser = argparse.ArgumentParser(description='Result Plot Script')
parser.add_argument('-rc', '--regressorchain', action='store_true', default=False, dest='plot_rc',
                    help='Plot regressor chain evaluation. Default = Binary Relevance.')
parser.add_argument('-s', '--samples', action='store', default=-1, type=int, dest='samples',
                    help='Number of splits to plot.')

args = parser.parse_args()


plot_title = ""
if args.plot_rc:

    knn = pd.read_csv("KNN_eval_all_labels_v2_rc.txt", header=5)
    hat = pd.read_csv("HAT_eval_all_labels_v2_rc.txt", header=5)
    arf = pd.read_csv("ARF_eval_all_labels_v2_rc.txt", header=5)
else:
    knn = pd.read_csv("KNN_eval_all_labels_v2.txt", header=5)
    hat = pd.read_csv("HAT_eval_all_labels_v2.txt", header=5)
    arf = pd.read_csv("ARF_eval_all_labels_v2.txt", header=5)

min_samples = min(knn["id"].iloc[-1],hat["id"].iloc[-1],arf["id"].iloc[-1])

model_min_samples = None
if min_samples == knn["id"].iloc[-1]:
    model_min_samples = knn
elif min_samples == hat["id"].iloc[-1]:
    model_min_samples = hat
elif min_samples == arf["id"].iloc[-1]:
    model_min_samples = arf

base_timestamp = int(datetime.datetime(2001,10,1,1,0).strftime('%s')) # 1/Oct/2001

index = []
for id in model_min_samples["id"]:
    if args.samples > 0 and id > args.samples:
        break
    index.append(datetime.datetime.fromtimestamp(base_timestamp + (60 * id)))
    min_samples = id

# print(index)

if args.plot_rc:
    plot_title = "Regressor Chain Evaluation (" + str(min_samples) + " samples)"
else:
    plot_title = "Multi Output Evaluation (" + str(min_samples) + " samples)"

plot_results(index,knn,hat,arf,plot_title, min_samples)

exit()

# ax = plt.subplot(311)
# plt.plot(knn.id, knn["current_amae_[MultiOutputKNN]"], label="KNN AMAE (Current)")
# plt.plot(knn.id, knn["mean_amae_[MultiOutputKNN]"], label="KNN AMAE (Mean)")
# #ax = plt.subplot(312)
# plt.plot(hat.id, hat["current_amae_[MultiOutputHAT]"], label="HAT AMAE (Current)")
# plt.plot(hat.id, hat["mean_amae_[MultiOutputHAT]"], label="HAT AMAE (Mean)")
# #ax = plt.subplot(313)
# plt.plot(arf.id, arf["current_amae_[MultiOutputARF]"], label="ARF AMAE (Current)")
# plt.plot(arf.id, arf["mean_amae_[MultiOutputARF]"], label="ARF AMAE (Mean)")

#print("AMAE -","KNN",knn["mean_amae_[MultiOutputKNN]"].iloc[-1],"HAT",hat["mean_amae_[MultiOutputHAT]"].iloc[-1],"ARF",arf["mean_amae_[MultiOutputARF]"].iloc[-1])

# plt.show()



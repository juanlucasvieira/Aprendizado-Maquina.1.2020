import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU
import numpy as np
from numpy import absolute, mean, std
import pandas as pd
import datetime
import matplotlib.ticker as ticker
from sklearn.preprocessing import minmax_scale

year = 2001
month = 10

def get_normalized(values):
    normalized_list = minmax_scale(values)
    # norm = np.linalg.norm(values)
    # normalized_values = values/norm
    return normalized_list

def get_day_average(values):
    mean_days_together = []
    for i in range(0,1440,1):
        aux = []
        # print(mean_list)
        for j in range(0,len(values),1440):
            print(dateList[j+i],values[j+i])
            aux.append(values[j+i])
        mean_value = mean(aux)
        # print("aux",aux)
        # print("mean_value",mean_value)
        mean_days_together.append(mean_value)
    print(len(mean_days_together))
    return mean_days_together

    


def plot_dataset(index, real_values, ylabel):
    plt.rcParams['figure.figsize'] = 10, 4
    fig, ax = plt.subplots()
    #ax.set_xticklabels('Monday', 'Tuesday')
    ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU)))
    ax.xaxis.set_minor_locator(dates.HourLocator(byhour=12))
    # ax.xaxis.set_major_formatter(dates.DateFormatter('%a %d'))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_formatter(dates.DateFormatter('%a %d'))
    
    # for tick in ax.xaxis.get_minor_ticks():
    #     tick.tick1line.set_markersize(0)
    #     tick.tick2line.set_markersize(0)
    #     tick.label1.set_horizontalalignment('center')
    ax.xaxis.set_tick_params(color='white',labelrotation=90, which="minor")
    ax.set_xmargin(.0)

    ax.set_ylabel(ylabel)
    #print(ax.get_yticks())

    ax.plot(index,real_values,label='Measured',color='dimgrey')

    #plt.show()
    # plt.setp(ax.get_xmajorticklabels(), visible=False)
    plt.savefig("plot_dataset"+ylabel+".pdf",bbox_inches='tight')
    plt.clf()

def plot_day(index, real_values, ylabel):
    #plt.rcParams['figure.figsize'] = 6, 4
    plt.rcParams['figure.figsize'] = 5, 4
    fig, ax = plt.subplots()
    
    #ax.set_xticklabels('Monday', 'Tuesday')
    #ax.xaxis.set_major_locator(dates.WeekdayLocator(byweekday=(MO, TU, WE, TH, FR, SA, SU)))
    ax.xaxis.set_major_locator(dates.HourLocator(interval=2))
    ax.xaxis.set_minor_locator(dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(dates.DateFormatter('%Hh'))
    ax.set_ylabel(ylabel)
    ax.set_xmargin(.02)
    ax.plot(index,real_values,label='Measured',linewidth=1,color='dimgrey')

    # plt.show()
    plt.savefig("plot_day"+ylabel+".pdf",bbox_inches='tight')
    plt.clf()

dataframe = pd.read_csv("dataset-2001-10.csv", header=0) # OCTOBER 

feature_cols = ['day', 'hour', 'minute', 'weekday'] # FEATURES

X = dataframe.loc[:,feature_cols]
y = dataframe.AcadBldg18AP2 # Lets try only one AP first...

#print(y)

dateList = []

samples = 44641 # Whole data
#samples = 1440 # 1 Day
#samples = 10080 # 1 week
# samples = 20160 # 2 week

for index in range(0,samples-1):
    currentDate = X.iloc[index]
    day, hour, minute = currentDate.day, currentDate.hour, currentDate.minute
    date = datetime.datetime(year,month,day,hour,minute)
    #print(currentDate)
    dateList.append(date)

#print(dateList)

average_day_one_ap = get_day_average(y)

plot_day(dateList[:1440], get_normalized(average_day_one_ap), "Normalized Average Users (AcadBldg18AP2)")

#plot_day(dateList[:1440], y.loc[:1439], "Number of Users in one AP (AcadBldg18AP2)")
plot_dataset(dateList, y.loc[:samples-2], "Number of Users (AcadBldg18AP2)")

yAll = dataframe.loc[:,"AcadBldg10AP1":"SocBldg9AP1"]
# print(yAll)

mean_list = []
print(np.shape(yAll)[1])
for i in range(np.shape(yAll)[0]):
    mean_list.append(mean(yAll.iloc[i,:]))
# norm = np.linalg.norm(mean_list)
# #normalized_mean_list = mean_list/norm

plot_dataset(dateList, get_normalized(mean_list),"Normalized Average Users (All APs)")
#plot_dataset(dateList, mean_list)

average_day_all_apps = get_day_average(mean_list)

plot_day(dateList[:1440],get_normalized(average_day_all_apps),"Normalized Average Users (All APs)")
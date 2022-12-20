""" 597SD Project  
    Joe Fitzgerald, Robbe De Lange, Burak Yanbul, Zach Grimes
    5G RTT Times for Various Conditions
    """


#libraries/imports
from lzma import FILTER_DELTA
import numpy as np
import matplotlib.pyplot as plt
import json 
from statistics import NormalDist
import math
import statistics
import random

#open JSON file into variable 
with open(r'C:\Users\Joe Fitzgerald\Desktop\597SD\test2.json') as json_file:
    data = json.load(json_file)
    #print(data[0]['_source']['layers']['tcp']['tcp.analysis']['tcp.analysis.ack_rtt'])

#print(len(data)) 

#Loop Thorugh Packets and Extract RTT time and append to vals
vals = []
for i in range(len(data)):
    vals.append(data[i]['_source']['layers']['tcp']['tcp.analysis']['tcp.analysis.ack_rtt'])  #gets RTT time from json file

#print(vals)

#Arrays of Unfiltered Data From File
nparray = np.array(vals) #np array of strings of RTT times
nparray_vals = nparray.astype(np.float) #conversion of RTT strings to floats
print(len(nparray_vals))

#Values from Unfiltered Data
mean = nparray_vals.mean() #mean of RTT times
sd = statistics.stdev(nparray_vals) #standard deviation of RTT

#Filter Out Outliers
maxx = (mean*1.75)
minx = (0)
filtData = []
for k in range(len(nparray_vals)):
    if (minx <= nparray_vals[k] <= maxx):
        filtData.append(nparray_vals[k])

#arrays of filteredData
filtDataa = np.array(filtData)
filteredData = filtDataa.astype(np.float)
print(len(filteredData))


#Values from Filtered Data
filteredMean = filteredData.mean()
filteredSD = statistics.stdev(filteredData)


#formatting Labels in Charts
textx = (maxx*0.6)
mean_forcharts = round(filteredMean, 6)
sd_forcharts = round(filteredSD, 6)
sdstring = str(sd_forcharts)
meanstr = str(mean_forcharts)
#print(maxx)


#X Value for PDF and Number of Bins for PDF
"""
trainx = np.linspace(minx, maxx, len(filteredData))
testx = np.linspace(m)
"""
bin = 200 #number of bins in histogram

"""
#Root Mean Square Error
for k in range(len(filteredData)):
    MSE = (np.square(np.subtract(filteredData[k], filteredMean))).mean()
    RMSE = math.sqrt(MSE)
RMSE_str = str(round(RMSE, 6))
"""

#Density Function
def normal_dist(x, y, z):
    prob_density = (np.pi*z) * np.exp(-0.5*((x-y)/z)**2)
    return prob_density


"""
#PDF Chart
plt.plot(x, pdf, color = 'red')
plt.xlabel('RTT Time (s)')
plt.ylabel('Probability Density')
plt.title("Probability Density Function of RTT (File 1)")
plt.text(textx, 0.00036, ("Standard Deviation = " + sdstring), fontsize = 10)
plt.text(textx, 0.00034, ("Mean = " + meanstr), fontsize = 10)
plt.text(0.0005, 0.00032, ("RMSE = " + RMSE_str), fontsize = 10)
plt.show()
"""
"""
#Histogram
plt.hist(nparray_vals, bin, range=[minx, maxx]) #plots histogram
plt.title("Histogram of RTT Time (File 1)")
plt.xlabel("RTT Time (s)")
plt.ylabel("Frequency")
plt.text(textx, 28, ("Standard Deviation = " + sdstring), fontsize = 10)
plt.text(textx, 26, ("Mean = " + meanstr), fontsize = 10)
plt.text(textx, 24, ("RMSE = " + RMSE_str), fontsize = 10)
plt.show()
"""

#percentiles
bottom10 = np.percentile(filteredData, 10) #10 percentile
bottom25 = np.percentile(filteredData, 25) #25 percentile
bottom50 = np.percentile(filteredData, 50) #50 percentile
bottom75 = np.percentile(filteredData, 75) #75 percentile
bottom90 = np.percentile(filteredData, 90) #90 percentile

print(round(bottom10, 6))
print(round(bottom25, 6))
print(round(bottom50, 6))
print(round(bottom75, 6))
print(round(bottom90, 6))


#Cross Validation
rndmSet = []
for t in range(len(filteredData)):
    rndmSet.append(filteredData[t])
rndmSetS = np.array(rndmSet)
rndmSetD = rndmSetS.astype(np.float)
#rndmSetFinal = random.sample(rndmSetD, len(rndmSetD))
rndmSetFinal = sorted(rndmSetD, key=lambda k: random.random())
#print(rndmSetD)
#print(rndmSetFinal)

#Randomizing Dataset and Splitting into Training and Validation Sets

print(len(rndmSetD))
print(len(rndmSetFinal))
testSet = []
trainSet = []
testSetSize = round((len(rndmSetFinal) * 0.2), 0)   #training set = to 4/5 of data set, test set = 1/5 data set
trainingSetSize = (len(rndmSetFinal) - testSetSize)

for k in range(0, int(testSetSize)):
    testSet.append(rndmSetFinal[k])
for f in range(int(testSetSize) + 1, int(trainingSetSize)):
    trainSet.append(rndmSetFinal[f])

teset = np.array(testSet)
trset = np.array(trainSet)
test_Set = teset.astype(np.float) #test Set
train_Set = trset.astype(np.float) #Training Set

#Test Set Values

testMean = test_Set.mean()          #Mean
testSD = statistics.stdev(test_Set) #Standard Deviation
tmr = round(testMean, 6)
tmsd = round(testSD, 6)
charttestmean = str(tmr)
charttestsd = str(tmsd)
testmin = 0
testmax = (testMean * 1.75)
testx = np.linspace(testmin, testmax, 200)
pdftest = normal_dist(testx, testMean, testSD)

#Training Set Values
trainMean = train_Set.mean()
trainSD = statistics.stdev(train_Set)
tsm = round(trainMean, 6)
tssd = round(trainSD, 6)
charttrainmean = str(tsm)
charttrainsd = str(tssd)
trainmin = 0
trainmax = (trainMean * 1.75)
trainx = np.linspace(trainmin, trainmax, 200)
pdftrain = normal_dist(trainx, trainMean, trainSD)


#RMSE
for k in range(len(test_Set)):
    testMSE = (np.square(np.subtract(test_Set[k], train_Set[k]))).mean()
    testRMSE = math.sqrt(testMSE)
print(testRMSE)
RMSE_str = str(round(testRMSE, 6))


#PDF of Train set
plt.plot(trainx, pdftrain, color = 'red')
plt.xlabel('RTT Time (s)')
plt.ylabel('Probability Density')
plt.title("Probability Density Function of RTT Training Set (File 2)")
plt.text(textx, 0.00034, ("Standard Deviation = " + sdstring), fontsize = 10)
plt.text(textx, 0.00032, ("Mean = " + meanstr), fontsize = 10)
plt.text(0.0005, 0.00030, ("RMSE = " + RMSE_str), fontsize = 10)
plt.show()

#Histogram Training Set
plt.hist(train_Set, bin, range=[min(train_Set), max(trainSet)]) #plots histogram
plt.title("Histogram of RTT Time Training Set (File 2)")
plt.xlabel("RTT Time (s)")
plt.ylabel("Frequency")
plt.text(textx, 195, ("Standard Deviation = " + charttrainsd), fontsize = 10)
plt.text(textx, 185, ("Mean = " + charttrainmean), fontsize = 10)
plt.show()

#PDF of Test set
plt.plot(testx, pdftest, color = 'red')
plt.xlabel('RTT Time (s)')
plt.ylabel('Probability Density')
plt.title("Probability Density Function of RTT Test Set (File 2)")
plt.text(textx, 0.00036, ("Standard Deviation = " + charttestsd), fontsize = 10)
plt.text(textx, 0.00034, ("Mean = " + charttestmean), fontsize = 10)
plt.text(0.0005, 0.00032, ("RMSE = " + RMSE_str), fontsize = 10)
plt.show()

#Histogram Test Set
plt.hist(test_Set, bin, range=[min(test_Set), max(test_Set)]) #plots histogram
plt.title("Histogram of RTT Time Test Set (File 2)")
plt.xlabel("RTT Time (s)")
plt.ylabel("Frequency")
plt.text(textx, 70, ("Standard Deviation = " + charttestsd), fontsize = 10)
plt.text(textx, 65, ("Mean = " + charttestmean), fontsize = 10)
plt.show()
#print(testSetSize)
#print(trainingSetSize)

#print(test_Set)
#print(train_Set)
#print(rndmSet)

#print(nparray_vals)

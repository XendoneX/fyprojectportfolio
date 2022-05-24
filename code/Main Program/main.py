import threading

import numpy as np
import joblib
import csv
import time
import sensorVariable
import pandas as pd
from tkinter import *
from tkinter import ttk
from datetime import datetime

#For machine learning
import matplotlib.pyplot as plt
import math
from temp import classifier
import sympy as sy
from scipy.stats import norm
from sklearn.model_selection import train_test_split as tts

def monitorSystem(handle): #Collects and stores hardware sensor values in a csv file
    csvFile = open('sensorResult.csv', 'w', newline='') #Opens a new writable csv file
    writer = csv.writer(csvFile)  #New Writer file
    cpuload=[] #compiles cpuload values in an array
    cputemp=[]
    cpupower=[]
    hddtemp=[]
    i=0 #Counter
    for i in range(100): #collects 100 values from each hardware sensor
        cpuLoad = sensorVariable.fetchCpuLoad(handle)  #Gets values from sensorVariable
        cpuTemp = sensorVariable.fetchCpuTemp(handle)
        cpuPower= sensorVariable.fetchCpuPower(handle)
        hddTemp = sensorVariable.fetchHddTemp(handle)
        cpuload.append(cpuLoad) #Appends the value in the array
        cputemp.append(cpuTemp)
        cpupower.append(cpuPower)
        hddtemp.append(cpuTemp)
        writer.writerow(np.array([cpuLoad, cpuTemp, cpuPower, hddTemp], dtype=float)) #writes the value in csv
        i+=1 #counter + 1
    csvFile.close() #Closes CSV file
    return cpuload, cputemp, cpupower, hddtemp #Retuns all arrays

def analyzeActivity(model): #Runs the machine learning model
    names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'hddTemp'] #Column names for all variables in machineLearning dataset
    data = pd.DataFrame(pd.read_csv('sensorResult.csv', names=names)) #Load machine learning dataset
    cpuload = data['cpuLoad'] #seperate cpuload values from other variables
    cputemp = data['cpuTemp']
    cpupower = data['cpuPower']
    hddtemp = data['hddTemp']

    cpuloadPrediction = model.CpuLoad(cpuload) #Values are runned through the machine learning model
    cputempPrediction = model.CpuTemp(cputemp)
    cpupowerPrediction = model.CpuPower(cpupower)
    hddtempPrediction = model.HddTemp(hddtemp)

    return cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction #Return an array of predictions

def checkActivity(cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction): #Counts the amount of malicious and non malicious values
    cpuload=sum(cpuloadPrediction) #sum prediction results
    cputemp=sum(cputempPrediction)
    cpupower=sum(cpupowerPrediction)
    hddtemp=sum(hddtempPrediction)
    status=[cpuload,cputemp,cpupower,hddtemp]  #compile all prediction values into one array
    print(cpuload, cputemp, cpupower, hddtemp)
    i=0 #count amount of hardware sensor returning as malicious
    for xi in status: #for each value in status array
        if xi <=65: #if less than 65 of the 100 values than it is malicious, where values 0 is malicious and 1 being non-malicious
            i+=1 #add 1 to the count

    if i>=3: #if more than 3 hardware sensors return malicious then malign activity is detected
        print('returned 0')
        return 0
    else:
        print('returned 1')
        return 1

def startScan(loadmodel): #Initiates program
    while True:
        print('Task Begin')
        now = datetime.now() #gets the current data and time
        csvFile= open('predictionResults.csv', 'a', newline='') #opens a csv file as append
        writer= csv.writer(csvFile) #new csv writer
        handle = sensorVariable.openhardwaremonitor() #opens a new handle from sensorVariable
        cpuload, cputemp, cpupower, hddtemp = monitorSystem(handle) #executes monitorSystem, returning cpuload, cputemp, cpupower, hddtemp values
        cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction = analyzeActivity(loadmodel) #executes analyzeActivity, returning cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction values
        result = checkActivity(cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction) #gets the final verdict from checkActivity

        for xi in range(len(cpuloadPrediction)): #Writes results in CSV
            writer.writerow([now.strftime("%d/%m/%Y %H:%M:%S"), cpuload[xi], cpuloadPrediction[xi], cputemp[xi], cputempPrediction[xi], cpupower[xi], cpupowerPrediction[xi], hddtemp[xi], hddtempPrediction[xi], result])

        if result==0: #if malign than make indicator red otherwise green
            status.itemconfig(square, fill='red')
            print('Task Complete')
        else:
            status.itemconfig(square, fill='green')
            print('Task Complete')

def sleepTime(): #waits for all processes to end
    time.sleep(60)

loadmodel = joblib.load('finalmodel.sav') #loads machine learning model
gui = Tk() #opens new TK gui
gui.title('Malware Detector v2.0') #name of gui
gui.geometry("400x600") #gui aspect ratio

ttk.Label(gui, text='Status:')
status = Canvas(gui, width=400, height=240) #draws a shape arround the rectagle indicator
status.pack()
square = status.create_rectangle(100, 100, 300, 300, fill='green') #creates the rectangle indicator with green as its default indicator

scanButton = ttk.Button(master=gui, text='Start Scan', command=threading.Thread(target=lambda: startScan(loadmodel)).start()) #button that starts the program
scanButton.pack(side='bottom', expand=True)

gui.mainloop()

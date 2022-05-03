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
import tkinter as tk
#For machine learning
import matplotlib.pyplot as plt
import math
from temp import classifier
import sympy as sy
from scipy.stats import norm
from sklearn.model_selection import train_test_split as tts

def monitorSystem(handle):
    csvFile = open('sensorResult.csv', 'w', newline='')
    writer = csv.writer(csvFile)
    i=0
    for i in range(100):
        cpuLoad = sensorVariable.fetch_cpuLoad(handle)
        cpuTemp = sensorVariable.fetch_cpuTemp(handle)
        cpuPower= sensorVariable.fetch_cpuPower(handle)
        hddTemp = sensorVariable.fetch_hddTemp(handle)
        writer.writerow(np.array([cpuLoad, cpuTemp, cpuPower, hddTemp], dtype=float))
        i+=1
    csvFile.close()

def analyzeActivity(model):
    names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'hddTemp']
    data = pd.DataFrame(pd.read_csv('sensorResult.csv', names=names))
    cpuload = data['cpuLoad']
    cputemp = data['cpuTemp']
    cpupower = data['cpuPower']
    hddtemp = data['hddTemp']
    #MLE
    print('Mle')
    cpuloadMu, cpuloadSigma = model.MLE(cpuload)
    cputempMu, cputempSigma = model.MLE(cputemp)
    cpupowerMu, cpupowerSigma = model.MLE(cpupower)
    hddtempMu, hddtempSigma = model.MLE(hddtemp)
    print(cpuloadMu, cpuloadSigma, cputempMu, cputempSigma, cpupowerMu, cpupowerSigma, hddtempMu, hddtempSigma)
    #PDF
    print('pdf')

    cpuloadPDF=model.PDF(cpuloadMu, cpuloadSigma, 3.0, 11.0, cpuload)
    cputempPDF =model.PDF(cputempMu, cputempSigma, 40, 55, cputemp)
    cpupowerPDF =model.PDF(cpupowerMu, cpupowerSigma, 1.9, 14.9, cpupower)
    hddtempPDF =model.PDF(hddtempMu, hddtempSigma, 41, 55, hddtemp)
    print('cpuload: ')
    for xi in cpuloadPDF:
        print(xi)
    print('cputemp: ')
    for xi in cputempPDF:
        print(xi)
    print('cpupower: ')
    for xi in cpupowerPDF:
        print(xi)
    print('hddtemp: ')
    for xi in hddtempPDF:
        print(xi)
    #predicting
    print('prediction')
    cpuloadPrediction = model.Predict(cpuloadMu, cpuloadSigma, cpuloadPDF, cpuload)
    cputempPrediction = model.Predict(cputempMu, cputempSigma, cputempPDF, cputemp)
    cpupowerPrediction = model.Predict(cpupowerMu, cpupowerSigma, cpupowerPDF, cpupower)
    hddtempPrediction = model.Predict(hddtempMu, hddtempSigma, hddtempPDF, hddtemp)

    return cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction

def checkActivity(cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction):
    cpuload=sum(cpuloadPrediction)
    cputemp=sum(cputempPrediction)
    cpupower=sum(cpupowerPrediction)
    hddtemp=sum(hddtempPrediction)
    status=[cpuload,cputemp,cpupower,hddtemp]
    print(cpuload, cputemp, cpupower, hddtemp)
    i=0
   # if cpuload<=45 and cputemp<=45 and cpupower<=45 and hddtemp<=45:
    for xi in status:
        if xi <=65:
            i+=1

    if i>=3:
        print('returned 0')
        return 0
    else:
        print('returned 1')
        return 1

def startScan(loadmodel):
    while True:
        now = datetime.now()
        csvFile= open('predictionResults.csv', 'a', newline='')
        writer= csv.writer(csvFile)
        handle = sensorVariable.openhardwaremonitor()
        monitorSystem(handle)
        print('Task Begin')
        cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction = analyzeActivity(loadmodel)

        result = checkActivity(cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction)
        for xi in range(len(cpuloadPrediction)):
            writer.writerow([now.strftime("%d/%m/%Y %H:%M:%S"), cpuloadPrediction[xi], cputempPrediction[xi], cpupowerPrediction[xi], hddtempPrediction[xi]])

        if result==0:
            status.itemconfig(square, fill='red')
            print('Task Complete')
        else:
            status.itemconfig(square, fill='green')
            print('Task Complete')

def sleepTime():
    time.sleep(60)


loadmodel = joblib.load('finalmodel.sav')
gui = Tk()
gui.title('Malware Detector v1.3')
gui.geometry("400x600")

ttk.Label(gui, text='Status:')
status = Canvas(gui, width=400, height=240)
status.pack()
square = status.create_rectangle(100, 100, 300, 300, fill='green')

pb = ttk.Progressbar(gui, orient=HORIZONTAL, length=200, mode='determinate')
pb.pack(expand=True)

scanButton = ttk.Button(master=gui, text='Start Scan', command=threading.Thread(target=lambda: startScan(loadmodel)).start())
scanButton.pack(side='bottom', expand=True)

gui.mainloop()

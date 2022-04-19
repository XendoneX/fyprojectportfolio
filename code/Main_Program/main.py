import numpy as np
import joblib
import csv
import time
import PySimpleGUI as sg
import datetime
import sensorVariable
import pandas as pd
from tkinter import *
from tkinter import ttk
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
    with open('cpuLoadPDF.csv', 'w', newline='') as cpuloadpdf:  # FILE NAME
        model.PDF(cpuloadMu, cpuloadSigma, 3.0, 11.0, cpuload, cpuloadpdf)
    with open('cpuTempPDF.csv', 'w', newline='') as cputemppdf:  # FILE NAME
        model.PDF(cputempMu, cputempSigma, 42, 55, cputemp, cputemppdf)
    with open('cpuPowerPDF.csv', 'w', newline='') as cpupowpdf:  # FILE NAME
        model.PDF(cpupowerMu, cpupowerSigma, 2.1, 11.1, cpupower, cpupowpdf)
    with open('hddTempPDF.csv', 'w', newline='') as hddtemppdf:  # FILE NAME
        model.PDF(hddtempMu, hddtempSigma, 42, 55, hddtemp, hddtemppdf)
    print('reading')
    cpuloadPDF = pd.DataFrame(pd.read_csv('cpuLoadPDF.csv')).to_numpy()
    cputempPDF = pd.DataFrame(pd.read_csv('cpuTempPDF.csv')).to_numpy()
    cpupowerPDF = pd.DataFrame(pd.read_csv('cpuPowerPDF.csv')).to_numpy()
    hddtempPDF = pd.DataFrame(pd.read_csv('hddTempPDF.csv')).to_numpy()
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
    print(cpuload, cputemp, cpupower, hddtemp)
    if cpuload<=45 and cputemp<=45 and cpupower<=45 and hddtemp<=45:
        return 0
    else:
        return 1


def startScan(loadmodel):
    handle = sensorVariable.openhardwaremonitor()
    monitorSystem(handle)
    print('Task Begin')
    cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction = analyzeActivity(loadmodel)
    result = checkActivity(cpuloadPrediction, cputempPrediction, cpupowerPrediction, hddtempPrediction)
    if result==0:
        status.itemconfig(square, fill='red')
        print('Task Complete')
    else:
        status.itemconfig(square, fill='green')
        print('Task Complete')


loadmodel = joblib.load('finalmodel.sav')
gui = Tk()
gui.title('Malware Detector v1.1')
gui.geometry("400x600")


ttk.Label(gui, text='Status:')
status = Canvas(gui, width=400, height=240)
status.pack()
square = status.create_rectangle(100, 100, 300, 300, fill='green')

pb = ttk.Progressbar(gui, orient=HORIZONTAL, length=100, mode='determinate')
pb.pack(expand=True)

scanButton = ttk.Button(master=gui, text='Start Scan', command=lambda: startScan(loadmodel))
scanButton.pack(side='bottom', expand=True)
gui.mainloop()

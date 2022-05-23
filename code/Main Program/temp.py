import numpy as np
import pandas as pd
import csv
import sympy as sy
import time
from sklearn.model_selection import train_test_split as tts
import joblib
import math
from decimal import Decimal


class classifier:

    def CpuLoad(data):
        names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'ramLoad', 'ramData', 'hddTemp', 'cpuTargets', 'hddTargets']
        dataset = pd.DataFrame(pd.read_csv('combined_data.csv', names=names))
        cpuload = dataset['cpuLoad']
        loadMu, loadSig = classifier.MLE(cpuload)
        loadPdf = classifier.PDF(loadMu, loadSig, 3.0, 11.0, cpuload)
        loadPA, loadPM = classifier.Threshhold(loadPdf)
        array = classifier.Predict(loadMu, loadSig, loadPdf, data, loadPA, loadPM)
        return array

    def CpuTemp(data):
        names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'ramLoad', 'ramData', 'hddTemp', 'cpuTargets', 'hddTargets']
        dataset = pd.DataFrame(pd.read_csv('combined_data.csv', names=names))
        cputemp = dataset['cpuTemp']
        tempMu, tempSig = classifier.MLE(cputemp)
        tempPdf = classifier.PDF(tempMu, tempSig, 40, 55, cputemp)
        tempPA, tempPM = classifier.Threshhold(tempPdf)
        array = classifier.Predict(tempMu, tempSig, tempPdf, data, tempPA, tempPM)
        return array

    def CpuPower(data):
        names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'ramLoad', 'ramData', 'hddTemp', 'cpuTargets', 'hddTargets']
        dataset = pd.DataFrame(pd.read_csv('combined_data.csv', names=names))
        cpupower = dataset['cpuPower']
        powerMu, powerSig = classifier.MLE(cpupower)
        powerPdf = classifier.PDF(powerMu, powerSig, 1.9, 14.9, cpupower)
        powerPA, powerPM = classifier.Threshhold(powerPdf)
        array = classifier.Predict(powerMu, powerSig, powerPdf, data, powerPA, powerPM)
        return array

    def HddTemp(data):
        names = ['cpuLoad', 'cpuTemp', 'cpuPower', 'ramLoad', 'ramData', 'hddTemp', 'cpuTargets', 'hddTargets']
        dataset = pd.DataFrame(pd.read_csv('combined_data.csv', names=names))
        hddtemp = dataset['hddTemp']
        hddMu, hddSig = classifier.MLE(hddtemp)
        hddPdf = classifier.PDF(hddMu, hddSig, 41, 55, hddtemp)
        hddPA, hddPM = classifier.Threshhold(hddPdf)
        array = classifier.Predict(hddMu, hddSig, hddPdf, data, hddPA, hddPM)
        return array

    def MLE(data):
        N = len(data)  # number of values in data
        # Mu Hat
        μHat = float((1 / N) * sum(
            data))  # 1/N (number of variables in the given dataset) X sum of every variable in the dataset

        # Sigma Hat
        finalFile = open('mleSigmaMu.csv', 'w', newline='')  # creates new writable csv file
        writer2 = csv.writer(finalFile)
        for xi in data:  # For each value in the given dataset #Calculate first half of equation
            p1 = (xi - μHat) ** 2  # Dataset value xi - μHat
            writer2.writerow(np.array([p1]))  # Write answer in csv
        finalFile.close()  # close csv file
        p1csv = pd.DataFrame(pd.read_csv('mleSigmaMu.csv'))  # Read csv file
        σHat = sy.sqrt(1 / N * np.sum(p1csv))  # square root(1/N (Number of variables in dataset)) x sum of equation 1
        return μHat, σHat  # return the 2 values

    def PDF(mu, sigma, minIntegral, maxIntegral,
            data):  # ths function will calculate the probability density function using the provided variables (mu= mean of dataset, lowest integral variable, highest integral value, csvfile to write results in)
        pdffile = open('hddPdf.csv', 'w', newline='')  # creates new csv to store PDF results
        pdf = csv.writer(pdffile)
        x = sy.Symbol('x')
        for xi in data:  # For each value in data produce PDF Value
            gx = 1 / (sigma * sy.sqrt(2 * np.pi)) * sy.exp((-(xi - mu) ** 2) / (2 * sigma ** 2))  # PDF Algorithm
            pdf.writerow(np.array([gx], dtype=float))  # Writes results in a csv file
        pdffile.close()

        val = minIntegral  # lowest number of the definite integral
        array = []
        while val != maxIntegral:  # function will run until the provided maximum integral in reached
            row = sy.integrate((1 / (sigma * sy.sqrt(2 * sy.pi))) * sy.exp((-(x - mu) ** 2) / (2 * sigma ** 2)), (
            x, val,
            val + 1))  # Sympy.integrate function uses the PDF algorithm located in the g(x) function above, proceeding to calculate the definite integral of the minumum integral and minumum integral+1 until it reaches the provided maximum integral.
            if row.has(x):
                array.append(float(0.0))  # writes the results in a row of the provided csv file
            else:
                if math.isnan(float(row)):  # Filter out NaN value to 0
                    array.append(float(0.0))
                else:
                    array.append(float(row))
            val += 1  # increments the min integral
        return array

    def Threshhold(pdf):
        pA = sum(pdf)
        pM = float(1)
        filteredPdf = [i for i in pdf if i != 0.0]  # filters out any value null values
        for xi in filteredPdf:
            pM = pM * xi

        return pA, pM

    def Predict(mu, sigma, pdf, data, pA, pM):  # Predicts the whether the provided data/variable is malicious or not
        csvfile = open('prediction.csv', 'w', newline='')
        writer = csv.writer(csvfile)
        print(data.name, ':', pA, pM)
        array = []
        for xi in data:
            p = 1 / (sigma * sy.sqrt(2 * np.pi)) * sy.exp((-(xi - mu) ** 2) / (2 * sigma ** 2))  # PDF Algorithm
            if math.isnan(
                    float(p) or pA == 0.0):  # Checks if values are NaN, if this is true, check values without PDF
                if data.name == 'cpuLoad':
                    if xi >= 3.0 and xi <= 11.0:
                        t = 0
                        array.append(t)
                    else:
                        t = 1
                        array.append(t)
                elif data.name == 'cpuTemp':
                    if xi >= 42 and xi <= 55:
                        t = 0
                        array.append(t)
                    else:
                        t = 1
                        array.append(t)
                elif data.name == 'cpuPower':
                    if xi >= 2.1 and xi <= 11.1:
                        t = 0
                        array.append(t)
                    else:
                        t = 1
                        array.append(t)
                elif data.name == 'hddTemp':
                    if xi >= 42 and xi <= 55:
                        t = 0
                        array.append(t)
                    else:
                        t = 1
                        array.append(t)
            else:
                if p <= pA and not p < pM:  # If the value of p is less then or equal to smallest pdf value and not smaller than highest pdf value than proceed
                    if xi > mu:  # If that provided data is more than the average, than the variable is malicious
                        t = 0
                        array.append(t)
                    else:  # Else it is not malicious
                        t = 1
                        array.append(t)
                else:  # Else it is not malicious
                    t = 1
                    array.append(t)

        return array  # return the results

    def Accuracy(target, prediction):  # Checks the accuracy of the classifier predictions
        newfile = open('results.csv', 'w', newline='')  # creates new csv file
        newfileW = csv.writer(newfile)
        i = 0  # Number of correct comparisons
        coVar = 0  # current variable in Prediction array
        for xi in target:  # For each value in target
            newfileW.writerow([xi, prediction[coVar]])  # Write results in csv file for manual checking
            if xi == prediction[
                coVar]:  # If the value xi in provided targets is equal to the target value predicted by the classifier than proceed
                i += 1  # Add 1 to correct comparisons
                coVar += 1  # Add 1 to current position in Prediction array
            else:
                coVar += 1  # Else add 1 to current position in prediction array
        return (i / len(target)) * 100  # Number of correct comparisons/the length of the dataset in percentage
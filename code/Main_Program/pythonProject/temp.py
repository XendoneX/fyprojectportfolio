import numpy as np
import pandas as pd
import csv
import sympy as sy
import time
from sklearn.model_selection import train_test_split as tts
import joblib
import math

class classifier:
    def MLE(data): 
        N=len(data) #number of values in data
    #Mu Hat  
        μHat=float((1/N)*sum(data)) #1/N (number of variables in the given dataset) X sum of every variable in the dataset
        
    #Sigma Hat
        finalFile = open('mleSigmaMu.csv', 'w', newline='') #creates new writable csv file
        writer2= csv.writer(finalFile)
        for xi in data: #For each value in the given dataset #Calculate first half of equation
            p1=(xi-μHat)**2 #Dataset value xi - μHat 
            writer2.writerow(np.array([p1])) #Write answer in csv
        finalFile.close() #close csv file
        p1csv = pd.DataFrame(pd.read_csv('mleSigmaMu.csv')) #Read csv file
        σHat = sy.sqrt(1/N*np.sum(p1csv)) #square root(1/N (Number of variables in dataset)) x sum of equation 1
        return μHat, σHat #return the 2 values
            
    def PDF(mu, sigma, minIntegral, maxIntegral, data, csvfile): #ths function will calculate the probability density function using the provided variables (mu= mean of dataset, lowest integral variable, highest integral value, csvfile to write results in)
        pdffile = open('hddPdf.csv', 'w', newline='') #creates new csv to store PDF results
        pdf= csv.writer(pdffile)
        x = sy.Symbol('x')
        for xi in data: #For each value in data produce PDF Value
            gx=1/(sigma*sy.sqrt(2*np.pi))*sy.exp((-(xi-mu)**2)/(2*sigma**2)) #PDF Algorithm
            pdf.writerow(np.array([gx], dtype=float)) #Writes results in a csv file
        pdffile.close()
            
        writer= csv.writer(csvfile) #creates a new csv writer function
        val=minIntegral #lowest number of the definite integral
        while val!=maxIntegral: #function will run until the provided maximum integral in reached
            row = sy.integrate((1/(sigma*sy.sqrt(2*sy.pi)))*sy.exp((-(x-mu)**2)/(2*sigma**2)), (x, val, val+1)) #Sympy.integrate function uses the PDF algorithm located in the g(x) function above, proceeding to calculate the definite integral of the minumum integral and minumum integral+1 until it reaches the provided maximum integral.
            if row.has(x):
                writer.writerow(np.array([0.0], dtype=float)) #writes the results in a row of the provided csv file
            else:
                if math.isnan(float(row)):  #Filter out NaN value to 0
                    writer.writerow(np.array([0.0], dtype=float)) 
                else:
                    writer.writerow(np.array([row], dtype=float))
            csvfile.flush()
            val+=1 #increments the min integral
            time.sleep(1) #pauses for 1 second in order to wait for the results to be writen in the csv file
    
    def Predict(mu, sigma, pdf, data): #Predicts the whether the provided data/variable is malicious or not
        ymin=float(min(pdf)) #smallest PDF value from the integral
        ymax=float(max(pdf)) #largest PDF value from the integral
    
        array=[]
        for xi in data:
            p = 1/(sigma*sy.sqrt(2*np.pi))*sy.exp((-(xi-mu)**2)/(2*sigma**2)) #PDF Algorithm
            if math.isnan(float(p)): #Filter out NaN values to 0
                newP=(float(p))
            else:
                newP=p
                
            if newP <= ymax and not newP < ymin: #If the value of p is less then or equal to smallest pdf value and not smaller than highest pdf value than proceed
                if xi>mu: #If that provided data is more than the average, than the variable is malicious
                    t=0  
                    array.append(t)
                else:     #Else it is not malicious
                    t=1
                    array.append(t)
            else:   #Else it is not malicious
                t=1
                array.append(t)
                
        return array #return the results
    
    def Accuracy(target, prediction):   #Checks the accuracy of the classifier predictions
        newfile=open('results.csv', 'w', newline='') #creates new csv file
        newfileW=csv.writer(newfile) 
        i=0 #Number of correct comparisons
        coVar=0 #current variable in Prediction array
        for xi in target: #For each value in target 
            newfileW.writerow([xi, prediction[coVar]]) #Write results in csv file for manual checking
            if xi == prediction[coVar]: #If the value xi in provided targets is equal to the target value predicted by the classifier than proceed
                i+=1    #Add 1 to correct comparisons 
                coVar+=1    #Add 1 to current position in Prediction array
            else:
                coVar+=1 #Else add 1 to current position in prediction array
        return (i/len(target))*100  #Number of correct comparisons/the length of the dataset in percentage
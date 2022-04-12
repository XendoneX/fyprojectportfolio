import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import csv
import sympy as sy
import time
from scipy.stats import norm
from sklearn.model_selection import train_test_split as tts
import joblib

#Imports data set with these column names
names=['cpuLoad', 'cpuTemp', 'cpuPower', 'ramLoad', 'ramData', 'hddTemp', 'cpuTargets', 'hddTargets']
data= pd.DataFrame(pd.read_csv('combined_data.csv', names=names))
mlefile = open('hddMle.csv', 'w', newline='') #creates new writable csv to store MLE results
mle= csv.writer(mlefile)

#Seperate dataset into individual columns
cpuload=data['cpuLoad']
cputemp=data['cpuTemp']
cpupower=data['cpuPower']
cpuTarget= data['cpuTargets']
hddtemp=data['hddTemp']
hddTarget=data['hddTargets']

modelfile = 'finalmodel.sav'

def split(data, target): #Splits the provided dataset and its targets
    X = data 
    y = target
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.6, random_state=(0)) #divides dataset and targets into test and target datasets in a 60:40 ratio
    return X_train, X_test, y_train, y_test

class classifier:
    def MLE(data): 
        N=len(data) #number of values in data
    #Mu Hat  
        μHat=float((1/N)*np.sum(data)) #1/N (number of variables in the given dataset) X sum of every variable in the dataset
        
    #Sigma Hat
        finalFile = open('mleSigmaMu.csv', 'w', newline='') #creates new writable csv file
        writer2= csv.writer(finalFile)
        for xi in data: #For each value in the given dataset #Calculate first half of equation
            p1=(xi-μHat)**2 #Dataset value xi - μHat 
            writer2.writerow(np.array([p1])) #Write answer in csv
        finalFile.close() #close csv file
        p1csv = pd.DataFrame(pd.read_csv('mleSigmaMu.csv')) #Read csv file
        σHat = math.sqrt(1/N*np.sum(p1csv)) #square root(1/N (Number of variables in dataset)) x sum of equation 1
        return μHat, σHat #return the 2 values
            
    def g(x, mu, sigma):
        return (1/(sigma*sy.sqrt(2*sy.pi)))*sy.exp((-(x-mu)**2)/(2*sigma**2))    #PDF Algorithm for Definite Integral
     
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
            row = sy.integrate(classifier.g(x, mu, sigma), (x, val, val+1)) #Sympy.integrate function uses the PDF algorithm located in the g(x) function above, proceeding to calculate the definite integral of the minumum integral and minumum integral+1 until it reaches the provided maximum integral.
            writer.writerow(np.array([row], dtype=float)) #writes the results in a row of the provided csv file
            csvfile.flush()
            val+=1 #increments the min integral
            time.sleep(1) #pauses for 1 second in order to wait for the results to be writen in the csv file
    
    def predict(mu, sigma, pdf, data): #Predicts the whether the provided data/variable is malicious or not
        ymin=float(min(pdf)) #smallest PDF value from the integral
        ymax=float(max(pdf)) #largest PDF value from the integral
    
        array=[]
        for xi in data:
            p = 1/(sigma*sy.sqrt(2*np.pi))*sy.exp((-(xi-mu)**2)/(2*sigma**2)) #PDF Algorithm
            if p <= ymax and not p < ymin: #If the value of p is less then or equal to smallest pdf value and not smaller than highest pdf value than proceed
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
    
    def accuracy(target, prediction):   #Checks the accuracy of the classifier predictions
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
    
        
def gaussian(val1, val2, mu, sigma, data, ca): #Draws Normal Distribution
    plt.hist(data, bins=25, density=True, alpha=0.6, color='b') #Plots histogram of the provided dataset variable
    
    xmin, xmax = plt.xlim() #Gets the smallest and largest values in the dataset
    x = np.linspace(xmin, xmax, 100) #creates values for the normal distribution using the smallest and largest data points
    p = norm.pdf(x, mu, sigma) #plots the normal distribution using the provided mu and sigma from MLE Function
    plt.plot(x, p, 'k', linewidth=2)#Plots the distribution
    plt.fill_between(x, p, where=((x >= mu+sigma) & (x < val2)), facecolor='orange', alpha=0.5) #Fill area under the normal distribution where malicious activity takes place
    plt.xlabel(data.name, fontsize=10) #Label X axis
    plt.show() #Show graph
    
    
if __name__ == "__main__":
#Create Model
    model=classifier
#INSERT VARIABLES HERE!!!!!!!!!!!!!!!!!!!!!!
    currentData=cpuload                     #Insert Dataset
    currentTarget=cpuTarget.to_numpy()      #Insert Targets
    val1=3.0                                 #Integral 1
    val2=11.0                              #Definite Integral 2
    
#Split main Dataset and Targets
    X_train, X_test, y_train, y_test = split(currentData, currentTarget)
    print('Accuracy using MLE for', currentData.name, 'Dataset')
    print('Find values between', val1, '&', val2)

#Training Process
    mu, sigma = model.MLE(X_train)
    with open('diPdf.csv', 'w', newline='') as dipdffile:    #FILE NAME   
        model.PDF(mu, sigma, val1, val2, X_train, dipdffile)
    names=['PDF']
    getPdf= pd.DataFrame(pd.read_csv('diPdf.csv', names=names))
    pdf=getPdf.to_numpy()
    pValue= np.sum(getPdf)
    prediction=model.predict(mu, sigma, pdf, X_train)
    accScore=model.accuracy(y_train, prediction)
    print('Training Accuracy:', accScore, '%')

  #  newdata= hddtemp.to_numpy(dtype=float)
    gaussian(val1, val2, mu, sigma, currentData, pValue)
   # train, test = split(hddtemp)
   
#Test Process
    testMu, testSigma = model.MLE(X_test)
    with open('testPDF.csv', 'w', newline='') as testFile:    #FILE NAME   
        model.PDF(testMu, testSigma, val1, val2, X_test, testFile)
    gettestPdf= pd.DataFrame(pd.read_csv('diPdf.csv', names=names))
    testPdf=gettestPdf.to_numpy()
    testPred=model.predict(testMu, testSigma, testPdf, X_test)
    testScore=model.accuracy(y_test, testPred)
    print('Testing Accuracy:', testScore, '%')
    
#create and test model
    joblib.dump(model, modelfile)
    loadmodel= joblib.load(modelfile)

    modelMu, modelSigma = loadmodel.MLE(currentData)
    with open('modelPdf.csv', 'w', newline='') as modelFile:    #FILE NAME   
        loadmodel.PDF(modelMu, modelSigma, val1, val2, currentData, modelFile)
    getmodelPdf= pd.DataFrame(pd.read_csv('modelPdf.csv', names=names))
    modelPdf=getmodelPdf.to_numpy()
    modelprediction=loadmodel.predict(modelMu, modelSigma, modelPdf, currentData)
    modelScore=loadmodel.accuracy(currentTarget, modelprediction)
    print('MODEL SCORE IS: ', modelScore, '%')
 #   print('test model=', testmodel)
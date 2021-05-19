'''
This document was written by Ebubekir Stuart Clark as part of a CHEM2999 project at UNSW
on April 2021
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

import json

import rdkit as rdk
from rdkit.Chem import AllChem as rdkac
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

MAGIC_NUMBER = 42 # used for random seed generation throughout

def binFrequencies (df, wavenumber, binSize):
    '''
    A function which categorises frequencies into a 0 or 1 depending on whether the molecule has an IR signal
    within a given range.
    '''
    for ind in df.index:
        listContains = False
        for fre in df.loc[ind, 'Frequencies']:
            if (wavenumber - binSize <= fre and fre <= wavenumber + binSize):
                listContains = True # This is to keep track of the occurance only once               
        if listContains == False:
            df.loc[ind, 'Frequencies'] = 0
        else:
            df.loc[ind, 'Frequencies'] = 1

def preprocess_data(wavenumber, binSize):
    '''
    A function which preprocesses a given data file, given a wavenumber of interest and bin size,
    and saves it as a dataframe in file named 'processedData.csv'
    '''
    df =  pd.read_csv('fundscaled_data.csv')
    df = df.drop(columns= ['Formula', 'Kind', 'Mode', 'Intensity', 'Problematic_Mode'])
    # Group the frequencies into columns
    df = df.groupby('SMILES')['Frequency'].apply(list).reset_index(name='Frequencies')

    # Crop the data to a certain variable frequency and bin size
    binFrequencies(df, wavenumber, binSize)
    
    # Return a list of the frequencies for a given wavenumber and binsize
    return df['Frequencies'].to_list()

def plot_prior(df):
    '''
    This function plots the prior class distribution for every bin in the spectra for our data
    '''
    smiles = df['SMILES']
    stepSize = 25
    wavenumbers = np.arange(0, 3800, stepSize)
    binSizes = [25, 50, 75, 100]
    for binSize in binSizes:
        ratios = []
        for k in wavenumbers:
            Freqs = preprocess_data(k, binSize)
            ratio = sum(Freqs)/len(Freqs)
            #if ratio < 0.5:
            #    ratio = 1 - ratio
            ratios.append(ratio)
        plt.plot(wavenumbers, ratios, label=f"binwidth = {binSize*2} cm$^{-1}$")
    plt.ylabel("Ratio of class 1")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.legend(bbox_to_anchor=(0.72, 0.86), fontsize=10)
    #plt.legend(loc="lower center", ncol=2, bbox_to_anchor=(0., 1.15, 1., .115), fontsize=15)
    plt.savefig("BinSizeDist.png")
    plt.show()

def getMolInfo(in_smiles, r):
    '''
    This is a function which, given ONE smiles code and a circular fingerprint radius, returns a 
    list of unique morgan fingerprints
    '''
    bit_info = {}
    m = rdk.Chem.MolFromSmiles(in_smiles)
    rdkac.GetMorganFingerprint(m, r, bitInfo = bit_info)
    return list(bit_info.keys())

def plot_unique_frags(uniqueFragLabels, uniqueFragsCount):
    '''
    This function plots the unique fragment feature space to a file named fragsPlot.png
    '''
    #plt.bar(uniqueFragLabels[:250], uniqueFragsCount[:250])
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(15,15))

    ax = sns.barplot(x=uniqueFragLabels[:100], y=uniqueFragsFreq[:100], dodge=False)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(labelsize=20)
    ax.set_ylabel('Frequency', fontsize=30)
    ax.set_xlabel('\nUnique Fragment IDs', fontsize=30)

    ax1 = fig.add_subplot(111, frameon=False)
    #ax1.set_xlabel('\n\nUnique Fragment IDs', fontsize=20)
    #ax1.set_ylabel('Frequency', fontsize=20)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    xt = uniqueFragLabels[:100]
    
    #ax.set_xticks(xt[::2])#, fontsize=4, rotation=90)
    ax.set_xticklabels(xt, fontsize=4, rotation=90)
    for label in ax.get_xticklabels()[::2]:
        label.set_visible(False)
    plt.savefig('fragsPlot2.png')
    plt.show()

def buildFeatureSpace(smiles, r):
    '''
    A function which builds a custom featuriser vector of fixed length given a list of SMILES codes and
    a circular fingerprint radius. Saves a barplot distribtion of the most frequent fragments. Returns
    a sorted list of uniqueFrags from most to least frequent
    '''
    uniqueFragsDict = {}
    for smile in smiles:
        # Get the molecular fragments as per the morgan fingerprint algorithm using rdkit
        molecularFrags = getMolInfo(smile, r)
        # loop through each morgan fingerprint unique id
        for frag in molecularFrags:
            if frag not in uniqueFragsDict:
                uniqueFragsDict[frag] = 1
            else:
                uniqueFragsDict[frag] += 1
    # uniqueFragsDict should now contain all the unique fragments in the set
    
    # Sort the dictionary by value
    sortedFrags = sorted(uniqueFragsDict.items(), key=lambda x: x[1], reverse=True)
    # Unpack the sorted list of tuples into lits of their fragments and counts
    uniqueFrags, uniqueFragsCount = zip(*sortedFrags)

    total = len(smiles)
    # convert counts to frequencies   
    uniqueFragsFreq = [x/len(smiles) for x in uniqueFragsCount]
    
    # convert frags to string labels
    uniqueFragLabels = [str(x) for x in uniqueFrags]
    
    # plot_unique_frags(uniqueFragLabels, uniqueFragsCount) #UNCOMMENT TO PLOT FEATURE SPACE

    uniqueFrags = list(uniqueFrags)
    #split = math.floor(len(uniqueFrags)/2)
    #return uniqueFrags[:split] # We can test returning a fraction of the full feature space 
    #return uniqueFrags[:100] # We can test returning the most frequent features only

    return uniqueFrags # Sorted list of fragments from most to least frequent

def findMorganID(smiles, fragID, r):
    '''
    A function which finds an example of a given fragID in a list of smiles codes.
    Returns that molecule's bit_info so it can be used to plot the fragment.
    '''
    for in_smiles in smiles:
        bit_info = {}
        m = rdk.Chem.MolFromSmiles(in_smiles)
        rdkac.GetMorganFingerprint(m, r, bitInfo = bit_info)
        if fragID in bit_info.keys():
            return in_smiles

def drawFrags(in_smiles, fragID, r, fn):
    '''
    A function which, given a smile code with a particular frag, a query fragment id, and the radius for 
    which it was calculated, draw the fragment as save it to a file
    '''
    bit_info = {}
    m = rdk.Chem.MolFromSmiles(in_smiles)
    rdkac.GetMorganFingerprint(m, r, bitInfo = bit_info)
    frags = [(m, x, bit_info) for x in bit_info]
    #img = Draw.DrawMorganBits(frags, legends=[str(x) for x in bit_info])
    img = Draw.DrawMorganBit(m, fragID, bit_info, legend=str(fragID))
    img.save(f'imgs/frag{fn}r{r}.png')

def customFeaturise(smiles, uniqueFrags, r):
    '''
    A function which given a smiles code list, returns a feature vector of len(uniqueFrags)
    to be an input for the classifier for each smiles. Returns a list of feature vectors
    '''
    featVecList = []
    for smile in smiles:
        frags = getMolInfo(smile, r)
        featVec = np.zeros(len(uniqueFrags)) # choose to represent lack of fragment as a 0?
        for frag in frags:
            if frag in uniqueFrags:
                ind = uniqueFrags.index(frag)
                featVec[ind] = 1 # presence of fragment set as 1
        featVecList.append(featVec)
    return featVecList

def findSMILESfromBitVector(x, smiles, uniqueFrags, r):
    '''
    A function which given a bit vector, reverse engineers it to find the SMILES code
    '''

    for smile in smiles:
        frags = getMolInfo(smile, r)
        featVec = np.zeros(len(uniqueFrags)) # choose to represent lack of fragment as a 0?
        for frag in frags:
            if frag in uniqueFrags:
                ind = uniqueFrags.index(frag)
                featVec[ind] = 1 # presence of fragment set as 1
        if all(map(lambda x, y: x == y, featVec, x)):
            return smile

def train_model(X, y, model):
    '''
    A function which featurises the SMILES code from the data, trains the model, and evaluates and analyses the 
    accuracy of the model. Returns a dictionary of the {'metric_name': metric_score}
    '''   
    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=MAGIC_NUMBER)
    # Fit and train the model
    model.fit(X=X_train, y=y_train)

    #test_score = recall_score(y_test, model.predict(X_test), zero_division=1)
    #return test_score

    # returns tn, fp, fn, tp
    #scores = confusion_matrix(y_test, model.predict(X_test)).ravel()
    y_pred = model.predict(X_test)
    return y_pred.tolist(), y_test
    
def empty_json(filename):
    '''
    This file empties the json file dataframe for a given filename
    '''
    data = {'calcs': []}

    json.dump(data, open(filename, "w"))

def varyingModelOptimisationToJSON(df):
    '''
    Given a dataframe, generate the predictions for varying model and radius,
    and dump into a JSON file named varied_data.json
    '''
    # clear any previous data in the JSON file
    empty_json("varied_data.json")

    smiles = df['SMILES'].to_numpy()

    models = [SVC(), DecisionTreeClassifier(), LogisticRegression(), BernoulliNB()]
    #model = DecisionTreeClassifier()
    radii = [0, 1, 2, 3]
    stepSize = 25
    wavenumbers = np.arange(0, 3800, stepSize).tolist()
    binsize = 100
    #binsizes= [10, 25, 50, 75] # OPTIONAL if we wish to vary binsizes instead of models
    # Loop through the radii
    for j, r in enumerate(radii):
        # Build a new feature space for each radii
        uniqueFrags = buildFeatureSpace(smiles, r)
        featVecs = np.stack(customFeaturise(smiles, uniqueFrags, r))
        # Loop through the models
        for i, model in enumerate(models):
        #for binsize in binsizes: # For varying binsizes instead of models
            # Loop through the wavenumbers and generate predictions
            y_pred_list = []
            y_true_list = []
            for k in wavenumbers:
                Freqs = preprocess_data(k, binsize) 
                y_pred, y_true = train_model(X=featVecs, y=Freqs, model=model)
                y_pred_list.append(y_pred) # append the vector to the list, can append model metric instead
                y_true_list.append(y_true)
            # insert calculations in the dictionary
            FILE  = open("data.json", "r")
            data = json.load(FILE)
            FILE.close()
            print(data)

            #append the data to the dataframe
            data['calcs'].append ({
                'binSize': binsize,
                'stepSize': stepSize,
                'radius': r,
                'model': str(model)[:-2],
                'feat_space_divisor': 1,
                'wavenumbers': wavenumbers,
                'y_pred_list': y_pred_list,
                'y_true_list': y_true_list,
            })

            json.dump(data, open("varied_data.json", "w"))

    print("------------------------")
    print("FINISHED CALCULATIONS")
    print("------------------------")

def drawCommonRFrags():
    '''
    This code is for generating png files for the 10 most common fragments for r=[0,1,2,3] and
    storing them in an /imgs/ directory
    '''
    smiles = df['SMILES'].to_numpy()

    # draw 10 most common fragments in r=0, r=1, r=2
    for r in range(0, 4):
        uniqueFrags = buildFeatureSpace(smiles, r)
        # test with first element in the list
        for query_id in uniqueFrags[:10]:
            query_smiles = findMorganID(smiles, query_id, r)
            drawFrags(query_smiles, query_id, r, fn='')

def openDTModel(df, wavenumber, binsize, r, max_depth=None, min_samples_leaf=1):
    '''
    This function opens up the Decision Tree model to closer analyse its decision pathway
    '''
    smiles = df['SMILES'].to_numpy()
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=MAGIC_NUMBER)

    # Build a new feature space for each radii
    uniqueFrags = buildFeatureSpace(smiles, r)
    Freqs = preprocess_data(wavenumber, binsize)
    featVecs = np.stack(customFeaturise(smiles, uniqueFrags, r))

    X_train, X_test, y_train, y_test = train_test_split(featVecs, Freqs, test_size=0.2, random_state=MAGIC_NUMBER)
    model = model.fit(X=X_train, y=y_train)
    y_predict = model.predict(X_test)

    fns = [] # stores the indices of false positives
    fps = [] # stores the indices of false negatives
    for ind, val in enumerate(y_predict):
        # 2 cases for FN and FP
        if y_test[ind] == 1 and val == 0: # FN
            fns.append(ind)
        if y_test[ind] == 0 and val == 1: # FP
            fps.append(ind)

    ratio = sum(y_test)/len(y_test)
    if ratio < 0.5:
        ratio = 1 - ratio
    print("Maximum classifier accuracy is: ", ratio)

    print(f'For model trained at wavenumber={wavenumber} and r={r}')

    #print("Differing Indices are: ", ind_diff)
    print("False negative indices are: ", fns)
    print("False positive indices are: ", fps)
    
    # draw every false negative
    for ind in fns: # for every misclassified instance
        mcSMILE = findSMILESfromBitVector(X_test[ind], smiles, uniqueFrags, r)  # find the smiles codes misclassified
        #print(mcSMILE)
        p = rdk.Chem.MolFromSmiles(mcSMILE)
        Draw.MolToFile(p, f"imgs/fn/{mcSMILE}.png") # draw the misclassified instance
    # draw every false positive
    for ind in fps: # for every misclassified instance
        mcSMILE = findSMILESfromBitVector(X_test[ind], smiles, uniqueFrags, r)  # find the smiles codes misclassified
        #print(mcSMILE)
        p = rdk.Chem.MolFromSmiles(mcSMILE)
        Draw.MolToFile(p, f"imgs/fp/{mcSMILE}.png") # draw the misclassified instanc
    
    acc = accuracy_score(y_test, y_predict, normalize=True)
    
    print(f'Accuracy score = {acc}')
    #print(model.decision_path(X_test))

    tree.plot_tree(model)
    plt.savefig(f'tree{r}.png')
    
    print(tree.export_text(model))

def plotFeatureNumber(fn, r):
    '''
    A function which plots a given feature number for a given radius and feature space = uniqueFrags
    '''
    smiles = df['SMILES'].to_numpy()
    uniqueFrags = buildFeatureSpace(smiles, r)

    f = uniqueFrags[fn]
    query_smiles = findMorganID(smiles, f, r)
    drawFrags(query_smiles, f, r, fn)

def optimiseDT(df, stepSize, r):
    '''
    A function which tunes the hyperparameters and optimises the Decision Tree classifier for
    various hypyerparameters and binsizes. Dumps results in JSON file named DTCV.json
    '''
    # clear any previous data in the JSON file
    empty_json("DTCV.json")

    smiles = df['SMILES'].to_numpy()
    wavenumbers = np.arange(0, 3800, stepSize).tolist()
    model = DecisionTreeClassifier()

    depth_vals = [1, 2, 3, 4]
    msl_vals = [4, 9, 16, 25]
    params = {'max_depth': depth_vals, 'min_samples_leaf': msl_vals}

    binsizes = [10, 25, 50, 75, 100]
    for binSize in binsizes:

        scores = []
        max_depth_list = []
        min_samples_leaf_list = []
        uniqueFrags = buildFeatureSpace(smiles, r) # Unique feature space
        featVecs = np.stack(customFeaturise(smiles, uniqueFrags, r)) # X feature matrix
        prior_ratios = []
        for k in wavenumbers:
            Freqs = preprocess_data(k, binSize)  # y target vector
            X_train, X_test, y_train, y_test = train_test_split(featVecs, Freqs, test_size=0.2, random_state=MAGIC_NUMBER)

            ratio = sum(y_test)/len(y_test)
            if ratio < 0.5:
                ratio = 1 - ratio
            prior_ratios.append(ratio)

            search = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=10, refit=True, return_train_score=True)
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            #print(search.best_params_['max_depth']) 
            #scores.append(search.best_score_)
            scores.append(accuracy_score(y_test, y_pred))
            max_depth_list.append(search.best_params_['max_depth'])
            min_samples_leaf_list.append(search.best_params_['min_samples_leaf'])
            #print("Best estimator depth = ", search.best_estimator_)

        # insert calculations in the dictionary
        FILE  = open("DTCV.json", "r")
        data = json.load(FILE)
        FILE.close()
        #print(data)
        print("Binsize = ", binSize)
        #append the data to the dataframe
        data['calcs'].append ({
            'binSize': binSize,
            'stepSize': stepSize,
            'r': r,
            'wavenumbers': wavenumbers,
            'scores': scores,
            'prior_ratios': prior_ratios,
            'max_depth_list': max_depth_list,
            'min_samples_leaf_list': min_samples_leaf_list,
        })

        json.dump(data, open("DTCV.json", "w"))

    print("------------------------")
    print("FINISHED CALCULATIONS")
    print("------------------------")

def plot_varied_data_from_JSON():
    '''
    This function plots the varied data calculated in varyingModelOptimisationToJSON() from varied_data.json 
    to a file named variedData.png
    '''
    FILE  = open("varied_data.json", "r")
    data = json.load(FILE)
    FILE.close()

    fig, ax = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(15,15))
    ax1 = fig.add_subplot(111, frameon=False)
    xt = np.arange(0, 3801, 500)
    yt = np.arange(0, 1, 0.1)
    all_accs = []

    for ind, ax in enumerate(ax.flat):
        calc = data['calcs'][ind]
        y_pred_list = calc['y_pred_list']
        y_true_list = calc['y_true_list']

        ratios = []
        accScores = []

        for i, vector in enumerate(y_pred_list):          
            accScores.append(accuracy_score(y_true_list[i], y_pred_list[i], normalize=True))
            # Ratio of max class
            ratio = sum(y_true_list[i])/len(y_true_list[i])
            if ratio < 0.5:
                ratio = 1 - ratio
            ratios.append(ratio)

        all_accs.append(sum(accScores)/len(accScores))

        ax.plot(calc['wavenumbers'], accScores, label='Model')
        ax.plot(calc['wavenumbers'], ratios, '--', label='Maximum Class Classifier')
        ax.set_xticks(xt)
        ax.set_xticklabels(xt, rotation=90)#, fontsize=20)
        
        ax.tick_params(labelsize=15)
        
        avg_acc = sum(accScores)/len(accScores)
        avg_ratio = sum(ratios)/len(ratios)
        avg_diff_percent = format((avg_acc - avg_ratio)*100, '.2f') # percentages to 2 decimal places

        ax.set_title(f"{calc['model']}, r = {calc['radius']} \n Average acc. difference = {avg_diff_percent}%")
        if ind == 1:
            ax.legend(loc="lower left", ncol=2, bbox_to_anchor=(0., 1.15, 1., .115), fontsize=15) # For prior prob plot
    
    ax1.set_ylabel("Accuracy", fontsize = 30, labelpad = 40)
    ax1.set_xlabel("\nWavenumber (cm$^{-1}$)", fontsize = 30, labelpad = 40)
    ax1.set_yticklabels([], fontsize=20)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    
    plt.savefig('variedData.png')
    plt.show()
    
def plot_DTCV_from_JSON():
    '''
    This function plots the varied data calculated in optimiseDT() from DTCV.json 
    to a file named DTCV.png
    '''
    FILE = open("DTCV.json", "r")
    data = json.load(FILE)
    FILE.close()

    fig, ax = plt.subplots(5, 1, sharex=True, sharey=False, figsize=(15,15))
    ax1 = fig.add_subplot(111, frameon=False)

    xt = np.arange(0, 3801, 500)
    yt = np.arange(0, 1, 0.1)

    for ind, ax in enumerate(ax.flat):
        calc = data['calcs'][ind]

        ax.plot(calc['wavenumbers'], calc['scores'], label='Model')
        ax.plot(calc['wavenumbers'], calc['prior_ratios'], '--', label='Maximum Class Classifier')
        ax.set_xticks(xt)
        ax.set_xticklabels(xt, rotation=90)#, fontsize=20)
        ax.tick_params(labelsize=15)
        avg_score = sum(calc['scores'])/len(calc['scores'])
        avg_ratio = sum(calc['prior_ratios'])/len(calc['prior_ratios'])
        avg_score_percent = format(avg_score*100, '.2f')
        avg_diff_percent = format((avg_score - avg_ratio)*100, '.2f')

        ax.set_title(f"binwidth = {calc['binSize']*2}, r = {calc['r']} \n Average accuracy difference = {avg_diff_percent}%")
        if ind == 0:
            ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0., 1.15, 1., .115), fontsize=15) # For prior prob plot

    ax1.set_ylabel("Accuracy", fontsize = 30, labelpad = 40)
    ax1.set_xlabel("\nWavenumber (cm$^{-1}$)", fontsize = 30, labelpad = 40)
    ax1.set_yticklabels([], fontsize=20)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_xticklabels([])

    plt.savefig("DTCV.png")
    plt.show()

if __name__ == '__main__':
    df =  pd.read_csv('fundscaled_data.csv')
    df = df.drop(columns= ['Formula', 'Kind', 'Mode', 'Intensity', 'Problematic_Mode'])
    df = df.groupby('SMILES')['Frequency'].apply(list).reset_index(name='Frequencies')
    smiles = df['SMILES'].to_numpy()

    # plot_prior(df) # plot the prior class distribution
    
    #varyingModelOptimisationToJSON(df) # Uncomment to generate PERSISTENT new data in varied_data.json
    #plot_varied_data_from_JSON() # Uncomment to plot the data generated in varyingModelOptimisationToJSON

    #drawCommonRFrags(df) # Uncomment to draw the 10 most common fragments for r=[0,1,2,3]
    
    r = 1
    binsize = 25
    stepsize = 25

    # Uncomment this AND the plot function within this function to plot the full feature space
    # buildFeatureSpace(smiles, r) 

    #openDTModel(df, 2300, binsize, r, max_depth=4 , min_samples_leaf=25)

    # what is feature x for r?
    '''
    fns = [5, 39, 7, 26, 0, 20] # fn = feature number of interest
    for fn in fns:
    #fn = 18
        plotFeatureNumber(fn, r) # plot each feature of interest
    '''
    #optimiseDT(df, stepsize, r) # generate PERSISTENT data for optimised decision tree in DTCV.json
    #plot_DTCV_from_JSON() # Uncomment to plot the data generated in optimiseDT

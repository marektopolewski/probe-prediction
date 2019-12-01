import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as RFClassifier

from helper_functions import std_norm, preprocess_probe_data, predict_tna
    
def makeForest():
    """
    Initialise a RF classifier with tuned parameters.
    :return: Tuned RF classfier model
    """
    return RFClassifier(n_estimators=1366, 
                        min_samples_split=5, 
                        min_samples_leaf=4, 
                        max_features='sqrt', 
                        max_depth=30, 
                        bootstrap=True)

def run():
    """
    Main method. Calculates prediction probablities of class for 
    probe B samples and exports to a CSV.
    """
    # Load data into pandas DataFrames
    print('\n[TASK1] Importing data ...')
    probeA = pd.read_csv('../probeA.csv',header=0)
    probeB = pd.read_csv('../probeB.csv',header=0)
    classA = pd.read_csv('../classA.csv', header=0)
    print('[TASK1] Complete.\n')

    # Preprocess data (mixed up column values)
    print('[TASK1] Preprocessing data ...')
    probeA = preprocess_probe_data(probeA)
    probeB = preprocess_probe_data(probeB)

    # Predict 'tna' for probeB
    probeA_tna = probeA['tna']
    probeA = probeA.drop('tna',axis=1)
    probeB['tna'] = predict_tna(probeA, probeB, probeA_tna)
    probeA['tna'] = probeA_tna

    # Standarise and normalise data
    probeA, probeB = std_norm(probeA, probeB)
    print('[TASK1] Complete.\n')

    # Initialise and fit RF model
    print('[TASK1] Building the model and predicting class ...')
    forest = makeForest()
    forest.fit(probeA, classA['class'])

    # Calculate probabilities
    columns = ['prob_class_0','prob_class_1']
    pred_prob = pd.DataFrame(forest.predict_proba(probeB),columns=columns)
    print('[TASK1] Complete.\n')

    # Export probabilities to csv
    print('[TASK1] Exporting results to "classB.csv" ...')
    output = pred_prob[columns[1]]
    output.to_csv('classB.csv', index=False)
    print('[TASK1] Complete.\n')

# Execute program
run()
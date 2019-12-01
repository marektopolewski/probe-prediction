import pandas as pd
import numpy as np

from helper_functions import std_norm, preprocess_probe_data, predict_tna

def run():
    """
    Main function. Predicts 'tna' values of for probe B samples 
    and exports to a CSV.
    """
    # Load data into pandas DataFrames
    print('\n[TASK2] Importing data ...')
    probeA = pd.read_csv('../probeA.csv',header=0)
    probeB = pd.read_csv('../probeB.csv',header=0)
    print('[TASK2] Complete.\n')

    # Preprocess data (mixed up column values)
    print('[TASK2] Preprocessing data ...')
    x_a = preprocess_probe_data(probeA.drop('tna', axis=1))
    x_b = preprocess_probe_data(probeB)
    t_a = pd.DataFrame(probeA['tna'], columns=['tna'])
    print('[TASK2] Complete.\n')

    # Predict 'tna' target for probeB (see method: predict_tna)
    print('[TASK2] Building the model and predicting tna ...')
    t_pred = predict_tna(x_a, x_b, t_a)
    print('[TASK2] Complete.\n')

    # Export 'tna' predictions to csv
    print('[TASK2] Exporting results to "tnaB.csv" ...')
    output = t_pred['tna']
    output.to_csv('tnaB.csv', index=False)
    print('[TASK2] Complete.\n')

# Execute program
run()
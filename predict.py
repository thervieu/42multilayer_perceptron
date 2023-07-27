import os, sys
import numpy as np
import pandas as pd


def main():
    if os.path.isfile('model.npy') is False:
        return print('model.npy not found. Please train model first')
    if os.path.isfile('predict.csv') is False:
        return print('model.npy not found. Please train model first')


    df = pd.read_csv('validation.csv')

if __name__=="__main__":
    main()
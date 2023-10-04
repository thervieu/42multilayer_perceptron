import os
import random
import numpy as np
import pandas as pd

from MLP import MLP

def get_data():
    df_valid = pd.read_csv('resources/validation.csv', header=None)

    df_valid = df_valid.drop(columns=[0])
    df_valid = df_valid.tail(-1)
    
    df_valid = df_valid.rename(columns={1: "class"})
    df_valid['class'] = df_valid['class'].map({'M': 1, 'B': 0})
    df_valid['vec_class'] = df_valid['class'].map({1: [0, 1], 0: [1, 0]})

    x_valid = df_valid.drop(columns=['class', 'vec_class']).to_numpy()
    y_valid = np.asarray(df_valid['vec_class'].tolist(), dtype=object)
    return (x_valid, y_valid)


def get_model():
    model_array = np.load(file='model.npy', allow_pickle=True)

    mlp = MLP(model_array[0])
    mlp.weights = model_array[1]
    mlp.biases = model_array[2]
    mlp.activations = model_array[3]
    
    return mlp


def main():
    if os.path.isfile('model.npy') is False:
        return print('model.npy not found. Please train model first')
    if os.path.isfile('resources/validation.csv') is False:
        return print('validation.csv not found. Please train model first')


    mlp = get_model()

    X_val, y_val = get_data()
    val_activations = mlp.forward_propagation(X_val)
    val_loss = mlp.compute_loss(y_val, val_activations[-1])

    # Calculate accuracy for validation data
    val_predictions = mlp.predict(X_val)
    val_accuracy = mlp.accuracy(y_val, val_predictions)
    print(f'val_loss: {val_loss:.3f}, val_acc: {val_accuracy:.3f}\n')

    print(f'Random Example:')
    nb = random.randint(1, 114)
    
    patient_pred = mlp.predict(np.asarray([X_val[nb]]))

    type_cancer = ['Benign', 'Malignent']
    print(f'Patient number {nb}\'s predicition : {patient_pred[0]}')
    print(f'Patient number {nb}\'s cancer is {type_cancer[np.argmax(patient_pred)]}')



if __name__=="__main__":
    main()
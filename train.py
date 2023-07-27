import os, sys, click
import numpy as np
import pandas as pd

from MLP import MLP

import matplotlib.pyplot as plt

def check_not_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise click.BadParameter("Value must not be negative.")
    return ivalue


def check_not_negative_or_zero_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise click.BadParameter("Value must be a positive integer.")
    return ivalue


def check_not_negative_float(value):
    fvalue = float(value)
    if fvalue < 0:
        raise click.BadParameter("Value must not be negative.")
    return fvalue


def check_not_negative_or_zero_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise click.BadParameter("Value must be a positive number.")
    return fvalue


def check_outliers(value):
    fvalue = float(value)
    if fvalue < 2.0 or fvalue > 4.0:
        raise click.BadParameter("Outliers must be between 2 and 4.")
    return fvalue


def get_data():
    df_train = pd.read_csv('resources/train.csv', header=None)
    df_valid = pd.read_csv('resources/validation.csv', header=None)

    df_train = df_train.drop(columns=[0])
    df_valid = df_valid.drop(columns=[0])
    df_train =df_train.tail(-1)
    df_valid = df_valid.tail(-1)
    # print(df_valid.iloc[:, 0])
    df_train = df_train.rename(columns={1: "class"})
    df_valid = df_valid.rename(columns={1: "class"})
    df_train['class'] = df_train['class'].map({'M': 1, 'B': 0})
    df_valid['class'] = df_valid['class'].map({'M': 1, 'B': 0})
    # print(df_valid['class'])
    df_train['vec_class'] = df_train['class'].map({1: [0, 1], 0: [1, 0]})
    df_valid['vec_class'] = df_valid['class'].map({1: [0, 1], 0: [1, 0]})

    x_train = df_train.drop(columns=['class', 'vec_class']).to_numpy()
    x_valid = df_valid.drop(columns=['class', 'vec_class']).to_numpy()
    y_train = np.asarray(df_train['vec_class'].tolist(), dtype=object)
    # print(df_valid['vec_class'])
    y_valid = np.asarray(df_valid['vec_class'].tolist(), dtype=object)
    # print(f'y valid {y_valid}')
    return (x_train, y_train), (x_valid, y_valid)


def plot_losses_and_accuracies(mlp):
    # Plot losses and accuracies
    epochs = np.arange(0, mlp.epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, mlp.train_losses, label='Training Loss')
    plt.plot(epochs, mlp.valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Training Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, mlp.train_accuracies, label='Training Accuracy')
    plt.plot(epochs, mlp.valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    


@click.command()
@click.option('-L', '--layers', type=check_not_negative_or_zero_int, default=5, help='Number of layers')
@click.option('-U', '--units', type=check_not_negative_or_zero_int, default=12, help='Number of units per layer')
@click.option('-lr', '--learning_rate', type=check_not_negative_or_zero_float, default=1.0, help="Learning Rate's value")
@click.option('-b', '--batch_size', type=check_not_negative_int, default=40, help='Size of batch')
@click.option('-e', '--epochs', type=check_not_negative_or_zero_int, default=300, help='Number of epochs')
@click.option('-p', '--plot', is_flag=True, help='Plot the graphs')
def main(layers, units, learning_rate, batch_size, epochs, plot):
    options = {
        'layers': layers,
        'units': units,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    }

    if (os.path.isfile('resources/train.csv') is False or
        os.path.isfile('resources/validation.csv') is False):
        return (print('Please split data first'))
    try:
        train_data, val_data = get_data()

        options['inputs'] = len(train_data[0][0])
        mlp = MLP(options)
        mlp.train(train_data[0], train_data[1], val_data)
        
        if plot:
            plot_losses_and_accuracies(mlp)
        
        np.save('model.npy', np.array([options, mlp.weights, mlp.biases, mlp.activations], dtype=object))
    except Exception as e:
        print(f'{e}')
        sys.exit(1)


if __name__=="__main__":
    main()
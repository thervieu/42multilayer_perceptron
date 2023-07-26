import sys, click
import timeit
import logging
import numpy as np
import pandas as pd

from MLP import MLP


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


def init_logging():
    logfile = 'metrics.log'
    try:
        level = logging.INFO
        format = '%(message)s'
        handlers = [
                logging.FileHandler(logfile),
                logging.StreamHandler()]
    except Exception as e:
        print("Can't write to {}.".format(logfile))
        print(e.__doc__)
        sys.exit(0)
    logging.basicConfig(level=level, format=format, handlers=handlers)


def describe(data):
    clean_data = {k: data[k] for k in data if not np.isnan(data[k])}
    values = np.sort(np.array(list(clean_data.values()), dtype=object))
    count = len(clean_data)
    stats = {'count': count}
    stats['mean'] = sum(clean_data.values()) / count
    stats['var'] = (
            1
            / (count - 1)
            * np.sum(np.power(values - stats['mean'], 2)))
    stats['std'] = np.sqrt(stats['var'])
    return stats


def feature_scaling(df, stats):
    for subj in stats:
        df[subj] = (df[subj] - stats[subj]['mean']) / stats[subj]['std']
    return df


def pre_process(df, stats, options):
    df = feature_scaling(df, stats)
    df = df.rename(columns={1: "class"})
    df['class'] = df['class'].map({'M': 1, 'B': 0})
    df['vec_class'] = df['class'].map({1: [0, 1], 0: [1, 0]})
    if options['shuffle']:
        df = df.sample(frac=1)
    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)

    x_train = dfs[0].drop(columns=['class', 'vec_class']).to_numpy()
    x_val = dfs[1].drop(columns=['class', 'vec_class']).to_numpy()
    y_train = np.asarray(dfs[0]['vec_class'].tolist())
    y_val = np.asarray(dfs[1]['vec_class'].tolist())
    return (x_train, y_train), (x_val, y_val)


@click.command()
@click.argument('dataset', nargs=1, default="data.csv")
@click.option('-L', '--layers', type=check_not_negative_or_zero_int, default=4, help='Number of layers')
@click.option('-U', '--units', type=check_not_negative_or_zero_int, default=12, help='Number of units per layer')
@click.option('-lr', '--learning_rate', type=check_not_negative_or_zero_float, default=1.0, help="Learning Rate's value")
@click.option('-b', '--batch_size', type=check_not_negative_int, default=40, help='Size of batch')
@click.option('-e', '--epochs', type=check_not_negative_or_zero_int, default=80, help='Number of epochs')
@click.option('-s', '--shuffle', is_flag=True, help='Shuffle the data set')
@click.option('-p', '--patience', type=check_not_negative_int, default=0, help='Number of epochs waited to execute early stopping')
@click.option('-bm', '--bonus_metrics', is_flag=True, help='Precision, Recall and F Score metrics')
def main(dataset, layers, units, learning_rate, batch_size, epochs, shuffle, patience, bonus_metrics):
    options = {
        'layers': layers,
        'units': units,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'shuffle': shuffle,
        'patience': patience,
        'bonus_metrics': bonus_metrics
    }
    print(options)

    # start = timeit.default_timer()
    # init_logging()
    
    try:
        data = pd.read_csv(dataset, header=None)
        data = data.drop(columns=[0])
        stats = {
            column: describe(sub_dict)
            for column, sub_dict in
            data.select_dtypes(include='number').to_dict().items()
        }
        train_data, val_data = pre_process(data, stats, options)
        options['inputs'] = len(train_data[0][0])
        mlp = MLP(options)
        mlp.train(train_data[0], train_data[1], val_data)

    except Exception as e:
        print(f'{e}')
        sys.exit(1)




if __name__=="__main__":
    main()
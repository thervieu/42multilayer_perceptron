import os
import numpy as np
import pandas as pd
import click


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


def pre_process(df, stats):
    df = feature_scaling(df, stats)
    
    df = df.sample(frac=1)

    dfs = np.split(df, [int((len(df) * 0.80))], axis=0)
    return dfs


@click.command()
@click.argument('dataset', nargs=1, default="resources/data.csv")
def main(dataset):
    if os.path.isfile(dataset) is False:
        return print(f'{dataset} does not exist')

    data = pd.read_csv(dataset, header=None)
    data = data.drop(columns=[0])
    stats = {
        column: describe(sub_dict)
        for column, sub_dict in
        data.select_dtypes(include='number').to_dict().items()
    }
    dfs = pre_process(data, stats)

    dfs[0].to_csv('resources/train.csv')
    dfs[1].to_csv('resources/validation.csv')


if __name__=="__main__":
    main()
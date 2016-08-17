import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def proportion(x, min_value):
    vc = x.value_counts()
    if vc.sum() < min_value:
        return vc * 0
    else:
        return vc/vc.sum()


def plot_bars(df, x, y, hue, min_value):
    if (x == hue) or (x == y) or (y == hue):
        print('x, y, hue cannot be equal.')
    else:
        min_value = int(min_value)

        if y == 'No y split.':
            plt.figure(figsize=(10,6))
            sub = df[[x, hue]].dropna(how='any')
            sub = sub.set_index(hue)[x] \
                     .groupby(level=0) \
                     .apply(proportion, min_value=min_value) \
                     .reset_index()
            sub = sub.rename(columns={x: 'frequence','level_1': x})
            sub = sub[~(sub.frequence == 0)]
            sns.barplot(data=sub, x=x, y='frequence', hue=hue)
        else:
            sub = df[[x, y, hue]].dropna(how='any')
            numy = sub[y].unique().shape[0]
            f, axarr = plt.subplots(1, numy, sharey=True, figsize=(16,6))
            subg = sub.groupby(y)
            for i, (k, ssub) in enumerate(subg):
                ssub = ssub.set_index(hue)[x] \
                           .groupby(level=0) \
                           .apply(proportion, min_value=min_value) \
                           .reset_index()

                ssub = ssub.rename(columns={x: 'frequence','level_1': x})
                ssub = ssub[~(ssub.frequence == 0)]
                sns.barplot(data=ssub, x=x, y='frequence',
                            hue=hue, ax=axarr[i])
                axarr[i].set_title('%s is %s' % (y, k))
        plt.xticks(rotation=45)
        plt.show()
        plt.close()


def plot_heatmap(df, x, y):
    if x == y:
        print('x, y cannot be equal.')
    else:
        plt.figure(figsize=(10,6))
        sub = df[[x, y]].dropna(how='any')
        sub['Count'] = 1
        sub = sub.groupby([x, y]).count()
        sub = sub.unstack().fillna(0).astype(int)
        sns.heatmap(data=sub, annot=True, square=True, fmt='d')
        plt.xticks(rotation=45)
        plt.show()

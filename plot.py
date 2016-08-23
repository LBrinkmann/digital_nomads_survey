import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
import sklearn.cluster


def proportion(x, min_value):
    vc = x.value_counts()
    if vc.sum() < min_value:
        return vc * 0
    else:
        s = vc.sum()
        vc[vc < 2] = 0
        return vc/s


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
            sns.barplot(data=sub, x=x, y='frequence', hue=hue, order=sub.sort_values(x)[x].unique().tolist())
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

def plot_n_cluster(dfps):
    all_inertia = []
    n_clusters = range(1,11)
    for n_c in n_clusters:
        km = sklearn.cluster.KMeans(n_clusters=n_c, init='k-means++',
                                    n_init=10, max_iter=300, tol=0.0001,
                                    precompute_distances='auto', verbose=0,
                                    random_state=None, copy_x=True, n_jobs=1)
        km = km.fit(dfps)
        all_inertia.append(km.inertia_)
    plt.plot(n_clusters, all_inertia)

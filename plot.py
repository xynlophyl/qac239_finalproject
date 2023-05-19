import matplotlib.pyplot as plt


def plot_distribution(df, label = ''):

    if not label:
        print('plot error: need to specify label')
        return

    if label not in df:
        print(f'plot error: label ({label}) not found in df')
        return

    vals = df[label]
    plt.hist(vals)
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title(f"Histogram of {label} in Facial Ages Dataset")
    plt.show()
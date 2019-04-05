import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def plot_num_scatter(df,col_1,col_2):
    data = pd.concat([df[col_1], df[col_2]], axis=1)
    data.plot.scatter(x=col_2,y=col_1, ylim=(0,800000))
    plt.show()

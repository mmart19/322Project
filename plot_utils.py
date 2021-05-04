import matplotlib.pyplot as plt
import numpy as np

def bar_chart_example(x, y):
    plt.figure( figsize=(28,8))
    plt.bar(x, y, align="edge", width=.3)
    plt.show()
    
def bar_chart_elite_win(x, y):
    plt.figure()
    plt.bar(x, y, align="edge", width=.3)
    plt.xticks(x)
    plt.show()
    
def bar_chart_seed(x, y, chart_label, x_label, y_label):
    plt.figure()
    plt.bar(x, y, align="edge", width=.3)
    plt.suptitle(chart_label, fontsize=12, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x)
    plt.yticks()
    plt.show()

    
def pie_chart_example(x, y):
    plt.figure()
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()

def histogram_example(data):
    # data is a 1D list of data values
    plt.figure()
    plt.hist(data, bins=10) # default is 10
    #plt.hist(data2, bins=20)
    plt.show()

def scatter_plot(x, y, chart_label, x_label, y_label):
    plt.figure()
    plt.suptitle(chart_label, fontsize=12, fontweight="bold")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x, y)
    plt.show()

def scatter_plot_linear(x, y, m, b):
    plt.figure()
    plt.scatter(x, y)
    plt.show()

def histogram_example_multiple_datasets(data, data2):
    # data is a 1D list of data values
    plt.figure()
    plt.hist(data, bins=10) # default is 10
    plt.hist(data2, bins=10)
    plt.show()


def box_plot_example(distributions, labels):
    # distributions: list of 1D lists of values
    plt.figure(figsize=(16,5))
    plt.boxplot(distributions)
    # boxes correspond to the 1st and 3rd quartiles
    # line in the middle of the box corresponds to the 2nd quartile (AKA median)
    # whiskers corresponds to +/- 1.5 * IQR
    # IQR: interquartile range (3rd quartile - 1st quartile)
    # circles outside the whiskers correspond to outliers
    
    # task: replace the 1 and 2 on the x axis with sigma=10 and sigma=5
    # add a parameters called labels to the function to do generally
    plt.xticks(list(range(1, len(labels) + 1)), labels)
    
    # annotations
    # we want to add "mu=100" to the center of our figure
    # xycoords="data": default, specify the location of the label in the same
    # xycoords = "axes fraction": specify the location of the label in absolute
    # axes coordinates... 0,0 is the lower left corner, 1,1 is the upper right corner
    # coordinates as the plotted data
    plt.annotate("$\mu=100$", xy=(1.5, 20), xycoords="data", horizontalalignment="center")
    plt.annotate("$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction", 
                 horizontalalignment="center", color="blue")

    plt.show()
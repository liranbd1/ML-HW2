import matplotlib.pyplot as plt
from mnist import MNIST
import numpy as np
import pandas as pd

# Ploting the images using matplotlibe subplots in one image
# Due to how data was pulled needed to transform the image list to np array and reshape the size
# to a n by n matrix (28x28 pixels).

def main():
    mndata = MNIST('./Files/')
    images, labels = mndata.load_training()
    #plot_images(images, labels)
    centroid_arr = calc_centroids(images, labels)
    #plot_images(centroid_arr, list(range(0,10)))   
    plot_euclidean_distance(centroid_arr)


def plot_images(images, labels):
    row_size = 2
    col_size = 5

    # plot images
    fig, axes = plt.subplots(row_size, col_size, figsize=(1.5*col_size,2*row_size))
    for i in range(10):
        ax = axes[i//col_size, i%col_size]
        img = np.array(images[i]).reshape(28,28)
        ax.imshow(img, cmap='gray')
        ax.set_title(labels[i])
    plt.tight_layout()
    plt.show()

def calc_centroids(images, labels):
    centroid_arr = []
    for num in range(10):
        img_arr_sum = np.zeros((28,28))
        indices = [i for i, x in enumerate(labels) if x == num]
        for index in indices:
            new_img_arr = np.array(images[index]).reshape(28,28)
            img_arr_sum = np.add(img_arr_sum, new_img_arr)
        centroid_arr.append(np.divide(img_arr_sum, len(indices)))
    return np.array(centroid_arr)

def plot_euclidean_distance(centroids_array):
    df = pd.DataFrame(columns=(range(0,10)),index=(range(0,10)))
    for i, centroid in enumerate(centroids_array):
        for  j, other_centroid in enumerate(centroids_array):
            df.loc[i,j] = np.format_float_positional(np.linalg.norm(centroid-other_centroid),precision=3)
    
    #pd.plotting.table(plt.axes(), df)
    #plt.show()
    cell_text = []
    for row in range(len(df)):
        cell_text.append(df.iloc[row])
    table = plt.table(cellText=cell_text, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    plt.axis('equal')
    plt.show()          

main()

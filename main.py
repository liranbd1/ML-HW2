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
    #centroid_arr = calc_centroids(images, labels)
    #plot_images(centroid_arr, list(range(0,10)))   
    #plot_euclidean_distance(centroid_arr)
    #var_dic = calc_variance(images)
    #plot_var_histogram(var_dic)
    #clean_images = remove_zero_var_pixels(var_dic, images)
    #bob = calc_mean(images)
    calc_cov_matrix(images)

def plot_images(images, labels):
    row_size = 2
    col_size = 5

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

    cell_values = []
    for row in range(len(df)):
        cell_values.append(df.iloc[row])
    table = plt.table(cellText=cell_values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    plt.axis('equal')
    plt.show()          

def calc_variance(images):
    variance_dic = {}
    for image in images:
        npimg = np.array(image).reshape(28,28)
        for cords, pixel in np.ndenumerate(npimg):
            if not cords in variance_dic.keys():
                variance_dic[cords] = []
            variance_dic[cords].append(pixel)
    
    for i, (key, value) in enumerate(variance_dic.items()):
        variance_dic[key] = np.floor(np.var(np.array(value)))

    return variance_dic

def plot_var_histogram(variance_dic):
    var_list = []
    for key, var in variance_dic.items():
        var_list.append(var)
    bin_jump = np.round(max(var_list)/10)
    hist_bins = np.arange(min(var_list), max(var_list), bin_jump).tolist()
    plt.hist(var_list, bins = hist_bins, rwidth=0.5)
    plt.show()

def remove_zero_var_pixels(var_dic, images):
    index_list = []
    clean_images = []
    for key, var in var_dic.items():
        if var == 0:
            i,j = key
            index_list.append((i*28)+j)
    for image in images:
        clean_image = []
        for index, pixel in enumerate(image):
            if not index in index_list:
                clean_image.append(pixel)
        clean_images.append(clean_image)
    return clean_images

def calc_mean(images):

    mean_dic = {}
    for image in images:
        npimg = np.array(image)
        for cords, pixel in np.ndenumerate(npimg):
            if not cords in mean_dic.keys():
                mean_dic[cords] = []
            mean_dic[cords].append(pixel)

    for i, (k,v) in enumerate(mean_dic.items()):
        mean_dic[k] = np.mean(np.array(mean_dic[k]))
    return mean_dic

def calc_cov_matrix(images):
    feature_dic = {}
    for image in images:
        for cord, pixel in enumerate(image):
            if not cord in feature_dic.keys():
                feature_dic[cord] = []
            feature_dic[cord].append(pixel)
    images_mat = np.zeros(len(images))
    for i, (k,v) in enumerate(feature_dic.items()):
        print(k)
        images_mat[k] = np.array(v) # Error need to fix for some reason can't make v into an array.


main()

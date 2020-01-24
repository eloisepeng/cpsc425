import numpy as np
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix
# import pickle

classes_dict = [""] * 15

def build_vocabulary(image_paths, vocab_size):
    """ Sample SIFT descriptors, cluster them using k-means, and return the fitted k-means model.
    NOTE: We don't necessarily need to use the entire training dataset. You can use the function
    sample_images() to sample a subset of images, and pass them into this function.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    vocab_size: the number of clusters desired.
    
    Returns
    -------
    kmeans: the fitted k-means clustering model.
    """
    n_image = len(image_paths)

    # Since want to sample tens of thousands of SIFT descriptors from different images, we
    # calculate the number of SIFT descriptors we need to sample from each image.
    n_each = int(np.ceil(10000 / n_image))

    # Initialize an array of features, which will store the sampled descriptors
    # keypoints = np.zeros((n_image * n_each, 2))
    descriptors = np.zeros((n_image * n_each, 128))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)
        sift_keypoints = features[:, :2]
        sift_descriptors = features[:, 2:]
        # TODO: Randomly sample n_each descriptors from sift_descriptor and store them into descriptors
        r = np.random.choice(sift_descriptors.shape[0], min(n_each, len(sift_descriptors)), replace=False)
        descriptors = np.vstack((descriptors, sift_descriptors[r,]))

    # TODO: perform k-means clustering to cluster sampled sift descriptors into vocab_size regions.
    # You can use KMeans from sci-kit learn.
    # Reference: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = KMeans(n_clusters=vocab_size, n_jobs=8).fit(descriptors)
    return kmeans
    
def get_bags_of_sifts(image_paths, kmeans):
    """ Represent each image as bags of SIFT features histogram.

    Parameters
    ----------
    image_paths: an (n_image, 1) array of image paths.
    kmeans: k-means clustering model with vocab_size centroids.

    Returns
    -------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    """
    n_image = len(image_paths)
    vocab_size = kmeans.cluster_centers_.shape[0]

    image_feats = np.zeros((n_image, vocab_size))

    for i, path in enumerate(image_paths):
        # Load features from each image
        features = np.loadtxt(path, delimiter=',',dtype=float)

        # TODO: Assign each feature to the closest cluster center
        # Again, each feature consists of the (x, y) location and the 128-dimensional sift descriptor
        # You can access the sift descriptors part by features[:, 2:]
        closest_cluster_center = kmeans.predict(features[:, 2:])
        # TODO: Build a histogram normalized by the number of descriptors
        np.add.at(image_feats[i], closest_cluster_center, 1/features.shape[0])

    return image_feats

def plot_histograms(image_feats, labels):
    """ image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    labels: class labels corresponding to each image

    Parameters
    ----------
    image_feats: an (n_image, vocab_size) matrix, where each row is a histogram.
    labels: class labels corresponding to each image
    
    Output/Display
    -------
    histograms of each class
    """
    hist = {}
    for i, label in enumerate(labels):
        hist_label = classes_dict[int(label)]
        cur_hist_data = hist.get(hist_label, (np.zeros((1, image_feats.shape[1])), 0))
        hist[hist_label] = (np.add(cur_hist_data[0], image_feats[i]), cur_hist_data[1]+1)

    for label, (f, count) in hist.items():
        plt.clf()
        plt.bar(np.arange(image_feats.shape[1]), (f[0]/count))
        plt.ylabel("Normalized Count")
        plt.xlabel("Cluster Centers")
        plt.title("Histogram of " + label)
        plt.show()
        # plt.savefig('histograms/' + label + '.png')

def plot_confusion_matrix_u(test_labels, pred_labels, normalize, type): {
    # Reference: https://scikit-learn.org/stable/auto_examples/model_selection
    #   /plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
    
    plot_confusion_matrix(
        test_labels, 
        pred_labels, 
        classes=classes_dict, 
        # normalize=normalize,
        title=type
    )
}

def load(ds_path):
    """ Load from the training/testing dataset.

    Parameters
    ----------
    ds_path: path to the training/testing dataset.
             e.g., sift/train or sift/test 
    
    Returns
    -------
    image_paths: a (n_sample, 1) array that contains the paths to the descriptors. 
    labels: class labels corresponding to each image
    """
    # Grab a list of paths that matches the pathname
    files = glob.glob(os.path.join(ds_path, "*", "*.txt"))
    n_files = len(files)
    image_paths = np.asarray(files)
 
    # Get class labels
    classes = glob.glob(os.path.join(ds_path, "*"))
    labels = np.zeros(n_files)

    for i, path in enumerate(image_paths):
        folder, fn = os.path.split(path)
        labels[i] = np.argwhere(np.core.defchararray.equal(classes, folder))[0,0]

    # Randomize the order
    idx = np.random.choice(n_files, size=n_files, replace=False)
    image_paths = image_paths[idx]
    labels = labels[idx]

    #save classes for histograms as global
    for i, c in enumerate(classes):
        classes_dict[i] = c.split('/')[2]

    return image_paths, labels


if __name__ == "__main__":
    paths, labels = load("sift/train")

    kmeans = build_vocabulary(paths, 200)
    # pickle.dump(kmeans, open("kmeans.pkl", "wb")) # dump kmeans onto disk 
    # kmeans = pickle.load(open("kmeans.pkl", "rb")) # load the kmeans from disk to a variable

    img_feats = get_bags_of_sifts(paths, kmeans)
    # pickle.dump(img_feats, open("img_feats.pkl", "wb")) # dump kmeans onto disk 
    # img_feats = pickle.load(open("img_feats.pkl", "rb")) # load the kmeans from disk to a variable

    plot_histograms(img_feats, labels)
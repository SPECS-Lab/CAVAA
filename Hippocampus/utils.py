import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, pearsonr
from scipy.signal import correlate2d
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AffinityPropagation
import skdim
from random import shuffle
from PIL import Image
import imagehash
import cmath
from sklearn.decomposition import PCA
import clip
from sklearn.manifold import TSNE, MDS
import umap
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from itertools import combinations
from tqdm import tqdm



def load_dataset(directory, task, file_format='.npy', load_position=True, position_filename='position.npy'):
    '''
    Loads all .npy or .jpg images and the pose data per image from a given directory.

    Args:
        directory (str): path to the images to be loaded.
        file_format (str): format of the images. Accepted formats are .npy and .jpg.
        load_position (bool): if True, it will also load the pose data.
        position_filename (str): name of the file with the pose data. The accepted format is .npy.

    Returns:
        images (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width) and normalized values [0,1].
        position (2D numpy array): pose data with (x,y) coordinates and angle (in degrees; [0,360]), wit shape (n_samples, 3).
    '''
    ## Load images.
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if filename.endswith(file_format) and filename != position_filename:
            filepath = os.path.join(directory, filename)
            if file_format == '.npy':
                image = np.load(filepath)
            elif file_format == '.jpg' or file_format == '.png':
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
    images = np.array(images)

    if np.max(images) > 1:   # normalize to [0,1] if values are RGB [0,255].
        images = images/255.

    ## Load position.  
    position = []
    if load_position:
        if position_filename in os.listdir(directory):
            position = np.load(directory+'/'+position_filename)
        else:
            pos_directory = directory+'/pos'
            for i, filename in enumerate(os.listdir(pos_directory)):
                filepath = os.path.join(pos_directory, filename)
                pos = np.load(filepath).tolist()
                position.append(pos)

    pos_cols = (0,2)
    if task == 'openArena':
        pos_cols = (0,1)

    position = np.array(position)[:, pos_cols]

    return images, position


def create_dataloader(dataset, batch_size=256, reshuffle_after_epoch=True):
    '''
    Creates a DataLoader for Pytorch to train the autoencoder with the image data converted to a tensor.

    Args:
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        batch_size (int; default=32): the size of the batch updates for the autoencoder training.

    Returns:
        DataLoader (Pytorch DataLoader): dataloader that is ready to be used for training an autoencoder.
    '''
    if dataset.shape[-1] == 3:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=reshuffle_after_epoch)


def train_autoencoder(model, train_loader, dataset, eval_functions=[], opt=optim.Adam, num_epochs=1000, 
                      learning_rate=1e-4, alpha=1e3, beta=0, L2_weight_decay=0, loss_threshold=None):
    '''
    Train an autoencoder and compute custom metrics during training.

    Args:
        model (torch.nn.Module): The autoencoder model.
        train_loader (DataLoader): DataLoader for training data.
        dataset (4D numpy array): image dataset with shape (n_samples, n_channels, n_pixels_height, n_pixels_width).
        eval_functions (list): A list of functions to evaluate the model's performance periodically.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        alpha, beta (float): Custom hyperparameters for loss regularization.
        L2_weight_decay (float): Weight decay for L2 regularization.
        loss_threshold (float, optional): If provided, training will stop early if the loss drops below this value.

    Returns:
        results (dict): A dictionary containing lists of metrics recorded during training including loss.
    '''
    optimizer = opt(model.parameters(), lr=learning_rate, weight_decay=L2_weight_decay)
    criterion = nn.MSELoss()
    model = model.to('cuda')
    results = {'loss': []}
    results.update({func.__name__: [] for func in eval_functions})

    for epoch in range(num_epochs):
        running_loss = 0.
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for inputs, _ in train_loader:
                inputs = inputs.to('cuda')
                loss = model.backward(optimizer=optimizer, criterion=criterion, x=inputs, y_true=inputs, alpha=alpha, beta=beta)
                running_loss += loss

                pbar.update(1)
                pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

        avg_loss = running_loss / len(train_loader)
        results['loss'].append(avg_loss)

        # Early stopping condition
        if loss_threshold is not None and avg_loss < loss_threshold:
            print(f"Stopping early at epoch {epoch+1} as loss {avg_loss:.4f} is below the threshold {loss_threshold:.4f}.")
            break

        # Evaluate the model with each function in eval_functions.
        if len(eval_functions) > 0:
            model.eval()
            with torch.no_grad():
                embeddings = get_latent_vectors(dataset=dataset, model=model)
                for func in eval_functions:
                    result = func(embeddings=embeddings)
                    results[func.__name__].append(result)

    return results


def predict(image, model):
    '''
    Returns the output of model(image), and reshapes it to be compatible with plotting funtions such as plt.imshow().

    Args:
        image (3D numpy array): sample image with shape (n_channels, n_pixels_height, n_pixels_width).
        model (Pytorch Module): convolutional autoencoder that is prepared to process images such as 'image'.

    Returns:
        output_img (3D numpy array): output image with shape (n_pixels_height, n_pixels_width, n_channels)
    '''
    if image.shape[-1] <= 4:
        image = np.transpose(image, (2,0,1))
    n_channels, n_pixels_height, n_pixels_width = image.shape
    image = np.reshape(image, (1, n_channels, n_pixels_height, n_pixels_width))
    image = torch.from_numpy(image).float().to(next(model.parameters()).device)
    output_img = model(image)[0].detach().cpu().numpy()
    output_img = np.reshape(output_img, (n_channels, n_pixels_height, n_pixels_width))
    output_img = np.transpose(output_img, (1,2,0))
    return output_img


def get_predictions(model, train_loader, loss_criterion=nn.MSELoss()):
    '''
    Computes the average loss function of the model over the dataset contained in train_loader.

    Args:
        model (torch.nn.Module): The autoencoder model.
        train_loader (DataLoader): DataLoader containing the samples used for training.
        loss_criterion (torch.nn.Loss): loss function used to compute the score.

    Returns:
        average_loss (float): returns the average loss over the whole dataset.
    '''
    model.eval()

    criterion = loss_criterion

    all_preds = []
    with torch.no_grad():
        total_loss = 0.
        for inputs, _ in train_loader:
            inputs = inputs.to('cuda')

            predictions = model(inputs)[0]
            all_preds.append(predictions.cpu().numpy())

            loss = criterion(predictions, inputs)
            total_loss += loss.item() * inputs.size(0)

    all_preds = np.concatenate(all_preds)
    average_loss = total_loss / len(train_loader.dataset)

    return all_preds, average_loss


def get_latent_vectors(dataset, model, batch_size=256):
    '''
    Returns the latent activation vectors of the autoencoder model after passing all the images in the dataset.

    Args:
        dataset (numpy array): image dataset with shape 
        model (Pytorch Module): convolutional autoencoder that is prepared to process the images in dataset.

    Returns:
        latent_vectors (2D numpy array): latent activation vectors, matrix with shape (n_samples, n_hidden), where n_hidden is the number of units in the hidden layer.
    '''
    if dataset.shape[-1] <= 4:
        dataset = np.transpose(dataset, (0,3,1,2))
    tensor_dataset = TensorDataset(torch.from_numpy(dataset).float(), torch.from_numpy(dataset).float())
    data_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    latent_vectors = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, _ = batch
            latent = model(inputs.to('cuda'))[1]
            latent_vectors.append(latent.cpu().numpy())
    latent_vectors = np.concatenate(latent_vectors)
    return latent_vectors


def extract_feature_images(model, embeddings, clamping_value='max_unit', input_dims=[84,84,3]):
    '''
    TO DO.
    '''
    indxs_active = np.arange(embeddings.shape[1])[np.any(embeddings, axis=0)]
    images = []

    # Precompute the clamp_value for 'mean' mode to be used for all active units
    if clamping_value == 'mean':
        activations = torch.tensor(embeddings)
        max_values = torch.max(activations, dim=0)[0]
        clamp_value = torch.mean(max_values).item()  # Get a Python float for consistent use in all units

    for i in np.arange(model.n_hidden):
        if i in indxs_active:
            input_ = torch.zeros(model.n_hidden).to('cuda')
            if clamping_value == 'max_unit':
                activations = torch.tensor(embeddings[:,i])
                clamp_value = torch.max(activations).item()  # Get a Python float for this specific unit

            input_[i] = clamp_value
            img = np.transpose(model.decoder(input_)[0].detach().cpu().numpy(), (1, 2, 0))
            images.append(img)
        else:
            img = np.zeros(input_dims)
            images.append(img)
    images = np.array(images)
    
    return images


def shuffle_2D_matrix(m):
    '''
    Shuffles a matrix across both axis (not only the first axis like numpy.permutation() or random.shuffle()).

    Args:
        m (2D numpy array): 2D matrix with arbitrary values.

    Returns:
        m_shuffled (2D numpy array): the original matrix 'm', with all the elements shuffled randomly.
    '''
    N = m.size
    ind_shuffled = np.arange(N)
    shuffle(ind_shuffled)
    ind_shuffled = ind_shuffled.reshape((m.shape[0], m.shape[1]))
    ind_x = (ind_shuffled/m.shape[1]).astype(np.int_)
    ind_y = (ind_shuffled%m.shape[1]).astype(np.int_)
    m_shuffled = m[ind_x, ind_y]
    return m_shuffled


def ratemap_filtered_Gaussian(ratemap, std=2):
    '''
    Adds Gaussians filters to a ratemap in order to make it more spatially smooth.

    Args:
        ratemap (2D numpy array): unfiltered ratemap with the activity counts across space.
        std (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units). 

    Returns:
        new_ratemap (2D numpy array): original ratemap filtered with Gaussian smoothing.
    '''
    new_ratemap = gaussian_filter(ratemap, std)   
    return new_ratemap


def generate_ratemaps(embeddings, position, n_bins=50, filter_width=3, n_bins_padding=0):
    '''
    Creates smooth ratemaps from latent embeddings (activity) and spatial position through time.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        n_bins_padding (int; default=0): the number of extra pixels with 0 value that are added to every side of the arena.

    Returns:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    # Build occupancy map
    occupancy_map = generate_occupancy_map(position, n_bins=n_bins, filter_width=0, n_bins_padding=0, norm=False)

    # Add activation values to each cell in the ratemap and adds Gaussian smoothing.
    n_latent = embeddings.shape[1]
    ratemaps = np.zeros((n_latent, int(n_bins+2*n_bins_padding), int(n_bins+2*n_bins_padding)))
    for i in range(n_latent):
        ratemap_ = np.zeros((n_bins, n_bins))
        for ii, c in enumerate(embeddings[:,i]):
            indx_x = pos_imgs_norm[ii,0]
            indx_y = pos_imgs_norm[ii,1]
            ratemap_[indx_x, indx_y] += c

        if len(occupancy_map) > 0:
            ratemap_ = np.divide(ratemap_, occupancy_map, out=np.zeros_like(ratemap_), where=occupancy_map!=0)
            #ratemap_ = ratemap_/occupancy_map

        ratemaps[i] = np.pad(ratemap_, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)
        if np.any(ratemaps[i]):
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemap_filtered_Gaussian(ratemaps[i], filter_width)
            ratemaps[i] = ratemaps[i]/np.max(ratemaps[i])
            ratemaps[i] = ratemaps[i].T
        
    return ratemaps


def stats_place_fields(ratemaps, peak_as_centroid=True, min_pix_cluster=0.03, max_pix_cluster=0.5, active_threshold=0.2):
    '''
    Runs a simple clustering algorithm to identify place fields, and compute their number, centroids, and sizes, for all ratemaps.

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).
        peak_as_centroid (bool; default=True): if True, the centroid will be taken as the peak of the place field; if False, it will take the 'center of mass'.
        min_pix_cluster (bool; default=0.02): minimum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        max_pix_cluster (bool; default=0.5): maximum proportion of the total pixels that need to be active within a region to be considered a place field, with a range [0,1].
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].

    Returns:
        all_num_fields (1D numpy array): array with the number of place fields per embedding unit, with shape (n_latent,).
        all_centroids (2D numpy array): array with (x,y) position of all place field centroids, with shape (total_n_place_fields, 2).
        all_sizes (1D numpy array): array with the sizes of all place fields across embedding units, with shape (total_n_place_fields,).
    '''
    all_num_fields = []
    all_centroids = []
    all_sizes = []
    for r in ratemaps:

        ratemap = r.copy()
        
        ## Params.
        total_area = ratemap.shape[0]*ratemaps.shape[1]
        cluster_min = total_area*min_pix_cluster  #50
        cluster_max = total_area*max_pix_cluster  #1250
        
        ## Clustering.
        ratemap[ratemap <  ratemap.max()*active_threshold] = 0
        ratemap[ratemap >= ratemap.max()*active_threshold] = 1

        # First pass of clustering.
        clustered_matrix = np.zeros_like(ratemap)
        current_cluster = 1

        # Go through every bin in the ratemap.
        for yy in range(1,ratemap.shape[0]-1):
            for xx in range(1,ratemap.shape[1]-1):
                if ratemap[  yy, xx ] == 1:
                    # Go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clustered_matrix[ yy+ty, xx+tx ] != 0:
                                clustered_matrix[ yy,xx ] = clustered_matrix[ yy+ty, xx+tx ]

                    if clustered_matrix[ yy, xx ] == 0:
                        clustered_matrix[ yy, xx ] = current_cluster
                        current_cluster += 1
                        
        # Refine clustering: neighbour bins to same cluster number.
        for yy in range(1,clustered_matrix.shape[0]-1):
            for xx in range(1,clustered_matrix.shape[1]-1):
                if clustered_matrix[  yy, xx ] != 0:
                    # go through every bin around this bin.
                    for ty in range(-1,2):
                        for tx in range(-1,2):
                            if clustered_matrix[ yy+ty, xx+tx ] != 0:
                                if clustered_matrix[ yy+ty, xx+tx ] != clustered_matrix[  yy, xx ]:
                                    clustered_matrix[ yy+ty, xx+tx ] = clustered_matrix[  yy, xx ]
                  
        ## Quantify number of place fields.
        clusters_labels = np.delete(np.unique(clustered_matrix), np.where(  np.unique(clustered_matrix) == 0 ) )
        n_place_fields_counter = 0
        clustered_matrix_ = np.copy(clustered_matrix)
        clusters_labels_ = np.copy(clusters_labels)
        for k in range(clusters_labels.size):
            n_bins = np.where(clustered_matrix == clusters_labels[k])[0].size
            if cluster_min <= n_bins <= cluster_max:
                n_place_fields_counter += 1
            else:
                clustered_matrix_[np.where(clustered_matrix_==clusters_labels[k])] = 0
                clusters_labels_ = np.delete(clusters_labels_, np.where(clusters_labels_ == clusters_labels[k]) )

        all_num_fields.append(n_place_fields_counter)
        
        ## Compute centroids.
        centroids = []
        for k in clusters_labels_:
            if peak_as_centroid:  # compute centroid as the peak of the place field.
                x, y = np.unravel_index(np.argmax( r * (clustered_matrix_==k) ), r.shape)
            else:                 # compute the centroid as weighted sum ('center of mass').
                w_x = r[np.where(clustered_matrix_==k)[0], :].sum(axis=1)
                w_x = w_x/w_x.sum()
                x = np.sum(w_x * np.where(clustered_matrix_==k)[0])
                
                w_y = r[:, np.where(clustered_matrix_==k)[1]].sum(axis=0)
                w_y = w_y/w_y.sum()
                y = np.sum(w_y * np.where(clustered_matrix_==k)[1])
            centroids.append([y,x])

        all_centroids += centroids
        
        ## Compute sizes of place fields.
        sizes = []
        for k in clusters_labels_:
            n_bins = np.where(clustered_matrix_ == k)[0].size
            sizes.append(n_bins/total_area)

        all_sizes += sizes
    
    return np.array(all_num_fields), np.array(all_centroids), np.array(all_sizes)


def clean_embeddings(embeddings, normalize=False):
    '''
    Takes only the embeddings that are active an any given point, and, if normalize=True, normalizes the values.

    Args:
        embeddings (2D numpy array): 2D matrix latent embeddings through time, with shape (n_samples, n_latent).
        normalize (bool; default=False): if True, the embedding activation values will be normalized to [-1,1], per each unit.

    Returns:
        embeddings_clean (2D numpy array): original embeddings matrix, with the silent units removed, with shape (n_samples, n_active).
    '''
    indxs_active = np.any(embeddings, axis=0)
    n_active = np.sum(indxs_active)
    embeddings_clean = embeddings[:,indxs_active]

    if normalize:  # normalize between [-1, 1].
        n_samples = embeddings.shape[0]
        maxs = np.tile( np.abs(embeddings_clean).max(axis=0), n_samples).reshape((n_samples, n_active))
        embeddings_clean = embeddings_clean / maxs

    return embeddings_clean


def clean_ratemaps(ratemaps):
    '''
    Discards the ratemaps of the silent units.
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with shape (n_latent, n_bins, n_bins).

    Returns:
        ratemaps_clean (3D numpy array): 3D matrix containing the ratemaps associated to the embedding units that are not silent, with shape (n_latent-n_silent, n_bins, n_bins).
    '''
    indxs_active = np.any(ratemaps, axis=(1,2))
    ratemaps_clean = ratemaps[indxs_active]

    return ratemaps_clean


def angular_distance(angle1, angle2):
    '''
    Computes the angular distance between two angles given in radians.

    Args:
        angle1, angle2 (float): angles to compare (in radians).

    Returns:
        dist (float): distance value between angle1 and angle2.
    '''
    delta = np.abs(angle1 - angle2)
    dist = np.min([delta, 2*np.pi - delta])

    return dist


def euclidean_distance(point1, point2):
    '''
    Computes the Euclidean distance (L2 norm) between two points in space.

    Args:
        point1, point2 (1D numpy array): array representing a point in am arbitrary N-dimensional space.

    Returns:
        dist (float): Euclidean distance between point1 and point2 in the N-dimensional space.
    '''
    dist = np.sqrt(np.sum((point1 - point2)**2))

    return dist


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1, vec2 (1D numpy array): Vectors.

    Returns:
        similarity (float): Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return similarity


def generate_occupancy_map(position, n_bins=50, filter_width=0, n_bins_padding=0, norm=True):
    '''
    Computes the occupancy map based on the position through time.

    Args:
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.
        filter_width (float; default=2): standard deviation of the Gaussian filter to be applied (in 'pixel' or bin units).
        padding_n (int; default=0): the number of extra pixels that are added to every side of the arena.

    Returns:
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).
    '''
    # Normalize position with respect to grid resolution to convert position to ratemap indices.
    pos_imgs_norm = np.copy(position)

    if np.min(pos_imgs_norm[:,0]) < 0:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] + np.abs(np.min(pos_imgs_norm[:,0]))
    else:
        pos_imgs_norm[:,0] = pos_imgs_norm[:,0] - np.min(pos_imgs_norm[:,0])

    if np.min(pos_imgs_norm[:,1]) < 0:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] + np.abs(np.min(pos_imgs_norm[:,1]))
    else:
        pos_imgs_norm[:,1] = pos_imgs_norm[:,1] - np.min(pos_imgs_norm[:,1])

    max_ = np.max(pos_imgs_norm)
    pos_imgs_norm[:,0] = pos_imgs_norm[:,0]/max_
    pos_imgs_norm[:,1] = pos_imgs_norm[:,1]/max_

    pos_imgs_norm *= n_bins-1
    pos_imgs_norm = pos_imgs_norm.round(0).astype(int)

    map_occ = np.zeros((n_bins, n_bins))
    for p in pos_imgs_norm:
        ind_x, ind_y = p
        map_occ[ind_x, ind_y] += 1

    map_occ = np.pad(map_occ, ((n_bins_padding, n_bins_padding), (n_bins_padding, n_bins_padding)), mode='constant', constant_values=0)

    map_occ = ratemap_filtered_Gaussian(map_occ, filter_width)

    if norm:
        map_occ = map_occ/np.sum(map_occ, axis=(0,1))

    occupancy_map = map_occ.T

    return occupancy_map


def spatial_information(ratemaps, occupancy_map):
    '''
    Spatial information score (SI) as computed in Skaggs et al. 1996. The SI is computed per rate (i.e., embedding unit).

    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        occupancy_map (2D numpy array): 2D matrix reflecting the occupancy time across the space, with shape (n_bins, n_bins).

    Returns:
        SI (1D numpy array): array with SI scores, in bit/spike, with shape (n_latent,).
    '''
    ratemaps_ = ratemaps[np.any(ratemaps, axis=(1,2))]
    FR = ratemaps_/(np.mean(ratemaps_, axis=(1,2))[:,np.newaxis,np.newaxis])
    OT = occupancy_map/np.sum(occupancy_map)
    log_FR = np.log2(FR, out=np.zeros_like(FR, dtype='float64'), where=(FR!=0))
    SI = np.sum(FR*OT*log_FR, axis=(1,2))
    
    return SI


def population_sparseness(ratemaps, active_threshold=0.2):
    '''
    Estimates the population sparseness as the expected number of active units per pixel (i.e., location in space).
    
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).
        active_threshold (float; default=0.2): percentage over the maximum activity from which pixels are considered to be active, otherwise they become 0; within a range [0,1].
    
    Returns:
        sparseness (float): population sparseness score as 1 minus the average proportion of active units across the environment, within the range [0,1].
    '''
    #ratemaps_thres = np.copy(ratemaps)
    ratemaps_thres = clean_ratemaps(ratemaps)
    ratemaps_thres[ratemaps_thres<active_threshold] = 0
    ratemaps_thres[ratemaps_thres>=active_threshold] = 1

    prop_active_per_pixel = np.mean(ratemaps_thres, axis=0)
    sparseness = 1 - np.mean(prop_active_per_pixel)

    return sparseness


def linear_decoding_score(embeddings, features, n_baseline=10000):
    '''
    Computes the score of linear regression of embeddings --> features. Features will normally be position (x,y) 
    or orientation (radians or in vectorial form).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        n_baseline (int; default=10000): number of permutation tests (i.e., shuffling the embeddings matrix) to compute the baseline.

    Returns:
        scores (float list): a list with two scores: (1) the evaluation of the linear regression, and (2) an average & std 
                             of n_baseline random permutation tests.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    linear_score = linear_model.score(embeddings, features)

    #baseline
    baselines = []
    for i in range(n_baseline):
        embeddings_shuffled = shuffle_2D_matrix(np.copy(embeddings))
        linear_model_baseline = LinearRegression()
        linear_model_baseline.fit(embeddings_shuffled, features)
        random_score = linear_model_baseline.score(embeddings_shuffled, features)
        baselines.append(random_score)

    baseline_score = [np.mean(baselines), np.std(baselines)]

    ratio = linear_score/(baseline_score[0])

    return linear_score, baseline_score, ratio


def linear_decoding_error(embeddings, features, norm=1):
    '''
    Computes the expected error of a linear decoder that uses the embeddings to predicts features (e.g. position in (x,y)).

    Args:
        embeddings (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        features (2D numpy array): 2D matrix containing the dependent variable, with shape (n_samples, n_features).
        norm (float; default=1): value used to normalize the MSE and bring it to a more convenient scale.

    Returns:
        mean_dist (float): average euclidean distance between the predictions of the decoder and the actual features, normalized by a scalar.
    '''
    linear_model = LinearRegression()
    linear_model.fit(embeddings, features)
    pred = linear_model.predict(embeddings)

    dist = np.sqrt(np.sum((pred - features)**2, axis=1))
    mean_dist = np.mean(dist) / norm

    return mean_dist


def autocorrelation_2d(ratemaps):
    '''
     Generates the autocorrelation matrices of the 2D ratemaps.
    
    Args:
        ratemaps (3D numpy array): 3D matrix containing the ratemaps associated to all embedding units, with 
                                   shape (n_latent, n_bins, n_bins).

    Returns:
        autocorr (3D numpy array): 3D matrix containing the 2D autocorrelation associated to the ratemaps, with 
                                   shape (n_latent, n_bins, n_bins).
    '''
    autocorr = []
    for i in range(ratemaps.shape[0]):
        autocorr_map = correlate2d(ratemaps[i], ratemaps[i], mode='same')
        autocorr_map /= ratemaps[i].size
        autocorr.append( autocorr_map )

    return autocorr


def pv_correlation(embeddings1, embeddings2, position, n_bins=50):
    '''
    Computes the population vector (PV) correlation coefficient between two ratemaps (normally corresponding to different epochs).

    Args:
        embeddings1 (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        embeddings2 (2D numpy array): 2D matrix containing the independent variable, with shape (n_samples, n_latent).
        position (2D numpy array): 2D matrix containing the (x,y) spatial position through time, with shape (n_samples, 2).
        n_bins (int; default=50): resolution of the (x,y) discretization of space from which the ratemaps will be computed.

    Returns:
        pv_corr (float): average correlation coefficient across spatial locations between the ratemaps corresponding to embeddings 1 and 2.
    '''
    n_total_bins = int(n_bins**2)
    ratemaps1 = ratemaps(embeddings1, position, n_bins=n_bins, filter_width=0, occupancy_map=[], n_bins_padding=0)
    ratemaps1 = np.reshape(ratemaps1, (ratemaps1.shape[0], n_total_bins))

    ratemaps2 = ratemaps(embeddings2, position, n_bins=n_bins, filter_width=0, occupancy_map=[], n_bins_padding=0)
    ratemaps2 = np.reshape(ratemaps2, (ratemaps2.shape[0], n_total_bins))

    corr_coefs = []
    for i in range(n_total_bins):
        corr_coef = pearsonr(ratemaps1[:,i], ratemaps2[:,i])[0]
        corr_coefs.append(corr_coef)

    pv_corr = np.nanmean(corr_coefs).round(3)

    return pv_corr

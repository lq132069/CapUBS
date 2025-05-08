import torch
import math
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def spectral_angle_distance(label, pred):
    # Flatten the images into tensors of shape (H*W, C)
    label_flat = label.view(-1, label.shape[-1])
    pred_flat = pred.view(-1, pred.shape[-1])

    # Normalize the hyperspectral vectors
    label_normalized = F.normalize(label_flat, dim=1)
    pred_normalized = F.normalize(pred_flat, dim=1)

    # Calculate the cosine of the spectral angle
    cos_similarity = torch.bmm(label_normalized.unsqueeze(1), pred_normalized.unsqueeze(2)).squeeze(1)

    # Calculate the spectral angle distance
    angle_distance = torch.acos(torch.clamp(cos_similarity, -1.0, 1.0))

    # Return the mean spectral angle distance
    return angle_distance.mean()


def SALoss(image1, image2):
    total_loss = 0.0

    for i in range(image1.shape[0]):
        # Get the current pair of hyperspectral images
        image_1 = image1[i, :, :, :]  # Replace with the method to get the i-th image pair
        image_2 = image2[i, :, :, :]  # Replace with the method to get the i-th image pair

        # Calculate the Spectral Angle Distance loss
        loss = spectral_angle_distance(image_1, image_2)

        # Accumulate the loss
        total_loss += loss
    return total_loss

def entropyLoss(matrix):
    # Maximize information entropy
    probs = F.softmax(matrix, dim=1)  # Calculate probability distribution
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # Avoid log(0)
    return entropy.mean()


def sparsityLoss(matrix):
    # L1 regularization for sparsity
    # return torch.norm(matrix, p=1)
    return torch.sum(torch.abs(matrix))


def norm_data(data):
    norm_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        input_max = np.max(data[i, :, :])
        input_min = np.min(data[i, :, :])
        norm_data[i, :, :] = (data[i, :, :] - input_min) / (input_max - input_min) # [0,1]
    return norm_data


def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def show_band_attention_train(band_attention, epoch):
    band_attention_np = band_attention.detach().numpy()
    np.save(f'res/band_selection/band_attention_epoch_{epoch}.npy', band_attention_np)
    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.plot(band_attention_np.flatten(), marker='o')  # Use flatten() to make it one-dimensional
    plt.title(f'Band Attention Epoch-{epoch}')
    plt.xlabel('Index')
    plt.ylabel('Attention Value')
    plt.grid()
    plt.savefig(f'res/band_selection/band_attention_epoch_{epoch}.jpg', format='jpg')  # Specify format as jpg
    # plt.show()


def show_band_attention_test(band_attention, dataset_name):
    band_attention_np = band_attention.detach().numpy()
    np.save('res/band_selection/band_attention_net_test.npy', band_attention_np)
    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.plot(band_attention_np.flatten(), marker='o')  # Use flatten() to make it one-dimensional
    plt.title(f'Band Attention Net Test')
    plt.xlabel('Index')
    plt.ylabel('Attention Value')
    plt.grid()
    plt.savefig(f'res/band_selection/band_attention_net_{dataset_name}.jpg', format='jpg')  # Specify format as jpg
    # plt.show()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fig_show(fig):
    # Plot the graph
    plt.figure()
    plt.imshow(fig)  # Use flatten() to make it one-dimensional
    plt.show()

def select_important_bands(data, band_importance, n, dataset_name):
    data = data.permute(1, 2, 0)
    # 1. Sort the band coefficients
    sorted_indices = torch.argsort(band_importance.squeeze()).tolist()  # Sort and convert to list
    num_bands = band_importance.shape[0]

    # 2. Select the highest-ranked band
    selected_bands = [sorted_indices[-1]]  # Select the most important band index

    # 3. Calculate the distances between bands
    distances = torch.zeros((num_bands, num_bands))  # Initialize the distance matrix
    for i in range(num_bands):
        for j in range(num_bands):
            if i != j:
                distances[i, j] = torch.mean(torch.abs(data[:, :, i] - data[:, :, j]))  # Calculate the mean absolute distance

    # 4. Calculate the distance factor
    distance_factors = torch.zeros(num_bands)
    for i in range(num_bands):
        if i == selected_bands[0]:  # For the most important band, set the distance factor to infinity
            distance_factors[i] = float('inf')
        else:
            # Calculate the average of the distance values with bands more important than it
            important_indices = [idx for idx in sorted_indices[:-1] if band_importance[idx] > band_importance[i]]
            if important_indices:
                # distance_factors[i] = torch.mean(distances[i, important_indices])
                distance_factors[i] = torch.min(distances[i, important_indices])
            else:
                distance_factors[i] = 0  # If there are no more important bands, set to 0
    # Find the maximum value, excluding infinity
    max_value = distance_factors[distance_factors != float('inf')].max()
    # Replace infinity with the maximum value
    distance_factors[distance_factors == float('inf')] = max_value

    # Find the minimum and maximum values
    min_value = distance_factors.min()
    max_value = distance_factors.max()

    # Normalize
    distance_factors = (distance_factors - min_value) / (max_value - min_value)

    # 5. Calculate the combined selection factor value
    combined_scores = band_importance.squeeze(1) * distance_factors.unsqueeze(1)

    # Get the indices of the top n maximum values
    top_n_indices = torch.topk(combined_scores.squeeze(), n).indices
    selected_bands = []
    selected_bands.extend(top_n_indices.tolist())  # Add the newly selected bands to selected_bands

    # 7. Generate the mask
    mask = torch.zeros(num_bands, dtype=torch.int)
    mask[selected_bands] = 1  # Set the positions of the selected bands to 1
    mask = mask.unsqueeze(1).unsqueeze(-1)  # Adjust the mask shape to fit the data


    combined_scores = combined_scores.squeeze().numpy()
    band_importance = band_importance.squeeze().numpy()
    # Plot combined_scores
    plt.figure(figsize=(10,6))

    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # Chinese font, prioritize KaiTi, then SimHei, then FangSong
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 14  # Font size
    plt.rcParams['axes.unicode_minus'] = False  # Display negative sign normally
    plt.rc('font', family='Times New Roman')
    bwith = 1.5  # Border width setting
    fig, ax = plt.subplots()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # Then plot band_importance and combined_scores
    plt.plot(band_importance, marker='.', label='Original', color='black', linewidth=2)
    plt.plot(combined_scores, marker='.', label='Advanced', color='blue', alpha=0.8, linewidth=2)  # Set transparency
    plt.xlabel('Band Index', fontsize=14)
    plt.ylabel('Score', fontsize=14)

    # Set x-axis ticks to display every 15 positions
    xticks = range(0, len(combined_scores), 15)

    # Display the labels at these positions as twice their original value
    xticklabels = [str(i * 2) for i in xticks]
    plt.xticks(xticks, xticklabels)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the image in .jpg format, using data_name as prefix
    plt.savefig(f'res/band_selection/Band_Combined_Scores_final_{dataset_name}.jpg', format='jpg', dpi=300, bbox_inches='tight')


    return selected_bands, combined_scores, mask


def select_data(mask, data, target):
    # mask = (band_attention_test > t).float()  # Generate mask torch.Size([50, 1, 1])
    # Expand the mask to match the shape of data and target
    mask_expanded = mask.squeeze(1).unsqueeze(-1)  # Adjust mask shape to fit data
    # Convert data and target to float32
    # Filter data and target using the mask
    data_band = data * mask_expanded  # Perform element-wise multiplication
    target_band = target * mask.squeeze(1).squeeze(1)  # Directly filter target

    # Convert the mask to boolean
    mask_boolean = (mask.squeeze(1).squeeze(1) > 0).bool()  # Become a one-dimensional boolean array

    # Keep only the parts where the mask is 1
    data_band_filtered = data_band[mask_boolean]  # Filter according to the mask
    target_band_filtered = target_band[mask_boolean]  # Filter according to the mask

    # Convert tensors to NumPy arrays
    data_band = data_band_filtered.detach().numpy()  # Keep only non-zero parts
    target_band = target_band_filtered.detach().numpy()  # Keep only non-zero parts

    return data_band, target_band


def plot_bandSelection_spectral_curve(data, label, band_selection, dataset_name, num_select):
    # Get the number of bands and number of classes
    unique_labels = np.unique(label)

    # Initialize storage for the spectral curves of each class
    average_spectra = {}

    # Calculate the average spectrum for each class
    for cls in unique_labels:
        if cls > 0:  # Assuming class labels start from 1, 0 represents background
            mask = (label == cls)
            average_spectra[cls] = np.mean(data[mask], axis=0)

    # Plot the spectral curves
    plt.figure(figsize=(10,6))

    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # Chinese font, prioritize KaiTi, then SimHei, then FangSong
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['font.size'] = 14  # Font size
    plt.rcParams['axes.unicode_minus'] = False  # Display negative sign normally
    plt.rc('font', family='Times New Roman')
    bwith = 1.5  # Border width setting
    fig, ax = plt.subplots()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    for cls, spectrum in average_spectra.items():
        plt.plot(spectrum, label=f'Class {cls}', linewidth=2)

    # Add vertical dashed lines to indicate selected bands
    for band in band_selection:
        plt.axvline(x=band, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Band Index', fontsize=14)
    plt.ylabel('Reflectance', fontsize=14)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))  # Add legend
    plt.grid(True)

    save_path = f'res/band_selection/band_selection_spectral_curve_{dataset_name}_band_num_{num_select}.jpg'
    plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight', pad_inches=0.1)  # Specify format as jpg
    # plt.show()
    plt.close()

import torch.multiprocessing as mp

def process_patch(network, patch, var_patch):
    with torch.no_grad():
        band_attention, _ = network(patch, var_patch)
    return band_attention


def test(network, test_data_loader, optim_model):
    network.load_state_dict(torch.load(optim_model))
    network.eval()

    total_band_attention = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # test_data_loader returns data and var_data
        patches = [(data.cpu(), var_data.cpu()) for (data, var_data) in test_data_loader]

        # Print data loading status
        print(f"Number of patches: {len(patches)}")
        # Use starmap to pass multiple arguments
        results = pool.starmap(process_patch, [(network, patch[0], patch[1]) for patch in patches])

    # Extend the results to total_band_attention
    total_band_attention.extend(results)
    # Average the band attention across all patches
    band_attention = torch.mean(torch.cat(total_band_attention), dim=0).reshape(1, -1)

    return band_attention
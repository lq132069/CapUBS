import torch.optim as optim
from torch.utils.data import DataLoader
from dataset.dataset import *
from model.framework import CapUBS
from utils import *
from tqdm import *
from spectral import *
import time
import scipy.io
from utils import plot_bandSelection_spectral_curve, test

def train(network, train_data_loader, num_epochs, learning_rate, save_model):

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    best_loss = float('inf')
    network.train()
    for epoch in tqdm(range(num_epochs), file=sys.stdout):
        total_loss = 0
        total_loss_rec = 0
        total_loss_bs = 0

        for (data,var_data) in train_data_loader:

            data = data.cpu()
            var_data = var_data.cpu()

            optimizer.zero_grad()

            band_attention, data_rec = network(data, var_data)
            loss, Loss_rec, Loss_bs = network.loss(band_attention, data, data_rec)
            loss.backward()
            total_loss += loss.item()
            total_loss_rec += Loss_rec.item()
            total_loss_bs += Loss_bs.item()

            optimizer.step()

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(network.state_dict(), save_model)

        if epoch % 1 == 0:
            tqdm.write('Epoch: {0}, Loss: {1}, Best_Loss: {2}, Loss_rec: {3}, Loss_bs: {4}'
                       .format(epoch, total_loss, best_loss, total_loss_rec, total_loss_bs))

        band_attention = torch.mean(band_attention, dim=0).reshape(1, -1)  # 形状将变为 [31, 1, 1]
        if epoch % 5 == 0:
            show_band_attention_train(band_attention, epoch)

    return band_attention




def create_data_loader(data, neighbor, overlap, batch_size, num_workers=1):

    dataset_train = Patch_data_loader_train(data, neighbor=neighbor, overlap=overlap, ratio=0.25)
    train_data_loader = DataLoader(dataset_train, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, drop_last=True)


    dataset_test = Patch_data_loader_train(data, neighbor=neighbor, overlap=overlap, ratio=0.5)
    test_data_loader = DataLoader(dataset_test, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=True)

    return train_data_loader, test_data_loader


def band_selection(network, save_model, test_data_loader, data, dataset_name):

    band_attention_test = test(network, test_data_loader, save_model)
    show_band_attention_test(band_attention_test, dataset_name)
    band_attention_test = band_attention_test.squeeze(0).unsqueeze(1).unsqueeze(2)

    return band_attention_test

if __name__ == '__main__':

    """
    CapUBS
    """
    time0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    or_data = scipy.io.loadmat("data/PaviaU.mat")['paviaU']
    label = scipy.io.loadmat("data/PaviaU_gt.mat")['paviaU_gt']
    data = or_data[0:30,0:30,::2]
    data = np.transpose(data, (2,0,1))

    img_channels = data.shape[0]
    img_width = data.shape[1]
    img_height = data.shape[2]

    # framework setting
    neighbor = 10
    overlap = 8
    num_workers = 1
    num_epochs = 100
    learning_rate = 0.0001
    batch_size = 16

    # capsule network setting
    cube_channels = img_channels
    cube_height = neighbor
    cube_width = neighbor
    num_output_units = cube_channels
    output_unit_size = 16
    num_iterations = 3

    train_data_loader, test_data_loader = create_data_loader(data, neighbor, overlap, batch_size, num_workers)

    network = CapUBS(image_width=cube_width,
                            image_height=cube_height,
                            image_channels=cube_channels,
                            num_output_units=num_output_units,
                            output_unit_size=output_unit_size,
                            num_iterations=num_iterations
                            ).to(device)

    initialize_weights(network)

    # train
    dataset_name = 'pavia'
    save_model = f'res/optim_model/optim_model_{dataset_name}.pth'
    time_train_start = time.time()
    band_attention = train(network, train_data_loader, num_epochs, learning_rate, save_model)
    time_train_over = time.time()

    # test
    time_test_start = time.time()
    band_attention_test = band_selection(network, save_model, test_data_loader, data,
                                         dataset_name=dataset_name)
    time_test_over = time.time()

    band_num_sets = list(range(5, 30, 2))
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data.astype(np.float32))
    for band_num in band_num_sets:

        print(f'band num is {band_num}')

        selected_bands, combined_scores, band_mask = select_important_bands(data, band_attention_test, band_num,
                                                                            dataset_name)
        selected_bands = [x * 2 for x in selected_bands]


        np.save(f'res/band_selection/our_{dataset_name}_band_num_{band_num}.npy', selected_bands)
        print(f'selected bands: {selected_bands}')

        plot_bandSelection_spectral_curve(or_data, label, selected_bands, dataset_name=dataset_name, num_select=band_num)

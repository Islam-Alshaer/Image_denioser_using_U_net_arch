import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
from torchsummary import summary  # unused in training run; enable if you need model summary
from PIL import Image

def unpickle(file : str) -> np.ndarray:
    with open(file, 'rb') as fo:
        dt = pickle.load(fo, encoding='bytes')
    return dt

taken_images_n = 2000
def load_stanford_dogs(data_path: str) -> np.ndarray:
    global taken_images_n
    data = np.zeros(shape=(taken_images_n, 3, 128, 128), dtype=np.uint8)
    for i, file in enumerate(os.listdir(data_path)):
        if i == taken_images_n:
            break
        im = Image.open(os.path.join(os.getcwd(), data_path, file))
        im = im.convert('RGB')
        im = im.resize((128, 128))
        im_numpy = np.array(im)
        # reshape it to be good with tensor
        im_reshaped = np.transpose(im_numpy, (2, 0, 1))  # from (128, 128, 3) to (3, 128, 128)
        data[i] = im_reshaped
    return data

def load_noisy_data():
    noisy_data = np.ndarray(shape=(0, 3, 128, 128), dtype=np.uint8)
    noisy_data = np.vstack((noisy_data, unpickle('erling.pkl')))
    noisy_data = np.vstack((noisy_data, unpickle('gaussian.pkl')))
    noisy_data = np.vstack((noisy_data, unpickle('impulse.pkl')))
    noisy_data = np.vstack((noisy_data, unpickle('rayleigh.pkl')))
    noisy_data = np.vstack((noisy_data, unpickle('erling.pkl')))
    noisy_data = np.vstack((noisy_data, unpickle('uniform.pkl')))
    return noisy_data

def load_prepare_data(original_data_path: str, split_ratio: float = 0.8):
    global taken_images_n
    #load
    original_data = load_stanford_dogs(original_data_path) #90MB (uint8)
    noisy_data = load_noisy_data() #500MB (uint8)
    #reshape noisy data
    print(type(noisy_data), noisy_data.shape)
    print(type(original_data), original_data.shape)

    #split into train and test  (labels in this context means original images)
    split_size = int(len(noisy_data) * split_ratio)
    # take from each one of the 6 sections a portion
    portion_size = split_size // 6

    # Each noise file/block is assumed to be `taken_images_n` long and placed consecutively in noisy_data.
    # For block i (0-based) the range is [i*taken_images_n, (i+1)*taken_images_n).
    test_slices = []
    train_slices = []
    for i in range(6):
        start = i * taken_images_n
        test_slices.append(noisy_data[start: start + portion_size])
        train_slices.append(noisy_data[start + portion_size: start + taken_images_n])

    test_data = np.vstack(test_slices)
    del test_slices
    train_data = np.vstack(train_slices)
    del train_slices
    del noisy_data  #we don't need it any more


    # labels (original images) correspond to the first `taken_images_n` images
    test_labels = original_data[:portion_size]
    train_labels = original_data[portion_size:taken_images_n]
    del original_data #we don't need it anymore

    # Convert to tensors but keep them on CPU to avoid exhausting VRAM. Move batches to GPU during training.
    # keep dtype float32 on CPU; dividing by 255.0 converts to [0,1]
    train_data = torch.from_numpy(train_data).float() / 255.0
    test_data = torch.from_numpy(test_data).float() / 255.0
    train_labels = torch.from_numpy(np.array(train_labels)).float() / 255.0
    test_labels = torch.from_numpy(np.array(test_labels)).float() / 255.0

    # normalize (in-place on CPU to save GPU memory)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    train_data = (train_data - mean) / std
    train_labels = (train_labels - mean) / std
    test_data = (test_data - mean) / std
    test_labels = (test_labels - mean) / std

    print(train_data.shape, train_labels.shape)
    print(test_data.shape, test_labels.shape)
    # print(train_data[0, 0, :5, :5]) #print 10 pixels from the first image
    # print(test_data[0, 0, :5, :5]) #the same as above
    print(train_data.max(), train_data.min())
    print(train_labels.max(), train_labels.min())
    print(test_data.max(), test_data.min())
    print(test_labels.max(), test_labels.min())

    return train_data, train_labels, test_data, test_labels



class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        global taken_images_n
        image = self.data[idx]
        label = self.labels[idx % len(self.labels)] #since the data is just the same dataset repeated for each noise type
        return image, label



class DoubleConv(nn.Module):
    """
    A helper module that applies two convolutional layers,
    each followed by Batch Normalization and a ReLU activation.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # First Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second Convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Denoise_U_net(nn.Module):
    def __init__(self):
        super(Denoise_U_net, self).__init__()

        #===============ENCODER==================#
        self.enc1 = DoubleConv(3, 8) #from (3, 128, 128) to (8, 128, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #from (8, 128, 128) to (8, 64, 64)

        self.enc2 = DoubleConv(8, 16) #from (8, 64, 64) to (16, 64, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #from (16, 64, 64) to (16, 32, 32)

        self.enc3 = DoubleConv(16, 32) #from (16, 64, 64) to (32, 64, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) #from (32, 64, 64) to (32, 32, 32)

        #=====================BOTTLE NECK==========#
        self.bottleneck = DoubleConv(32, 64) #from (32, 32, 32) to (64, 16, 16)


        #=================DECODER==================#
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) #from (64, 16, 16) to (32, 32, 32) to upsample
        self.dec3 = DoubleConv(64, 32) #skip connection we concatinate enc3 (32 filter) with upconv3 (32 filter) = 64 fitler

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) #from (32, 32, 32) to (16, 64, 64) to upsample
        self.dec2 = DoubleConv(32, 16) #skip connection


        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2) #from (16, 64, 64) to (8, 128, 128) to upsample
        self.dec1 = DoubleConv(16, 8) #skip connection

        self.final_conv = nn.Conv2d(8, 3, kernel_size=1) #return to original shape RGB


    def forward(self, x):
        #===========ENCODER===============#
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        #===========BOTTELNECK=============#
        b = self.bottleneck(p3)

        #=========DECODER(with Skip Connections)========
        # Level 3
        u3 = self.upconv3(b)
        cat3 = torch.cat((u3, e3), dim=1)
        d3 = self.dec3(cat3)  # Blend with convolutions

        # Level 2
        u2 = self.upconv2(d3)
        cat2 = torch.cat((u2, e2), dim=1)
        d2 = self.dec2(cat2)  # Blend with convolutions

        # Level 1
        u1 = self.upconv1(d2)
        cat1 = torch.cat((u1, e1), dim=1)
        d1 = self.dec1(cat1)  # Blend with convolutions

        #============BACK TO ORIGINAL IMAGE================#
        out = self.final_conv(d1)  # (3, 32, 32)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")

    # limit resources (to prevent system crash) - lower fraction if you still see OOMs
    if device.type == 'cuda':
        try:
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
        except Exception:
            pass

    # Create datasets and loaders (datasets are kept on CPU; batches will be moved to GPU)
    X_test, y_test, X_train, y_train = load_prepare_data(original_data_path='Images')

    train_dataset = CIFARDataset(X_train, y_train)
    test_dataset = CIFARDataset(X_test, y_test)

    # Reduce batch size and enable pin_memory so transfers to GPU are faster/non-blocking.
    # If you still see OOM, reduce batch_size further (e.g., 4 or 2).
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = Denoise_U_net().to(device)

    #a dummy batch of 32 inputs
    # dummy_batchy_little_glatchy = torch.randn(32, 3, 128, 128).type(torch.float).to(device) #random numbers but from a normal distribution and so
    #
    # with torch.inference_mode():
    #     model.eval()
    #     output = model.forward(dummy_batchy_little_glatchy)
    #     print(output.shape)
    #     summary(model, (3, 128, 128))

    #====================training=====================#
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 20
    losses_log = []
    # Use mixed precision when running on CUDA to reduce memory and speed up training
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad() # set gradients to zero

            # Move minibatch to device (non_blocking=True when pin_memory=True in DataLoader)
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            # forward/backward with optional autocast and GradScaler
            with torch.amp.autocast('cuda', enabled=use_amp):
                y_pred = model(batch_X)
                loss = criterion(y_pred, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # print error for train
        avg_loss = epoch_loss / len(train_loader)
        print(f"average train loss for epoch {epoch + 1}/{epochs}: {avg_loss:.6f}")
        torch.save(model.state_dict(), f'examples/parameters_epoch{epoch+1}.pth')

        # print error for test
        with torch.inference_mode():
            model.eval()
            total_loss = 0
            for batch_X_test, batch_y_test in test_loader:
                batch_X_test = batch_X_test.to(device, non_blocking=True)
                batch_y_test = batch_y_test.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    y_pred_test = model(batch_X_test)
                    loss = criterion(y_pred_test, batch_y_test)
                total_loss += loss.item()

            avg_loss_test = total_loss / len(test_loader)
            print(f"average loss for test set: {avg_loss_test:.6f}")
            losses_log.append(avg_loss_test)

    #search the best test loss
    best_epoch = losses_log.index(min(losses_log))
    print(f"best epoch is: {best_epoch+1} (one-indexed)")
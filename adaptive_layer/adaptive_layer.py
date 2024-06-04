import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def RAI(fan_in, fan_out):
    V = np.random.randn(fan_out, fan_in + 1) * 0.6007 / fan_in ** 0.5
    for j in range(fan_out):
        k = np.random.randint(0, high=fan_in + 1)
        V[j, k] = np.random.beta(2, 1)
    W = V[:, :-1]
    b = V[:, -1]
    return W, b


class Conv_AE(nn.Module):
    def __init__(self, n_hidden=1000, weight_init='RAI'):
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (84,84,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
        '''
        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 10, 10

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

        if weight_init == 'RAI':
            self.apply_custom_initialization()

    def apply_custom_initialization(self):
        fan_in, fan_out = self.fc1.weight.data.size(1), self.fc1.weight.data.size(0)
        W, b = RAI(fan_in, fan_out)
        self.fc1.weight.data = torch.from_numpy(W).float()
        self.fc1.bias.data = torch.from_numpy(b).float()

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))  
        x = x.view(-1, 64 * self.dim1 * self.dim2)
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x)) 
        x = x.view(-1, 64, self.dim1, self.dim2) 
        x = F.relu(self.conv4(x)) 
        x = F.relu(self.conv5(x))  
        x = torch.sigmoid(self.conv6(x))  
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, alpha=0):
        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        batch_size, hidden_dim = hidden.shape
        ZT_Z = torch.mm(hidden.t(), hidden)
        I = torch.eye(hidden_dim, device='cuda')
        orthonormal_loss = alpha * torch.norm(I - ZT_Z) / (batch_size*hidden_dim) 
    
        loss = recon_loss + orthonormal_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()


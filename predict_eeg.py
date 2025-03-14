import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from get_data import EEGDataLoader

class EEGDataset(Dataset):
    def __init__(self, data, targets):
        """
        Args:
            data (numpy array): EEG data of shape (trials, channels, 100 timepoints)
            targets (numpy array): Corresponding target data of shape (trials, channels, 1 timepoint)
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

loader = EEGDataLoader("/home/hanwei/Music/MBCI006")
X_train_all, X_train_next, y_train_onehot_all = loader.load_data()
print(X_train_all.shape)
print(X_train_next.shape)
print(y_train_onehot_all.shape)
# Create DataLoader
batch_size = 32
train_dataset = EEGDataset(X_train_all, X_train_next)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pos_embedding[:, :x.size(1), :]

class EEGTransformer(nn.Module):
    def __init__(self, input_dim=20, seq_len=500, d_model=128, nhead=4, num_layers=6, dim_feedforward=256, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = LearnablePositionalEncoding(d_model, seq_len)

        # Transformer Encoder with LayerNorm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, 
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output Layer
        self.fc_out = nn.Linear(d_model, 20) #(d_model,channels)

        # Dropout & Layer Norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch_size, 20, 100) -> EEG input
        Output: (batch_size, 20, 1) -> Next timepoint prediction per channel
        """
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        x = self.embedding(x)  
        x = self.positional_encoding(x)
        x = self.layer_norm(x)  # Normalize before transformer

        # Transformer Encoding (Local Attention Window)
        x = self.transformer_encoder(x[:, -50:, :])  # Last N time steps only
        # print(x.shape)

        # Extract N time step
        # x = x[:, :, :]  # (batch_size, d_model)
        # print(x.shape)
        # Fully Connected Output
        x = self.fc_out(x)  # (batch_size, 1)
        # print(x.shape)
        # Reshape for final output
        # x = x.unsqueeze(1).expand(-1, 20, 50)  # (batch_size, channels, pred_length)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        return x

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGTransformer().to(device)
from torchinfo import summary
summary(model,input_size= (32,20,500)) # batch, channels, input_timepoints
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training Loop
num_epochs = 100
loss_history = []

def train_model(model, train_loader, epochs=50, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber Loss

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            x, y = batch  # (batch_size, channels, input_timepoints), (batch_size, channels, pred_length)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        scheduler.step()  # Adjust learning rate
        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    return model

model = train_model(model, train_loader, epochs=num_epochs, lr=5e-2)

# Plot Loss Curve
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

def visualize_attention(model, sample):
    model.eval()
    sample = sample.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        x = sample.permute(0, 2, 1)  # Reshape for Transformer
        x = model.embedding(x)  
        x = model.positional_encoding(x)  

        # Manually extract attention weights
        attn_weights_list = []
        for layer in model.transformer_encoder.layers:
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            attn_weights_list.append(attn_weights.cpu().numpy())  # Convert to NumPy
            
    # Plot the attention heatmap
    attn_map = attn_weights_list[0][0]  # Get attention from first layer
    plt.figure(figsize=(10, 5))
    plt.imshow(attn_map, cmap="viridis", aspect="auto")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Timepoints (Query)")
    plt.ylabel("Timepoints (Key)")
    plt.title("Attention Heatmap (First Layer)")
    plt.show()

# Test on a sample
sample_eeg = train_dataset[0][0]  # Get a sample EEG data
visualize_attention(model, sample_eeg)

import seaborn as sns
import matplotlib.pyplot as plt

def visualize_eeg_prediction(original, predicted):
    """
    Concatenates the original timepoints with the predicted 101st timepoint
    and visualizes it as a heatmap.

    Args:
        original (torch.Tensor): Shape (20, timepoints) - Original EEG data
        predicted (torch.Tensor): Shape (20, pred_length) - Predicted next timepoint
    """
    # Convert to NumPy for visualization
    original = original.cpu().numpy()
    predicted = predicted.cpu().numpy()

    print(original.shape)
    print(predicted.shape)

    # Concatenate along the time dimension
    eeg_full = np.concatenate((original, predicted), axis=1)

    # Plot the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(eeg_full, cmap="viridis", cbar=True)
    plt.xlabel("Timepoints (100 original + 1 predicted)")
    plt.ylabel("EEG Channels")
    plt.title("EEG Prediction Heatmap")
    plt.show()

# Example usage
sample_input, sample_output = train_dataset[0]  # Get a sample EEG trial
sample_pred = model(sample_input.unsqueeze(0).to(device)).squeeze(0).detach().cpu()  # Predict next timepoint

print(sample_input.shape)
print(sample_pred.shape)
print(sample_output.shape)

visualize_eeg_prediction(sample_input, sample_pred*2)
visualize_eeg_prediction(sample_input, sample_output)
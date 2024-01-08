import os
import time
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from model import MLP
from vocab import get_vocab


TEXT_FILE = "./data/names.txt"
EOT_TOKEN = "<|endoftext|>"
LATENT_SIZE = 8
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 16384
NUM_EPOCHS = 200
LEARNING_RATE = 3e-4


save_folder = f"./models/{int(time.time())}"
os.mkdir(save_folder)

# data processing
with open(TEXT_FILE, "r") as f:
    text_data = [line.strip() for line in f.readlines()]

vocab_dict, vocab_size = get_vocab(text_data, EOT_TOKEN)

training_list = []
for i, name in enumerate(text_data):
    latent = torch.randn(LATENT_SIZE)
    for j, letter in enumerate(list(name)):
        training_list.append({
            "number": i,
            "length": len(name),
            "index": j,
            "normalized_index": -1 + 2 * j / (len(name)),
            "letter": letter,
            "id": vocab_dict["letter2id"][letter],
            "latent": latent
        })
    training_list.append({
        "number": i,
        "length": len(name),
        "index": len(name),
        "normalized_index": -1 + 2 * len(name) / len(name),
        "letter": EOT_TOKEN,
        "id": vocab_dict["letter2id"][EOT_TOKEN],
        "latent": latent
    })

# training setup
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input": torch.tensor(self.data[idx]["index"], dtype=torch.float32).view(-1, 1),
            "output": torch.tensor(self.data[idx]["id"], dtype=torch.int64),
            "latent": self.data[idx]["latent"]
        }

dataset = CustomDataset(training_list)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MLP(input_size=1,
            latent_size=LATENT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            output_size=vocab_size).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# training
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0

    for data in tqdm(dataloader):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            x=data["input"].to("cuda").view(-1, 1),
            latent=data["latent"].to("cuda")
        )

        # Compute the loss
        loss = criterion(outputs, data["output"].to("cuda"))

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {average_loss:.4f}')
    torch.save(model.state_dict(), f"{save_folder}/epoch_{epoch}.ckpt")

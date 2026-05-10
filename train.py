import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LST import ActionLSTM
from CustomDataset import ActionDataset

ACTIONS = ['wave', 'fist', 'grab']
DATA_PATH = 'MP_Data_V2'
BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 50000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ActionDataset(DATA_PATH, ACTIONS)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ActionLSTM(input_size=258, hidden_size=128, num_classes=len(ACTIONS)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training on {DEVICE}...")
for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs,labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")
torch.save(model.state_dict(), f"models/{DEVICE}.pt")
print("Model saved as action_model.pth")
import torch.nn as nn



class ActionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ActionLSTM,self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]

        logits = self.fc(last_time_step)
        return logits
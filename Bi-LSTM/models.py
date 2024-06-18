import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class WeatherLSTM(nn.Module):
    def __init__(self, config):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size, hidden_size=config.hidden_size,
            num_layers=config.num_layers, batch_first=True, bidirectional=True
            )
        self.fc = nn.Linear(2 * config.hidden_size, config.output_size) 

    def forward(self, x, seq_len, config):

        h0 = torch.zeros(2 * config.num_layers, x.size(0), config.hidden_size).to(config.device)
        c0 = torch.zeros(2 * config.num_layers, x.size(0), config.hidden_size).to(config.device)

        packed = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)

        out, _ = self.lstm(packed, (h0, c0))

        out, _ = pad_packed_sequence(out, batch_first=True)

        last_output = out[torch.arange(out.size(0)),  seq_len - 1, :]
        out = self.fc(last_output)

        return out.unsqueeze(dim=1).unsqueeze(dim=2)


class EarlyFusion(torch.nn.Module):
    def __init__(self, config):
        super(EarlyFusion, self).__init__()
        self.config = config
        self.emb_snp = nn.Embedding(config.vocab_size_snp, config.embedding_dim_snp)

        self.weather_lstm = WeatherLSTM(config)

        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=config.channel[0],
                      kernel_size=(config.kernel_size[0], config.embedding_dim_snp),
                      stride=config.stride[0], padding=(2, 0), padding_mode='zeros'),
            nn.BatchNorm2d(config.channel[0]),
            nn.LeakyReLU(config.slope)
        )

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=config.channel[0], out_channels=config.channel[1],
                      kernel_size=config.kernel_size[1], stride=config.stride[1],
                      padding=2, padding_mode='zeros'),
            nn.BatchNorm1d(config.channel[1]),
            nn.LeakyReLU(config.slope)
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=config.channel[1], out_channels=config.channel[2],
                      kernel_size=config.kernel_size[2], stride=config.stride[2],
                      padding=2, padding_mode='zeros'),
            nn.BatchNorm1d(config.channel[2]),
            nn.LeakyReLU(config.slope)
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            nn.Linear(72864, 400), 
            nn.LeakyReLU(config.slope),
            nn.Dropout(config.dropout),
            nn.Linear(400, 20),
            nn.LeakyReLU(config.slope),
            nn.Linear(20, 1)
        )

    def forward(self, x_snp, x_env, seq_len, args):
        x_env = self.weather_lstm(x_env, seq_len, args)
        x = torch.add(self.emb_snp(x_snp), x_env)
        x = self.conv2d(x)
        x = x.squeeze(dim=3)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

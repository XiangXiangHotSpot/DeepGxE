import torch
import torch.nn as nn
import math
from functools import partial
from collections import OrderedDict


class WeatherTransformer(nn.Module):
    def __init__(self, config):
        super(WeatherTransformer, self).__init__()

        tokens_dim=9
        num_tokens=41

        output_size = config.embedding_dim_snp
        mlp_hidden_dim = int(tokens_dim * config.mlp_ratio)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, tokens_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens+1, tokens_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = tokens_dim,
            nhead=config.nhead,
            dim_feedforward=mlp_hidden_dim,
            norm_first=True,
            batch_first = True
            )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
            )
        
        self.head = nn.Linear(tokens_dim, output_size)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(_init_vit_weights)

        
    def forward(self, x, seq_len, config):
        #  x: -> [batch_size, num_tokens, embedding_dim_snp]
        #  seq_len: -> [batch_size]        

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        _, S, E = x.shape
        padding_mask = torch.arange(S).expand(len(seq_len), S) >= seq_len.unsqueeze(1)

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask.to(config.device))

        return self.head(x[:, 0])


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# Early Fusion Genomic Selection Model
class EarlyFusion(torch.nn.Module):
    def __init__(self, config):
        super(EarlyFusion, self).__init__()
        self.config = config
        self.emb_snp = nn.Embedding(config.vocab_size_snp, config.embedding_dim_snp)

        self.weather_transformer = WeatherTransformer(config)

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
        x_env = self.weather_transformer(x_env, seq_len, args)
        x_snp = self.emb_snp(x_snp)

        x = torch.add(x_snp, x_env.unsqueeze(dim=1).unsqueeze(dim=2))
        x = self.conv2d(x)
        x = x.squeeze(dim=3)
        x = self.conv1d_1(x)
        x = self.conv1d_2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

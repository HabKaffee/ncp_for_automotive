from typing import Any
from ncps.torch import LTC
from ncps.wirings import AutoNCP

import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.encoder import EncoderResnet18

class Model(nn.Module):
    def __init__(self, output_size, units) -> None:
        super().__init__()
        self.encoder = EncoderResnet18()
        self.input_size = self.encoder.output_size
        self.output_size = output_size
        self.units = units
        self.rnn = LTC(self.input_size, AutoNCP(self.units, self.output_size), batch_first=True)

    def forward(self, input, hx=None, timespans=None):
        features = self.encoder.extract_features(input)
        return self.rnn(features, hx, timespans)


class SequenceLearner(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optim = self.configure_optimizers()
    
    def training_step(self, data, true_angle):
        y_hat, _ = self.model.forward(data)
        y_hat = torch.mean(y_hat[0])
        loss = nn.MSELoss()(y_hat, true_angle)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}
    
    def validation_step(self, batch, batch_idx):
        # x, y = batch
        # y_hat, _ = self.model.forward(x)
        # y_hat = y_hat.view_as(y)
        # loss = nn.MSELoss()(y_hat, y)

        # self.log("val_loss", loss, prog_bar=True)
        # return loss
        return 0.0
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

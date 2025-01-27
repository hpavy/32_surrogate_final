import torch
import torch.nn as nn
from torch.autograd import grad


class PINNs(nn.Module):
    def __init__(self, hyper_param):
        super().__init__()
        self.init_layer = nn.ModuleList([nn.Linear(5, hyper_param["nb_neurons"])])
        self.hiden_layers = nn.ModuleList(
            [
                nn.Linear(hyper_param["nb_neurons"], hyper_param["nb_neurons"])
                for _ in range(hyper_param["nb_layers"] - 1)
            ]
        )
        self.final_layer = nn.ModuleList([nn.Linear(hyper_param["nb_neurons"], 3)])
        self.layers = self.init_layer + self.hiden_layers + self.final_layer
        self.initial_param()

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            if k != len(self.layers) - 1:
                x = torch.tanh(layer(x))
            else:
                x = layer(x)
        return x  # Retourner la sortie

    def initial_param(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


if __name__ == "__main__":
    hyper_param = {"nb_layers": 12, "nb_neurons": 64}
    piche = PINNs(hyper_param)
    nombre_parametres = sum(p.numel() for p in piche.parameters() if p.requires_grad)
    print(nombre_parametres)

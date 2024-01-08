from torch import nn


# model
class MLP(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)

        self.latent_norm = nn.LayerNorm(latent_size)
        self.latent_linear = nn.Linear(latent_size, hidden_size)

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.Dropout())
            layers.append(nn.Linear(hidden_size, hidden_size))

        # Add the final layer
        layers.append(nn.GELU())
        layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.Dropout())
        layers.append(nn.Linear(hidden_size, output_size))

        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, latent):
        x = self.input_linear(x)

        latent = self.latent_norm(latent)
        latent = self.latent_linear(latent)

        x = x + latent

        for layer in self.layers:
            x = layer(x)
        return x

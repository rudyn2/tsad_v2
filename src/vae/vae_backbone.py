from re import X
from turtle import forward
import torch
import torch.nn as nn
from torch import Tensor
from vae import BaseVAE

__TIM_MODELS__ = {
    "mobilenetv2_100": 320,
    'mobilenetv3_small_100': 576,
    'mobilenetv3_small_075': 432,
    'mobilenetv3_large_100': 960,
    'mobilenetv3_large_075': 720,
    'regnety_032': 1512,
    'efficientnet_b0': 320,
    'efficientnet_b1': 320,
    'efficientnet_b2': 352,
    'efficientnet_b3': 384,
    'efficientnet_b4': 448,
    'efficientnet_b5': 512,
    'efficientnet_b6': 576,
    'efficientnet_b7': 640,
    'efficientnet_b8': 704,
    'efficientnet_l2': 1376,
    'efficientnet_lite0': 320,
    'efficientnet_lite4': 448,
}
class TimmBackbone(nn.Module):
    def __init__(self, model_name: str = "mobilenetv2_100", in_channels=4, out_channels=512):
        super(TimmBackbone, self).__init__()
        self.name = model_name

        import timm
        self.backbone = timm.create_model(
            model_name, pretrained=True, features_only=True)
        self.conv_input_channels = torch.nn.Conv2d(
            in_channels, 3, kernel_size=(1, 1), bias=False
        )
        self.conv_adjust_channels = torch.nn.Conv2d(__TIM_MODELS__[model_name], out_channels,
                                                    kernel_size=(1, 1),
                                                    bias=False)
        self.pool_adjust_dim = torch.nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.conv_input_channels(x)
        x = self.backbone(x)[-1]
        x = self.conv_adjust_channels(x)
        x = self.pool_adjust_dim(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class BasicDecoder(nn.Module):
    def __init__(self, latent_dim, num_clases):
        super(BasicDecoder, self).__init__()
        # Build Decoder
        modules = []
        hidden_dims = [16, 32, 64, 128, 256, 512]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 9)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=5,
                               stride=3,
                               padding=2,
                               output_padding=2),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=num_clases, stride=1,
                      kernel_size=3, padding=1))

    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 3, 3)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

class UpConvBlock(nn.Module):
    """
    x2 space resolution
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(UpConvBlock, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), **kwargs)
        self.double_conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up_conv(x)
        x = self.double_conv(x)
        return x

class ImageSegmentationBranch(nn.Module):

    def __init__(self, in_channels: int, output_channels: int):
        super(ImageSegmentationBranch, self).__init__()
        self.in_channels = in_channels
        self.up1 = UpConvBlock(in_channels, in_channels // 2)
        self.up2 = UpConvBlock(in_channels // 2, in_channels // 4, padding=(1, 1), output_padding=(1, 1))
        self.up3 = UpConvBlock(in_channels // 4, in_channels // 8, padding=(1, 1), output_padding=(1, 1))
        self.up4 = UpConvBlock(in_channels // 8, in_channels // 16, padding=(1, 1), output_padding=(1, 1))
        self.up5 = UpConvBlock(in_channels // 16, in_channels // 32, padding=(1, 1), output_padding=(1, 1))
        self.up6 = UpConvBlock(in_channels // 32, in_channels // 64, padding=(1, 1), output_padding=(1, 1))
        # self.output_conv = nn.Conv2d(in_channels // 64, output_channels, kernel_size=(1, 1))
        self.output_conv = OutputConv(in_channels // 64, output_channels, mid_channels=in_channels // 128)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        x = self.output_conv(x)
        return x

class OutputConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super(OutputConv, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, mid_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d((288, 288))
        self.last_layer = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = self.double_conv(x)
        x = self.avg_pool(x)
        x = self.last_layer(x)
        return x


class VAEBackbone(BaseVAE):
    def __init__(self,
                 latent_dim: int,
                 num_clases: int = 6,
                 in_channels: int = 4,
                 loss=None,
                 backbone: str='efficientnet_l2',
                 out_backbone: int=512,
                 **kwargs) -> None:
        super(VAEBackbone, self).__init__()
        self.out_backbone = out_backbone
        self.pred_dims = out_backbone*(4 * 4)
        self.latent_dim = latent_dim
        if loss:
            self.loss_fn = loss
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.encoder = TimmBackbone(backbone, in_channels, out_backbone)
        
        self.fc_mu = nn.Linear(self.pred_dims, latent_dim)
        self.fc_var = nn.Linear(self.pred_dims, latent_dim)

        self.pre_decoder = nn.Linear(latent_dim, 256*3*3)
        self.decoder = BasicDecoder(latent_dim, num_clases)
        # self.decoder = ImageSegmentationBranch(latent_dim, num_clases)

    def encode(self, input: Tensor) -> "list[Tensor]":
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor) -> Tensor:
        # z = self.pre_decoder(z)
        # z = z.view(-1, 256, 3, 3)
        result = self.decoder(z)
        return result
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> "list[Tensor]":
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss_function(self,
                      recons,
                      target,
                      mu,
                      log_var,
                      **kwargs) -> dict:
        if not 'M_N' in kwargs.keys():
            kld_weight = 1
        else:
            # Account for the minibatch samples from the dataset
            kld_weight = kwargs['M_N']
        recons_loss = self.loss_fn(recons, target)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
        

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from models.carlaDataset import HDF5Dataset

    model = VAEBackbone(256).to('cuda')

    path = '/home/client/databases/batches/'
    dataset = HDF5Dataset(path)
    obs, semantic = dataset[0]

    obs = obs.to('cuda').unsqueeze(dim=0)
    semantic = semantic.to('cuda').long().unsqueeze(dim=0)

    reconst, mu, log_var = model(obs)
    print(model.loss_function(reconst, semantic, mu, log_var, M_N=1))
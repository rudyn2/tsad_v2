import argparse
import sys

sys.path.append('..')
import matplotlib.pyplot as plt

import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim

from vae import VanillaVAE
from models.carlaDataset import HDF5Dataset
from termcolor import colored
from losses import DiceLoss, FocalLoss, WeightedPixelWiseNLLoss
from utils import DiceCoefficient, iou_pytorch

class VAETrainer(object):
    def __init__(self,
        model,
        optimizer,
        dataset,
        epochs,
        batch_size,
        train_test_split = 0.1,
        eval_frequency = 1,
        test_frequency = 1,
        checkpoint_frequency = 1,
        device = 'cpu',
        kld_weight = 1,
        checkpoint_dir = '/home/client/params',
        images_dir = '/home/client/images',
        metrics = None,
    ) -> None:
        super(VAETrainer).__init__()
        self._epochs = epochs
        self._eval_frequency = eval_frequency
        self._test_frequency = test_frequency
        self._model = model
        self._optimizer = optimizer
        self.device = device
        self._kld_weight = kld_weight
        self._current_train_step = 0
        self._current_eval_step = 0
        self._best_loss = 1e10
        self._checkpoint_dir = checkpoint_dir
        self._images_dir = images_dir
        self._checkpoint_frequency = checkpoint_frequency
        self._metrics = metrics

        print(colored("[*] Initializing dataset and dataloader", "white"))
        self._eval_size = int(len(dataset) * train_test_split)
        self._train_size = len(dataset) - self._eval_size
        train_set, val_set = torch.utils.data.random_split(dataset, [self._train_size, self._eval_size])
        self._train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
        self._eval_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True)
        print(colored("[+] Dataset & Dataloader Ready!", "green"))

    def train(self):
        for epoch in range(self._epochs):
            running_loss = 0.0
            running_seg_loss = 0.0
            running_kld_loss = 0.0

            for i, batch in enumerate(tqdm(self._train_loader, f"Training epoch {epoch}/{self._epochs}")):
                state, target_semantic = batch
                state = state.to(self.device)
                target_semantic = target_semantic.to(self.device).long()

                self._optimizer.zero_grad()

                reconst, mu, log_var = self._model(state)
                loss = self._model.loss_function(reconst, target_semantic, mu, log_var, M_N=self._kld_weight)
                loss['loss'].backward()
                self._optimizer.step()

                running_loss += loss['loss'].item()
                running_seg_loss += loss['Reconstruction_Loss']
                running_kld_loss += loss['KLD']
                self.batch_log(loss, reconst, target_semantic)
                
            self.train_log(running_loss, running_seg_loss, running_kld_loss)
            if epoch % self._eval_frequency == 0:
                self.eval()
            
            if epoch % self._test_frequency == 0:
                self.test()

            if epoch % self._checkpoint_frequency == 0:
                self.save(epoch)

            self._current_train_step += 1

    def batch_log(self, loss, prediction, target):
        to_log = {
            "Loss": loss['loss'].item(),
            "Seg Loss": loss['Reconstruction_Loss'],
            "KLD Loss": loss['KLD'], 
        }
        # with torch.no_grad():
        #     for name, loss_fn in self._metrics.items():
        #         to_log[name] = loss_fn(prediction, target)
        wandb.log(to_log)

    def train_log(self, loss, seg_loss, kld_loss):
        wandb.log({
            "Train Epoch": self._current_train_step,
            "Train Loss": loss,
            "Train Seg Loss": seg_loss,
            "Train KLD Loss": kld_loss,
        })

    def eval_log(self, loss, accuracy, seg_loss, kld_loss):
        wandb.log({
            "Eval Epoch": self._current_eval_step,
            "Eval Loss": loss,
            "Eval Accuracy": accuracy,
            "Eval Seg Loss": seg_loss,
            "Eval KLD Loss": kld_loss,
        })

    def eval(self):
        running_loss = 0.0
        running_seg_loss = 0.0
        running_kld_loss = 0.0
        nb_samples = 0
        correct = 0
        with torch.no_grad():
            for i, batch in enumerate(self._eval_loader):
                state, target_semantic = batch
                state = state.to(self.device)
                target_semantic = target_semantic.to(self.device).long()

                reconst, mu, log_var = self._model(state)
                loss = self._model.loss_function(reconst, target_semantic, mu, log_var, M_N=self._kld_weight)

                running_loss += loss['loss'].item()
                running_seg_loss += loss['Reconstruction_Loss']
                running_kld_loss += loss['KLD']
                nb_samples += state.shape[0]
                correct += (reconst.argmax(dim=1) == target_semantic).sum().item() / reconst.shape[-1] / reconst.shape[-2]
        
        running_loss /= nb_samples
        correct /= nb_samples

        if running_loss < self._best_loss:
            self._best_loss = running_loss
            self.save(self._current_train_step, name="best_vae.pt")

        self._current_eval_step += 1
        self.eval_log(running_loss, correct, running_seg_loss, running_kld_loss)

    def test(self):
        random_idx = np.random.randint(0, self._eval_size)
        obs, semantic = dataset[random_idx]
        obs = obs.to(self.device).unsqueeze(dim=0)
        semantic = semantic.to(self.device).long().unsqueeze(dim=0)
        with torch.no_grad():
            reconst = self._model(obs)[0].argmax(dim=1)
        reconst = reconst.squeeze(dim=0).cpu().numpy()
        semantic = semantic.squeeze(dim=0).cpu().numpy()
        plt.imsave(f"{self._images_dir}/real_{self._current_train_step}.png", semantic, vmin=0, vmax=6)
        plt.imsave(f"{self._images_dir}/pred_{self._current_train_step}.png", reconst, vmin=0, vmax=6)
    
    def save(self, epoch, name="checkpoint_vae.pt"):
        torch.save({
            "epoch": epoch,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }, f"{self._checkpoint_dir}/{name}")
    
    def load(self, mode='eval'):
        checkpoint = torch.load(self._checkpoint_dir)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if mode == 'eval':
            self._model.eval()
        elif mode == 'train':
            self._model.train()
        else:
            raise AttributeError()
        return checkpoint["epoch"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for VAE training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # VAE Config
    vae_config = parser.add_argument_group("VAE config")
    vae_config.add_argument('-lr', '--learning-rate', default=0.001, type=float,
            help='Learning Rate')
    vae_config.add_argument('-ls', '--latent-space', default=2048, type=int,
            help='Latent space')
    vae_config.add_argument('-kldw', '--kld-weight', default=1, type=float,
            help='Kullback Divergence loss weight')
    vae_config.add_argument('-L', '--loss', default='dice', type=str,
            help='Loss function, can be "dice", "focal" or "weighted"')

    # Training Config
    train_config = parser.add_argument_group("Training config")
    train_config.add_argument('-D', '--data', required=True, type=str,
                        help='Path to data folder')
    train_config.add_argument('-E', '--epochs', default=20,
                        type=int, help='Training epochs')
    train_config.add_argument('-BS', '--batch-size', default=128,
                        type=int, help='Batch size')
    train_config.add_argument('-TTS', '--train-test-split', default=0.1,
                        type=float, help='Percentage of dataset that goes to eval dataset (between 0 and 1)')
    train_config.add_argument('-EF', '--eval-frequency', default=1,
                        type=int, help='Eval every this number of epochs')
    
    args = parser.parse_args()

    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(colored("Using device: ", "white") + colored(device, "green"))

    wandb.init(project="Vae")
    config = wandb.config
    config.args = args

    print(colored("[*] Initializing model, optimizer and loss", "white"))
    if args.loss == 'dice':
        loss = DiceLoss()
    elif args.loss == 'focal':
        loss = FocalLoss()
    elif args.loss == 'weighted':
        loss = WeightedPixelWiseNLLoss({0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    else:
        raise NotImplementedError()
    model = VanillaVAE(args.latent_space, loss=loss).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    print(colored("[+] Model and optimizer are ready!", "green"))

    dataset = HDF5Dataset(args.data)
    metrics = {
        'seg_dice': DiceCoefficient,
        'seg_iou': iou_pytorch,
        'seg_dice_without_background': lambda p, t: DiceCoefficient(p, t, ignore_index=5),
        'seg_iou_without_background': lambda p, t: iou_pytorch(p, t, ignore_index=5),
        'seg_dice_cars_pred': lambda p, t: DiceCoefficient(p, t, only_consider=0),
        'seg_dice_tl': lambda p, t: DiceCoefficient(p, t, only_consider=1),
        'seg_dice_roadlines': lambda p, t: DiceCoefficient(p, t, only_consider=2),
        'seg_dice_roads': lambda p, t: DiceCoefficient(p, t, only_consider=3),
        'seg_dice_sidewalks': lambda p, t: DiceCoefficient(p, t, only_consider=4),
        'seg_dice_background': lambda p, t: DiceCoefficient(p, t, only_consider=5),
    }

    trainer = VAETrainer(model, optimizer, dataset, args.epochs, args.batch_size, args.train_test_split, args.eval_frequency, device=device, kld_weight=args.kld_weight, metrics=metrics)
    trainer.train()
"""solver.py"""

import warnings

warnings.filterwarnings("ignore")

import os
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from models.BVAE import BetaVAE_H, BetaVAE_B
from models.BRVAE import BRES_VAE
from dataset import CustomDataset
from options import TrainOptions

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
writer = SummaryWriter()


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


class Trainer(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim  # model
        self.beta = args.beta  # traine
        self.gamma = args.gamma  # traine
        self.C_max = args.C_max  # traine
        self.C_stop_iter = args.C_stop_iter  # traine
        self.objective = args.objective  # traine
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.nc = 3
        self.decoder_dist = "gaussian"
  
        if args.model == "H":
            net = BetaVAE_H
        elif args.model == "B":
            net = BetaVAE_B
        elif args.model == 'R':
            net = BRES_VAE
        else:
            raise NotImplementedError("only support model H or B")

        self.net = cuda(net(self.nc, self.z_dim), self.use_cuda)
        self.optim = optim.Adam(
            self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

        self.ckpt_dir = os.path.join(args.ckpt_dir, "ckpt")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, "output")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        transform = transforms.Compose(
            [
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
            ]
        )
        test_dataset = CustomDataset(
            data_root=args.dset_dir, csv_file=args.test_csv_file, transform=transform
        )
        self.data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
        )

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        ep_no = 0
        while not out:
            epoch_recon_loss = 0
            epoch_total_kld = 0
            epoch_mean_kld = 0
            i = 0
            for x, _ in self.data_loader:

                self.global_iter += 1
                i += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                mu, logvar, x_recon = self.net(x)
                #import pdb;pdb.set_trace();
                recon_loss = self.net.reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = self.net.kl_divergence_loss(
                    mu, logvar
                )

                if self.objective == "H":
                    beta_vae_loss = recon_loss + self.beta * total_kld
                elif self.objective == "B":
                    C = torch.clamp(
                        self.C_max / self.C_stop_iter * self.global_iter,
                        0,
                        self.C_max.item(),
                    )
                    beta_vae_loss = recon_loss + self.gamma * (total_kld - C).abs()
                elif self.objective == "Standart":
                    beta_vae_loss = recon_loss + total_kld

                epoch_recon_loss += recon_loss.item()
                epoch_total_kld += total_kld
                epoch_mean_kld += mean_kld.item()

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.global_iter % self.display_step == 0:
                    pbar.write(
                        "[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}".format(
                            self.global_iter,
                            recon_loss.item(),
                            total_kld.item(),
                            mean_kld.item(),
                        )
                    )

                    var = logvar.exp().mean(0).data
                    var_str = ""
                    for j, var_j in enumerate(var):
                        var_str += "var{}:{:.4f} ".format(j + 1, var_j)
                    pbar.write(var_str)

                    if self.objective == "B":
                        pbar.write("C:{:.3f}".format(C.item()))

                    writer.add_scalar(
                        "reconstruction loss", recon_loss.item(), self.global_iter
                    )
                    dim_wise_kld_dict = np.array(
                        [[str(i), value.item()] for i, value in enumerate(dim_wise_kld)]
                    )
                    kl_dict = dict(
                        zip(
                            dim_wise_kld_dict[:, 0],
                            np.around(dim_wise_kld_dict[:, 1].astype(np.float), 4),
                        )
                    )
                    kl_dict["total_kld"] = total_kld.item()
                    kl_dict["mean_kld"] = mean_kld.item()
                    # print(kl_dict.keys(),kl_dict.values())
                    writer.add_scalars("kl divergence", kl_dict, self.global_iter)

                    var_dict = np.array(
                        [
                            [str(i), value.item()]
                            for i, value in enumerate(logvar.exp().mean(0))
                        ]
                    )
                    var_dict = dict(
                        zip(
                            var_dict[:, 0],
                            np.around(var_dict[:, 1].astype(np.float), 4),
                        )
                    )
                    writer.add_scalars("posterior variance", var_dict, self.global_iter)

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint("last")
                    pbar.write("Saved checkpoint(iter:{})".format(self.global_iter))

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break
            print("epoch is done", i, ep_no)
           
            
            
            x_grid = make_grid(x)
            reconstructed_grid = make_grid(F.sigmoid(x_recon))
            writer.add_image('original images', x_grid, ep_no)
            writer.add_image('reconstructed images', reconstructed_grid, ep_no)
            self.traverse()
            ep_no += 1
            epoch_recon_loss /= i
            epoch_total_kld /= i
            epoch_mean_kld /= i
            pbar.write(
                "Epoch is done recon loss:{},total kld:{},mean kld:{}".format(
                    epoch_recon_loss, epoch_total_kld, epoch_mean_kld
                )
            )
        pbar.write("[Training Finished]")
        pbar.close()
        
        writer.close()

    def traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img,_ = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        
        fixed_idx = 0
        fixed_img,_ = self.data_loader.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

        Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            
            writer.add_images('traversal images',samples, len(interpolation))

        # if self.save_output:
        #     output_dir = os.path.join(self.output_dir, str(self.global_iter))
        #     os.makedirs(output_dir, exist_ok=True)
        #     gifs = torch.cat(gifs)
        #     gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
        #     for i, key in enumerate(Z.keys()):
        #         for j, val in enumerate(interpolation):
        #             save_image(tensor=gifs[i][j].cpu(),
        #                        fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
        #                        nrow=self.z_dim, pad_value=1)

        #         grid2gif(os.path.join(output_dir, key+'*.jpg'),
        #                  os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ("Only bool type is supported. True or False")
        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {
            "net": self.net.state_dict(),
        }
        optim_states = {
            "optim": self.optim.state_dict(),
        }

        states = {
            "iter": self.global_iter,
            "model_states": model_states,
            "optim_states": optim_states,
        }

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode="wb+") as f:
            torch.save(states, f)
        if not silent:
            print(
                "=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter)
            )

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint["iter"]
            self.net.load_state_dict(checkpoint["model_states"]["net"])
            self.optim.load_state_dict(checkpoint["optim_states"]["optim"])
            print(
                "=> loaded checkpoint '{} (iter {})'".format(
                    file_path, self.global_iter
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def save_images(self, x, x_recon, en):
        self.net_mode(train=False)
        x = make_grid(x, normalize=True)
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        writer.add_images("_reconstruction", images, en)
        self.net_mode(train=True)


if __name__ == "__main__":
    train_args = TrainOptions().parse()
    seed = train_args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Trainer(train_args)

    if train_args.train:
        net.train()
    else:
        net.traverse()


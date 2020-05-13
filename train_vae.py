from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from collections.abc import Iterable
import torch.nn.init as init
import numpy as np
import os
from models.VAE import VAE
from dataset import CustomDataset
from options import TrainOptions

writer = SummaryWriter()


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({"state": model.state_dict(), "epoch": epoch}, path)

    # notify that we successfully saved the checkpoint.
    print("=> saved the model {name} to {path}".format(name=model.name, path=path))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print(
        "=> loaded checkpoint of {name} from {path}".format(
            name=model.name, path=(path)
        )
    )

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint["state"])
    epoch = checkpoint["epoch"]
    return epoch


def train_model(
    model,
    dataset,
    epochs=10,
    batch_size=32,
    sample_size=32,
    lr=3e-04,
    weight_decay=1e-5,
    loss_log_interval=30,
    image_log_interval=300,
    checkpoint_dir="./checkpoints",
    resume=False,
    cuda=False,
):
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,)

    if resume:
        epoch_start = load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1

    for epoch in range(epoch_start, epochs + 1):
        model.train()

        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _) in data_stream:
            # where are we?
            iteration = (epoch - 1) * (len(dataset) // batch_size) + batch_index

            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description(
                (
                    "epoch: {epoch} | "
                    "iteration: {iteration} | "
                    "progress: [{trained}/{total}] ({progress:.0f}%) | "
                    "loss => "
                    "total: {total_loss:.4f} / "
                    "re: {reconstruction_loss:.3f} / "
                    "kl: {kl_divergence_loss:.3f}"
                ).format(
                    epoch=epoch,
                    iteration=iteration,
                    trained=batch_index * len(x),
                    total=len(data_loader.dataset),
                    progress=(100.0 * batch_index / len(data_loader)),
                    total_loss=total_loss.item(),
                    reconstruction_loss=reconstruction_loss.item(),
                    kl_divergence_loss=kl_divergence_loss.item(),
                )
            )

            if iteration % loss_log_interval == 0:

                writer.add_scalars(
                    "losses",
                    {
                        "reconstruction": reconstruction_loss.item(),
                        "kl divergence": kl_divergence_loss.item(),
                        "total": total_loss.item(),
                    },
                    iteration,
                )

        print("#epoch checkpoint!", epoch)
        # save the checkpoint.
        model.eval()
        save_checkpoint(model, checkpoint_dir, epoch)

        images = model.sample(sample_size)
        grid = make_grid(images)
        writer.add_image("images", grid, 0)
        writer.add_graph(model, images)


if __name__ == "__main__":
    train_args = TrainOptions().parse()
    cuda = (not train_args.no_cuda) and torch.cuda.is_available()
    transform = transforms.Compose(
        [
            transforms.Resize((train_args.image_size, train_args.image_size)),
            transforms.ToTensor(),
        ]
    )
    train_dataset = CustomDataset(
        data_root=train_args.dset_dir, csv_file=train_args.csv_file, transform=transform
    )
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        pin_memory=True,
    )
    net = VAE(
        label="vgg",
        image_size=train_args.image_size,
        channel_num=3,
        kernel_num=128,
        z_size=train_args.z_dim,
    )

    if cuda:
        net.cuda()

    # run a test or a training process.
    train_model(
        net,
        dataset="vgg",
        epochs=train_args.epochs,
        batch_size=train_args.batch_size,
        sample_size=32,
        lr=3e-05,
        weight_decay=1e-06,
        checkpoint_dir=train_args.ckpt_dir,
        loss_log_interval=train_args.log_interval,
        image_log_interval=500,
        resume=False,
        cuda=cuda,
    )

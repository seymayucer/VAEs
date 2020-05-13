import argparse
from utils import str2bool


class TrainOptions():
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, parser):
    
        self.isTrain = True
        parser = argparse.ArgumentParser(description='VAEs')

        parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
        parser.add_argument('--seed', default=1, type=int, help='random seed')
        parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
        parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
        parser.add_argument('--batch_size', default=64, type=int, help='batch size')

        parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
        parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
        parser.add_argument('--objective', default='H', type=str, help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
        parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
        parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding beta-VAE')
        parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
        parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
        parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
        parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
        parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

        #parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
        parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
        parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
        parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

        parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
        parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

        parser.add_argument('--display_step', default=100, type=int, help='number of iterations after which loss data is printed and visdom is updated')
        parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is saved')

        parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
        parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
        parser.add_argument('--dset_dir', default='../FaceDatasets/VGGFace2/aligned_train', type=str, help='dataset directory')
        parser.add_argument('--csv_file', default='../FaceDatasets/metadata/train_500_image_data.csv', type=str, help='dataset name')
        parser.add_argument('--test_csv_file', default='../FaceDatasets/metadata/val_500_image_data.csv', type=str, help='dataset name')
             
        parser.add_argument('--image-size', type=int, default=64, metavar='N',help='input shape/ size for training (default: 64)')
    
        parser.add_argument('--epochs', type=int, default=100, metavar='N',help='number of epochs to train (default: 10)')
        parser.add_argument('--no-cuda', action='store_true', default=False,help='enables CUDA training')
        
        parser.add_argument('--log-interval', type=int, default=1, metavar='N',help='how many batches to wait before logging training status')
    

        self.initialized = True

        return parser
    
    def parse(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        return parser.parse_args()


#args = parser.parse_args()
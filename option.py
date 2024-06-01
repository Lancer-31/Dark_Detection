import argparse

parser = argparse.ArgumentParser(description='Visibility Classify')


# Hardware specifications
parser.add_argument('--num_workers', type=int, default=4, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='./Dark2/', help='dataset directory')
parser.add_argument('--dir_data2', type=str, default='./Fog3/', help='dataset directory')
parser.add_argument('--dir_data1', type=str, default='./FRIDA/5fenlei2/', help='dataset directory')
parser.add_argument('--path', type=str, default='/home/tjh/tjh/data/CULane/', help='dataset directory')
parser.add_argument('--work_dir', type=str, default='./checkpoint3/', help='dataset directory')
parser.add_argument('--work_dir1', type=str, default='./checkpoint1/', help='dataset directory')
parser.add_argument('--work_dir2', type=str, default="/home1/tjh/tjh/HVP/checkpoint2/", help='dataset directory')
parser.add_argument('--work_dir4', type=str, default='./checkpoint4/', help='dataset directory')
parser.add_argument('--work_dir5', type=str, default='./checkpoint5/', help='dataset directory')
parser.add_argument('--work_dir6', type=str, default='./checkpoint6/', help='dataset directory')
parser.add_argument('--work_dir7', type=str, default='./checkpoint7/', help='dataset directory')
parser.add_argument('--lane_work_dir', type=str, default='/home/tjh/tjh/HVP/lane_checkpoint/', help='dataset directory')
parser.add_argument('--resume', type=str, default=None, help='checkpoint directory')
parser.add_argument('--resume1', type=str, default="/home1/tjh/tjh/HVP/checkpoint3/ep038.pth", help='checkpoint directory')
parser.add_argument('--resume2', type=str, default=None, help='checkpoint directory')

# Model specifications
parser.add_argument('--models', default='.', help='models name')
parser.add_argument('--pre_train', type=str, default='.', help='pre-trained models directory')

## lane feature detection
parser.add_argument('--backbone', default='18', help='resnet models')
parser.add_argument('--lane_model', default='/home/tjh/tjh/lane-HVP/lane_parameter/ep10.pth', help='pretrained lane feature detection models')

# Training specifications
parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=24, help='input batch size for training')

# Optimization specifications
parser.add_argument('--optimizer', type=str, default='Adam', help='learning rate')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lr_vis', type=float, default=2e-4, help='learning rate')
parser.add_argument('--lr_decay', type=int, default=30, help='learning rate decay per N epochs')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9, help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999, help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,  help='weight decay')
parser.add_argument('--SGD_weight_decay', type=float, default=4e-7, help='weight decay')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='learning rate')

#lane
parser.add_argument('--num_class', type=int, default=5, help='number of epochs to train')
parser.add_argument('--griding_num', type=int, default=128, help='number of epochs to train')
parser.add_argument('--ignore_label', type=int, default=-1, help='number of epochs to train')
parser.add_argument('--TRAIN.IGNORE_LABEL', type=int, default=-1)
parser.add_argument('--LOSS.BALANCE_WEIGHTS', type=int, default=-1)
parser.add_argument('--output',type=str, default='/home/tjh/tjh/data/road/dataset3/mask', help='Output directory')


parser.add_argument('--self_attention', type=int, default=1, help='number of epochs to train')
parser.add_argument('--use_norm', type=int, default=1, help='number of epochs to train')
parser.add_argument('--tanh', type=int, default=1, help='number of epochs to train')
parser.add_argument('--linear', type=int, default=1, help='number of epochs to train')
parser.add_argument('--linear_add', type=int, default=1, help='number of epochs to train')
parser.add_argument('--latent_threshold', type=int, default=1, help='number of epochs to train')
parser.add_argument('--latent_norm', type=int, default=1, help='number of epochs to train')
parser.add_argument('--input_nc', type=int, default=3, help='number of epochs to train')
parser.add_argument('--output_nc', type=int, default=3, help='number of epochs to train')
parser.add_argument('--ndf', type=int, default=64, help='number of epochs to train')
parser.add_argument('--which_model_netD', type=str, default='basic', help='number of epochs to train')
parser.add_argument('--n_layers_D', type=int, default=5, help='number of epochs to train')
parser.add_argument('--n_layers_patchD', type=int, default=4, help='number of epochs to train')
parser.add_argument('--patchD_3', type=int, default=5, help='number of epochs to train')
parser.add_argument('--patchSize', type=int, default=32, help='number of epochs to train')
parser.add_argument('--norm', type=str, default='instance', help='number of epochs to train')
parser.add_argument('--no_lsgan', action='store_true', help='number of epochs to train')
parser.add_argument('--syn_norm', action='store_true', help='number of epochs to train')
parser.add_argument('--vgg_mean', action='store_true', help='number of epochs to train')
parser.add_argument('--vgg_maxpooling', action='store_true', help='number of epochs to train')
parser.add_argument('--no_vgg_instance', action='store_true', help='number of epochs to train')
parser.add_argument('--use_avgpool', type=float, default=0, help='number of epochs to train')
parser.add_argument('--skip', type=int, default=0, help='number of epochs to train')
parser.add_argument('--vgg_choose', type=str, default='relu5_1', help='number of epochs to train')
parser.add_argument('--noise', type=float, default=0, help='number of epochs to train')

parser.add_argument('--stage', type=int, default=3, help='number of epochs to train')

args = parser.parse_args()



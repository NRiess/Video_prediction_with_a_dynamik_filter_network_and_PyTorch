import torch
from main_pytorch import *

options = {
    # global setup settings, and checkpoints
    'name': 'highwayDriving',
    'seed': 123,
    'checkpoint_output_directory': 'checkpoints',

    # model and dataset
    'dataset_file': 'datasets.dataset_highwayDriving',
    'model_file': 'models.model_highwayDriving_pytorch',
    'pretrained_model_path': None,

    # training parameters
    'image_dim': 64,
    'batch_size': 7,  # 16
    'loss': 'squared_error',
    'learning_rate': 1e-3,
    'decay_after': 20,
    'batches_per_epoch': 10, #200
    'save_after': 10
}

modelOptions = {
    'batch_size': options['batch_size'],
    'npx': options['image_dim'],
    'input_seqlen': 3,
    'target_seqlen': 3,
    'buffer_len': 2,
    'dynamic_filter_size': (11, 11),
    'refinement_network': False,
    'dynamic_bias': True
}
options['modelOptions'] = modelOptions

datasetOptions = {
    'batch_size': options['batch_size'],
    'image_size': options['image_dim'],
    'num_frames': modelOptions['input_seqlen'] + modelOptions['target_seqlen'],
    'mode': 'test'
}
options['datasetOptions'] = datasetOptions

parser = argparse.ArgumentParser()

parser.add_argument('--test-batch-size', type=int, default=options['batches_per_epoch'], metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')

parser.add_argument('--batch-size', type=int, default=options['batch_size'], metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',  # 100
                    help='number of epochs to train (default: 14)')
args = parser.parse_args()

input_seqlen = options['modelOptions']['input_seqlen']

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net(batch_size=options['batch_size']).to(device)
model.load_state_dict(torch.load("highway_model.pt, epoch: 100"))

dataset = importlib.import_module(options['dataset_file'])
dh_test = dataset.DataHandler(**options['datasetOptions'])


writer = SummaryWriter('runs')
test(model, device, dh_test, input_seqlen, writer, "/home/rns4fe/Documents/code-for-github2/dfn_pytorch/highway_model.pt, epoch: 100")


import importlib
from utils.helperFunctions_pytorch import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from layers.dynamic_filter_layer_pytorch import DynamicFilterLayer

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
    'batch_size': 7,                        # 16
    'loss': 'squared_error',
    'decay_after': 20,
    'batches_per_epoch': 200,               # 200
    'save_after': 10
}

modelOptions = {
    'batch_size': options['batch_size'],
    'npx': options['image_dim'],
    'input_seqlen': 3,
    'target_seqlen': 3,                    # 3
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
    'mode': 'train'
}
options['datasetOptions'] = datasetOptions

parser = argparse.ArgumentParser()

parser.add_argument('--test-batch-size', type=int, default=options['batches_per_epoch'], metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--gamma', type=float, default=0.5, metavar='M')             # 0.5
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=123, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=True,
                    help='For Saving the current Model')

parser.add_argument('--batch-size', type=int, default=options['batch_size'], metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR')                 # 1e-3
parser.add_argument('--epochs', type=int, default=100, metavar='N',               # 100
                    help='number of epochs to train (default: 14)')
parser.add_argument('--num_of_train_frames', type=int, default=10, metavar='N',               # 100
                    help='number of frames used for training (default: 10)')
args = parser.parse_args()

# original net:

class Net(nn.Module):
    def __init__(self, npx=64, batch_size=options['batch_size'], input_seqlen=3, target_seqlen=3, buffer_len=1, dynamic_filter_size=(9,9), refinement_network=False, dynamic_bias=False):
        super(Net, self).__init__()
        self.npx = npx
        self.batch_size = batch_size
        self.input_seqlen = input_seqlen
        self.target_seqlen = target_seqlen
        self.nInputs = buffer_len
        self.dynamic_filter_size = dynamic_filter_size
        self.refinement_network = refinement_network
        self.dynamic_bias = dynamic_bias
        self.filter_size = self.dynamic_filter_size[0]

        ## encoder
        self.conv1 = torch.nn.Conv2d(1, 32,(3,3) , stride=(1,1), padding='same', bias=True)
        self.conv2 = torch.nn.Conv2d(32, 32,(3,3) , stride=(2,2), padding=1, bias=True)
        self.conv3 = torch.nn.Conv2d(32, 64,(3,3) , stride=(1,1), padding='same', bias=True)
        self.conv4 = torch.nn.Conv2d(64, 64,(3,3) , stride=(1,1), padding='same', bias=True)

        ## mid
        self.conv5 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding='same', bias=True)
        self.conv6 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding='same', bias=True)

        ## decoder
        self.conv7 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding='same', bias=True)
        self.upscale = torch.nn.Upsample(scale_factor=2)
        self.conv8 = torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding='same', bias=True)
        self.conv9 = torch.nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding='same', bias=True)
        self.conv10 = torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding='same', bias=True)

        ## filter-generative layers
        self.conv11 = torch.nn.Conv2d(128, self.filter_size**2 + self.dynamic_bias,(1,1) , stride=(1,1), padding=(0,0), bias=True)
        self.softmax = torch.nn.Softmax()

        #self.conv12 = torch.nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding='same', bias=True)
        #self.conv13 = torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding='same', bias=True)
        #self.conv14 = torch.nn.Conv2d(32, 1, (3, 3), stride=(1, 1), bias=True)

    def forward(self, input_batch):

        hidden_state = np.zeros((self.batch_size, 64, int(self.npx/2), int(self.npx/2)))
        hidden_state = torch.tensor(hidden_state).float().to('cuda')
        inputs = input_batch

        outputs = []
        for i in range(self.input_seqlen - self.nInputs + self.target_seqlen): # i = 0,1,2,3,4
            test1 = self.input_seqlen - self.nInputs + self.target_seqlen # 5 = 3 - 1 + 3
            test2= self.input_seqlen - self.nInputs # 2 = 3 - 1

            input = inputs[:, 0:self.nInputs,:,:] # 0:1
            output, hidden_state, filters = self.init_hidden(input, hidden_state)
            inputs = inputs[:,1:,:,:]
            if i >= self.input_seqlen - self.nInputs: # i= 2, 3, 4
                inputs = torch.cat([inputs, output], dim=1)
                inputs = inputs.float()
                outputs.append(output)
        return outputs, filters



    def init_hidden(self, input_batch, hidden_state):

        npx = self.npx  # image size
        filter_size = self.dynamic_filter_size[0]

        ###############################
        #  filter-generating network  #
        ###############################
        ## encoder

        output = F.leaky_relu(self.conv1(input_batch), inplace=False)
        output = F.leaky_relu(self.conv2(output), inplace=False)
        output = F.leaky_relu(self.conv3(output), inplace=False)
        output = F.leaky_relu(self.conv4(output), inplace=False)

        ## mid
        hidden = F.leaky_relu(self.conv5(hidden_state))
        hidden = F.leaky_relu(self.conv6(hidden))
        output = torch.stack([output, hidden])
        output = torch.sum(output, dim=0)
        hidden_state = output

        ## decoder
        output = F.leaky_relu(self.conv7(output))
        output = self.upscale(output)
        output = F.leaky_relu(self.conv8(output))
        output = F.leaky_relu(self.conv9(output))

        output = F.leaky_relu(self.conv10(output))

        ## filter-generative layers
        l_filter = self.conv11(output)

        #########################
        #  transformer network  #
        #########################
        ## get inputs

        output = input_batch[:,self.nInputs-1:self.nInputs,:,:]

        ## dynamic convolution
        filters = l_filter[:,0:filter_size ** 2,:,:]

        #torch.set_printoptions(precision=0, threshold=100000000, linewidth=300)
        #print(filters[0,:,:,:])

        filters = filters.permute(0, 2, 3, 1)
        filters = torch.reshape(filters,(-1, filter_size**2))
        filters = self.softmax(filters)
        filters = torch.reshape(filters, (-1, npx, npx, filter_size ** 2))
        filters = filters.permute(0, 3, 1, 2)

        output_dynconv = DynamicFilterLayer([output, filters], filter_size=(filter_size,filter_size,1), pad=(filter_size//2, filter_size//2))

        ########################
        #  refinement network  #
        ########################
        output = output_dynconv

        return output, hidden_state, filters




def train(args, model, dh, optimizer, epoch, input_seqlen, writer):
    model.train()
    losses = []
    torch.set_printoptions(threshold=10_000)

    for batch_idx in range(0, options['batches_per_epoch']):

        batch = dh.GetBatch()  # generate data on the fly
        batch_input = batch[..., :input_seqlen].transpose(0, 4, 2, 3, 1).squeeze(axis=4)  # first frames
        batch_target = batch[..., input_seqlen:].transpose(0, 4, 2, 3, 1).squeeze(axis=4)  # last frame
        batch_input = torch.tensor(batch_input)
        batch_target = torch.tensor(batch_target)
        data, target = batch_input.to('cuda'), batch_target.to('cuda')

        optimizer.zero_grad()
        outputs, filters = model(data)
        outputs = torch.squeeze(torch.stack(outputs, dim=1), dim=2) #.detach()
        target = target.double()
        mse = nn.MSELoss(reduction='sum')
        last_input_rep = data[:,2:3,:,:].repeat(1,3,1,1)
        loss = (mse(outputs, target) - 0.5*mse(last_input_rep, target))/options['modelOptions']['target_seqlen'] / options['batch_size']

        losses.append(loss)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx, options['batches_per_epoch'],
                100. * batch_idx / options['batches_per_epoch']))

    all_losses = sum(losses)/len(losses)
    writer.add_scalar('training_loss', loss.item(), epoch)
    print('Train Epoch: {} \t Average Loss: {:.6f}'.format(epoch, all_losses))


def test(model_file_name, model, device, dh_test, input_seqlen, writer):
    #model = Net(batch_size=options['batch_size'])
    #model.load_state_dict(torch.load(model_file_name))
    #model.to(device)
    model.eval()
    with torch.no_grad():
        batch = dh_test.GetBatch()
        batch_input = batch[..., :input_seqlen].transpose(0, 4, 2, 3, 1).squeeze(axis=4)  # first frames
        batch_target = batch[..., input_seqlen:].transpose(0, 4, 2, 3, 1).squeeze(axis=4)  # last frame

        batch_input = torch.tensor(batch_input)
        batch_target = torch.tensor(batch_target)

        data, target = batch_input.to(device), batch_target.to(device)
        pred, filters = model(data)


        # visualize filter
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        example_filter = filters[0, :, :, :].cpu().numpy()
        z, x, y = example_filter.nonzero()
        sc = ax.scatter(x, y, z, c=z, alpha=1, s=0.1)
        ax.view_init(elev=0, azim=0)
        plt.title('filter')
        plt.colorbar(sc)


        # visualize frames with prediction
        fig, axs = plt.subplots(3, 3)
        torch.set_printoptions(precision=8, threshold=10_000, linewidth=1000)
        for i in range(0, 3):
            single_input = data[0,i,:,:]
            single_input = torch.reshape(single_input, (1, options['image_dim'],options['image_dim']))
            single_input = single_input.to('cpu')
            #print('input', single_input)
            writer.add_image('targets{}'.format(i), single_input)
            single_input = single_input.permute(1, 2, 0)
            axs[0, i].imshow(single_input)
            axs[0, i].set_title('input {}'.format(i))

            single_target = target[0,i,:,:]
            single_target = torch.reshape(single_target, (1, options['image_dim'],options['image_dim']))
            single_target = single_target.to('cpu')
            #print('target', single_target)
            writer.add_image('targets{}'.format(i), single_target)
            single_target = single_target.permute(1, 2, 0)
            axs[1, i].imshow(single_target)
            axs[1, i].set_title('target {}'.format(i))

            single_pred = pred[i]
            single_pred = single_pred[0,:,:,:]
           # print('pred', single_pred)
            single_pred = torch.reshape(single_pred, (1, options['image_dim'], options['image_dim']))
            single_pred = single_pred.to('cpu')
            writer.add_image('prediction{}'.format(i), single_pred)
            single_pred = single_pred.permute(1, 2, 0)
            axs[2, i].imshow(single_pred)
            axs[2, i].set_title('pred {}'.format(i))

            diff = single_target-single_pred
            diff = diff[:,:,0]
            #print('pred - target', diff)

            mse = nn.MSELoss(reduction='sum')
            loss = mse(single_pred, single_target) / options['modelOptions']['target_seqlen']
            print(loss)

        #print(pred[0][0,:,:,:]-data[0,0,:,:])
        #print(pred[0][0,:,:,:]-data[0,1,:,:])
        #print(pred[0][0,:,:,:]-data[0,2,:,:])

        data.tolist()
        writer.close()

    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

def main():
    writer = SummaryWriter('runs')
    # Training settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # ------------ data setup ----------------
    print('Prepare data...')
    dataset = importlib.import_module(options['dataset_file'])
    dh_train = dataset.DataHandler(**options['datasetOptions'])
    dh_train.data_ = dh_train.data_[0:args.num_of_train_frames, :, :,:]
    dh_train.dataset_size_ = args.num_of_train_frames

    datasetOptions = {
        'batch_size': options['batch_size'],
        'image_size': options['image_dim'],
        'num_frames': modelOptions['input_seqlen'] + modelOptions['target_seqlen'],
        'mode': 'test'
    }
    options['datasetOptions'] = datasetOptions

    dh_test = dataset.DataHandler(**options['datasetOptions'])
    #dh.data_ = dh.data_[0:10, :, :,:]
    #dh.dataset_size_ = 10

    input_seqlen = options['modelOptions']['input_seqlen']

    model = Net(batch_size=options['batch_size'])
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=options['decay_after'], gamma=args.gamma)
    starttime = time.process_time()


    for epoch in range(1, 102): #args.epochs + 1):
        train(args, model, dh_train, optimizer, epoch, input_seqlen, writer)

        time_delta = time.process_time() - starttime
        hours, rem = divmod(time_delta, 3600)
        minutes, seconds = divmod(rem, 60)
        time_delta_left = (time.process_time() - starttime) * args.epochs / epoch
        hours_left, rem = divmod(time_delta_left, 3600)
        minutes_left, seconds_left = divmod(rem, 60)
        print('Time: {:0>2}:{:0>2}:{:0>2}\t/\t{:0>2}:{:0>2}:{:0>2}'.format(int(hours),int(minutes), int(seconds), int(hours_left),int(minutes_left),int(seconds_left)))
        scheduler.step()

        if epoch%10==0:
            torch.save(model.state_dict(), "highway_model.pt, epoch: {}".format(epoch))


    model_file_name = "highway_model.pt, epoch: 100"
    test(model_file_name, model, device, dh_test, input_seqlen, writer)

if __name__ == '__main__':
   main()

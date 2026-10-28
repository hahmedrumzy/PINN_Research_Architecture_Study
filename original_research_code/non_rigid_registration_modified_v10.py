import torch
import numpy as np
import argparse
import torch.utils.data
import logging
import os



LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())
#print(LOGGER)


#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def flatten(x):
    return x.view(x.size(0), -1)
    
def chamfer_loss(x,y,ps):
    A= x    
    B= y                                                 
    r=torch.sum(A*A,dim=2) 
    r=r.unsqueeze(-1) 
    r1=torch.sum(B*B,dim=2) 
    r1=r1.unsqueeze(-1)
    t=(r.repeat(1,1,ps) -2*torch.bmm(A,B.permute(0,2,1)) + r1.permute(0, 2, 1).repeat(1,ps,1))
    d1,_=t.min(dim=1)
    d2,_=t.min(dim=2)
    ls=(d1+d2)/2
    return ls.mean()

def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        #layers.append(torch.nn.BatchNorm1d(outp))
        #layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum, track_running_stats=False))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp

    return layers


    
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out
        
def mlp_layers_wo_relu(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    '''
    It is the same with mlp_layers function, except that the ReLU layer is removed. 
    '''
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers        
    
class MLPNet_wo_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers_wo_relu(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out    
    

def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    #a, _ = torch.max(x, dim=-1, keepdim=True)
    return a


class TNet(torch.nn.Module):
    """ [B, K, N] -> [B, K, K]
    """
    def __init__(self, K):
        super().__init__()
        # [B, K, N] -> [B, K*K]
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        self.lin = torch.nn.Linear(256, K*K)

        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        K = inp.size(1)
        N = inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp) # [1, K, K]

        x = self.mlp1(inp)
        x = flatten(torch.nn.functional.max_pool1d(x, N))
        x = self.mlp2(x)
        x = self.lin(x)

        x = x.view(-1, K, K)
        x = x + eye
        return x


class PointNet_features_four_by_four(torch.nn.Module):
    def __init__(self, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(4, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        #self.sy = torch.nn.Sequential(torch.nn.MaxPool1d(num_points), Flatten())
        self.sy = sym_fn

        self.tnet1 = TNet(4) if use_tnet else None
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None

        self.t_out_t2 = None
        self.t_out_h1 = None    

    def forward(self, points):
        """ points -> features
            [B, N, 4] -> [B, K]
        """
        x = points.transpose(1, 2) # [B, 4, N]
        if self.tnet1:
            t1 = self.tnet1(x)
#            print('x shape', x.size())
#            print('t1 shape', t1.size())
            x = t1.bmm(x)
#            print('x shape', x.size())
#        print('the device of x', x.get_device())
        x = self.h1(x)
        if self.tnet2:
            t2 = self.tnet2(x)
            self.t_out_t2 = t2
            x = t2.bmm(x)
        self.t_out_h1 = x # local features

        x = self.h2(x)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x = flatten(self.sy(x))

        return x

class deform_four_by_four(torch.nn.Module):
    def __init__(self, num_c=3,  dim_k=1024):
        super().__init__()
        self.ptfeatures = PointNet_features_four_by_four(use_tnet=True)
              
        # the MLP layers
        mlp_list_layers2= [int(1024), int(512), int(256), int(128), int(64)]
        self.list_layers2 = MLPNet(2*dim_k+3, mlp_list_layers2, b_shared=True).layers
        # the last layer
        last_layer      = [int(3)]
        self.last_layer = MLPNet_wo_relu(64, last_layer, b_shared=True).layers      
                
    def forward(self, data):
        source         = data[:,0,:,:]
        target         = data[:,1,:,:]
        
        allones_source    = torch.ones(source.size(dim=0), source.size(dim=1), 1)
        allones_target    = torch.ones(target.size(dim=0), target.size(dim=1), 1)
        # added due to something in executing compute_TRE_4by4.py
        
#        source    = source.to('cuda')' 
#        target    = target.to('cuda')'        
        
        allones_source    = allones_source.to('cuda')
        allones_target    = allones_target.to('cuda')

#        print('is source on GPU?', source.is_cuda)
#        print('is target on GPU?', target.is_cuda)        
#        print('is allones_source on GPU?', allones_source.is_cuda)
#        print('is allones_target on GPU?', allones_target.is_cuda) 
        
        source_four       = torch.cat((source, allones_source), -1)
        target_four       = torch.cat((target, allones_target), -1)
        
        pffeat_src     = self.ptfeatures(source_four)      
        pffeat_target  = self.ptfeatures(target_four)        # (batch, dim_k)                              
        
        
        global_feature = torch.cat( (pffeat_src,pffeat_target), -1)                                      # (batch, 2*dim_k)
        num_source     = source.shape[1]                                                                 # number of points in the source point set
        global_feature_repeated                 = global_feature.unsqueeze(1).repeat(1, num_source, 1)                   # (batch, num_points_source, 2*dim_k)
        global_feature_repeated_conca           = torch.cat((global_feature_repeated, source) , -1)                # (batch, num_points_source, 2*dim_k+3)
        displacements_source_before_last_layer  = self.list_layers2(global_feature_repeated_conca.permute(0,2,1)) #input to self.list_layers2 is of  (batch, 2*dim_k+3, num_points_source); while output of self.list_layers2 is of          (batch, 64,num_points_source)      
        displacements_source                    = self.last_layer(displacements_source_before_last_layer)# (batch, 3, num_points_source), note that we permute the array
        deformed_source                         = source + displacements_source.permute(0,2,1)           # (batch,  num_points_source, 3), note that we permute the batch of point clouds back

        return deformed_source


def options(argv=None):
    parser = argparse.ArgumentParser(description='Non-Rigid Registration')
    
    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)') 
    parser.add_argument('-t', '--traindata-path', required=True, type=str,
                        metavar='PATH', help='path to the input training dataset')    
    parser.add_argument('-v', '--testdata-path', required=True, type=str,
                        metavar='PATH', help='path to the input validation dataset')    
        
    # settings for input data
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
                        
    # settings for PointNet
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')                    
    

    # settings for training
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--epochs', default=10000, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
                        
    parser.add_argument('--surfacechamfer', default = 'surfaceonly', type=str, 
                        help='use surface points or all points in the chamfer loss')                    
                                              
    args = parser.parse_args(argv)
    return args

def get_sample(data_dir):
    samples = []
    for target in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, target)
        samples.append(path)
    return samples

def downsample_mri_us(sample_return,number_of_samples=1024):
    # the input 1. is the list that contains the mri and us points
    #           2. the number of points in the sampled shape
    mri_points_original =sample_return[0]
    us_points_original  =sample_return[1]
    
    indices_mri = np.random.choice(mri_points_original.shape[0], number_of_samples, replace=False)
    indices_us  = np.random.choice(us_points_original.shape[0], number_of_samples, replace=False)    
    
    mri_points_downsampled=  mri_points_original[indices_mri,:]
    us_points_downsampled =  us_points_original[indices_us,:]
    
    
#    sample_return_downsampled = []
#    sample_return_downsampled.append(mri_points_downsampled)
#    sample_return_downsampled.append(us_points_downsampled)
    
    sample_return_downsampled = np.zeros([2,number_of_samples,3])
#    sample_return_downsampled[0,:,:]= mri_points_downsampled
#    sample_return_downsampled[1,:,:]= us_points_downsampled    
    # not sampled 
    sample_return_downsampled[0,:,:]= mri_points_original
    sample_return_downsampled[1,:,:]= us_points_original  
    return sample_return_downsampled
def convert_list_into_array(sample_return):
    # This script implements the part that converts the data list into array, but without downsampling
    #  Input is a list while the output is an array. 
    number_of_samples         = sample_return[0].shape[0]
    sample_return_downsampled = np.zeros([2,number_of_samples,3])
    
    sample_return_downsampled[0,:,:]= sample_return[0]
    sample_return_downsampled[1,:,:]= sample_return[1]  
    
    return sample_return_downsampled

class prostateset(torch.utils.data.Dataset):
    def __init__(self, rootdir, fileloader, transform = None, downsampleornot =False ):
        samples = get_sample(rootdir)
        self.rootdir    = rootdir
        self.fileloader = fileloader
        self.samples    = samples
        self.transform  = transform 
        self.downsampleornot = downsampleornot # whether we downsample the data again 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        path = self.samples[index]
        #  FOR DEBUGGING ONLY 
        # path = "/home/zmin/registration_mz/train_data/0.npy"
        
        sample = self.fileloader(path, allow_pickle=True)
        if self.transform is not None:
           sample= self.transform(sample)
#        print('sample shape in the prostateset', sample.shape)
        sample_return =[]
        sample_return.append(sample[0])
        sample_return.append(sample[1])  
#        print(sample[0].shape)
        if self.downsampleornot:
          sample_return_downsampled = downsample_mri_us(sample_return)
        else:
#          print('shape of sample_return',sample_return[0].shape[0])
                             
          sample_return_downsampled = convert_list_into_array(sample_return)
#        print('sample_return_downsampled shape in the prostateset', sample_return_downsampled.shape)
        return sample_return_downsampled


class prostateset_v2(torch.utils.data.Dataset):
    '''
    This class implements that the data is read 
    This definition is copied from 'non_rigid_registration_pinns_modified_v10.py'
    
    '''
    def __init__(self, ALL_SAMPLES, transform = None, downsampleornot =False):

        self.samples    = ALL_SAMPLES
        self.transform  = transform 
        self.downsampleornot = downsampleornot # whether we downsample the data again 
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self,index):
        sample = self.samples[index]
        if self.transform is not None:
           sample= self.transform(sample)              
        sample_return =[]
        sample_return.append(sample[0])
        sample_return.append(sample[1])  
        if self.downsampleornot:
          sample_return_downsampled = downsample_mri_us(sample_return)
        else:                             
          sample_return_downsampled = convert_list_into_array(sample_return)
        return sample_return_downsampled
        
def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))

def main(args): 
    loader = np.load
    
    # read the data before the dataset class is used
    samples_path = get_sample(args.traindata_path)
    print('samples_path',           samples_path)
    print('type of samples_path',   type(samples_path))
    print('length of samples_path', len(samples_path))
    
    ALL_SAMPLES = []
    for index in range(0, len(samples_path)):
        current_sample_path = samples_path[index] 
        current_sample      = np.load(current_sample_path, allow_pickle=True)
        ALL_SAMPLES.append(current_sample)
    print('type of ALL_SAMPLES',        type(ALL_SAMPLES))
    print('type of ALL_SAMPLES[0]',     type(ALL_SAMPLES[0]))
    print('shape of ALL_SAMPLES[0]',    ALL_SAMPLES[0].shape)  #(4, 1024, 3)  
    print('shape of ALL_SAMPLES[1]',    ALL_SAMPLES[1].shape)  #(4, 1024, 3)   
    print('shape of ALL_SAMPLES[0][0]', ALL_SAMPLES[0][0].shape)
    
    trainset = prostateset_v2(ALL_SAMPLES)
    testset  = prostateset_v2(ALL_SAMPLES)        
    
        
#    # dataset
#    trainset = prostateset(args.traindata_path, loader)
#    testset  = prostateset(args.testdata_path, loader)

    # training
    run(args, trainset, testset)

def train_1(model, trainloader, optimizer, device, surfacechamfer):
    model.train()
    vloss = 0.0
    count = 0
    for i, data in enumerate(trainloader):
        
#        print(type(data))
        data = data.float()
        data          =  data.to(device)
        deformed      =  model(data)
#        print('is the data input to the model on gpu?', data.is_cuda)
#        print('is the model on gpu?', next(model.parameters()).is_cuda)
#        print('is the prediction on gpu?', deformed.is_cuda)
#        print(i)
#        print('the data used in training', data.shape)  # [batch, 2, 1024, 3]
        #deformed      =  deformed.to(device)
        #print('device', device)
        #data          =  data.to(device)
                
        
        # ''' use the surface points only in the chamfer loss ''' since the first half points are boundary points
        if surfacechamfer =='surfaceonly':
          num_points_used_in_loss_cham =  int(data.shape[2]/2)
          loss_cham                    =  chamfer_loss(deformed[:,0:num_points_used_in_loss_cham,:],    data[:,1,0:num_points_used_in_loss_cham,:],    num_points_used_in_loss_cham)  
          print('training loss_cham', loss_cham.item())      
        else:     
          ''' use all points in the chamfer loss'''
          loss_cham                    =  chamfer_loss(deformed,    data[:,1,:,:],    data.shape[2])
          print('training loss_cham', loss_cham.item())
        
        # forward + backward + optimize 
        optimizer.zero_grad()
        loss_cham.backward()
        optimizer.step()    
        
        vloss += loss_cham.item()
        count += 1
        
    ave_vloss   = float(vloss)/count
    return ave_vloss 
    
def eval_1(model, testloader, device):
    model.eval() #
    vloss = 0
    count = 0

    with torch.no_grad():
         for i, data in enumerate(testloader):
             data = data.float()
             deformed      =  model(data)
             deformed      =  deformed.to(device)
             data          =  data.to(device)               

             # ''' use the surface points only in the chamfer loss '''since the first half points are boundary points
#             num_points_used_in_loss_cham = int(data.shape[2]/2) #print('num_points_used_in_loss_cham', num_points_used_in_loss_cham)
#             loss_cham                    = chamfer_loss(deformed[:,0:num_points_used_in_loss_cham,:],    data[:,1,0:num_points_used_in_loss_cham,:],    num_points_used_in_loss_cham)  
             
             ''' use all points in the chamfer loss '''  
             loss_cham     =  chamfer_loss(deformed, data[:,1,:,:], data.shape[2])           
 
             print('test loss_cham', loss_cham.item())    
             vloss += loss_cham.item()
             count += 1    
    
    ave_vloss   = float(vloss)/count
    return ave_vloss 
    
    
def run(args, trainset, testset):

    if not torch.cuda.is_available():
       args.device = 'cuda'
    args.device = torch.device(args.device)   
    
    
    # the model 
    model = deform_four_by_four()
    #print('args.device', args.device)
    model.to(args.device) 
    #print(model)
    
    if args.resume:
       assert os.path.isfile(args.resume)
       checkpoint       = torch.load(args.resume)
       args.start_epoch = checkpoint['epoch']
       model.load_state_dict(checkpoint['model'])
    
    # optimizer
    min_loss = float('inf')
    print(min_loss)
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.001)
    else:
        optimizer = torch.optim.SGD(learnable_params,  lr=0.1)    
    
#    print(optimizer)
      
     
    # dataloader 
#    trainloader = torch.utils.data.DataLoader(
#       trainset,
#       batch_size=args.batch_size, shuffle=True, num_workers=args.workers)     

    # training       
    LOGGER.debug('train, begin')
    
#    print(model.state_dict())
#    print(optimizer.state_dict())


    loss_values_training_save    = []
    loss_values_validation_save  = []
    for epoch in range(args.start_epoch, args.epochs):
#        trainloader =1
#        testloader = 1
        
        #print(args.device)
        
#        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
#        testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last =True )
#        testloader  = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last =True) 
         
        running_loss = train_1(model, trainloader, optimizer, args.device, args.surfacechamfer)
#        val_loss     = eval_1(model,  testloader, args.device) 
        
               
#        is_best  = val_loss < min_loss
#        min_loss = min(val_loss, min_loss)
#        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss)
                
        # if we do not use the validation during training
        is_best = running_loss<min_loss
        min_loss = min(running_loss, min_loss)         
        
                
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer': optimizer.state_dict(),}
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')                                
        save_checkpoint(snap, args.outfile,     'snap_last')
        save_checkpoint(model.state_dict(), args.outfile, 'model_last')        
       
#        if epoch==19999:
#           save_checkpoint(snap, args.outfile,     'snap_epoch19999')
#           save_checkpoint(model.state_dict(), args.outfile, 'model_epoch19999')               
       
        # append and save the training loss 
        loss_values_training_save.append(running_loss) #training loss
#        loss_values_validation_save.append(val_loss)   #validation loss

        with open(args.outfile+'training.txt', 'w') as fp:
            for item in loss_values_training_save:
                # write each item on a new line
                fp.write("%s\n" % item)
                
#        with open(r'/home/zmin/registration_mz/modified_codes_v10/without_pinns/out_put_folder/validation.txt', 'w') as fp:
#            for item in loss_values_validation_save:
#                # write each item on a new line
#                fp.write("%s\n" % item)

        
    LOGGER.debug('train, end')    

if __name__ == '__main__': 
   print('The Main Function is Running')
   
   ARGS = options()
   print(ARGS.dim_k)
   print(ARGS.device) 
   main(ARGS)









import torch
import numpy as np
import argparse
import torch.utils.data
import logging
import os

def stressxx(xx):
    return (lmbd+2*mu)*strainxx(xx) + lmbd*strainyy(xx) + lmbd*strainzz(xx)
def stressyy(xx):
    return lmbd*strainxx(xx) + (lmbd+2*mu)*strainyy(xx) + lmbd*strainzz(xx)
def stresszz(xx):
    return lmbd*strainxx(xx) + lmbd*strainyy(xx) + (lmbd+2*mu)*strainzz(xx)
def stressxy(xx):
    return 2.0*mu*strainxy(xx)
def stressxz(xx):
    return 2.0*mu*strainxz(xx)
def stressyz(xx):
    return 2.0*mu*strainyz(xx)



def get_lame_vectors_torch(source_point_set, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(8221.47),lame2_version2=torch.tensor(167.78)):
    '''
    Get the point-set-wise lame1 vector and lame2 vector; 
    source_point_set: [1024,3] 
    '''
    
    # assign the tensors to the cuda 0
    lame1_version1 = lame1_version1.to('cuda')
    lame2_version1 = lame2_version1.to('cuda')
    lame1_version2 = lame1_version2.to('cuda')
    lame2_version2 = lame2_version2.to('cuda')
        
    if torch.is_tensor(source_point_set):
       pass
    else:  #convert from array to tensor
       source_point_set                         = torch.from_numpy(source_po1int_set)
    
    source_point_set = source_point_set.to('cuda')
    
#    source_demean_norm                       = torch.norm(source_point_set, dim=1)
#    the_index_smaller_than_threshold         = source_demean_norm<0.48    
#    the_index_smaller_than_threshold_one_more_dimension =torch.unsqueeze(the_index_smaller_than_threshold, -1)
#
#    the_index_larger_than_threshold          = source_demean_norm>=0.48
#    the_index_larger_than_threshold_one_more_dimension =torch.unsqueeze(the_index_larger_than_threshold, -1)


    x_coordinates          = source_point_set[:,2] #2
    x_coordinates_sorted,_ = torch.sort(x_coordinates)
    

    the_distance_two_ends            = x_coordinates_sorted[-1] - x_coordinates_sorted[0]
    two_thirds_the_distance_two_ends = 2*the_distance_two_ends/3
    threshold                        = x_coordinates_sorted[0] + two_thirds_the_distance_two_ends

    the_index_smaller_than_threshold     = torch.zeros(source_point_set.shape[0])
    the_index_smaller_than_threshold     = x_coordinates<=threshold
    the_index_smaller_than_threshold_one_more_dimension =torch.unsqueeze(the_index_smaller_than_threshold, -1)

    lame1_vector = torch.where(the_index_smaller_than_threshold_one_more_dimension==True, lame1_version2, lame1_version1)
    lame2_vector = torch.where(the_index_smaller_than_threshold_one_more_dimension==True, lame2_version2, lame2_version1)


#    #the_index_larger_than_threshold          = torch.logical_not(the_index_smal   ler_than_threshold)
#    print('the_index_smaller_than_threshold', the_index_smaller_than_threshold)
#    print('is the_index_smaller_than_threshold a tensor?', torch.is_tensor(the_index_smaller_than_threshold))
#    print('the size of the the_index_smaller_than_threshold', the_index_smaller_than_threshold.size())
#    print('the size of the unsqueezed the_index_smaller_than_threshold_one_more_dimension', the_index_smaller_than_threshold_one_more_dimension.size())
#    print('the_index_larger_than_threshold', the_index_larger_than_threshold)
#    print('is the_index_larger_than_threshold a tensor', torch.is_tensor(the_index_larger_than_threshold))
#    print('the size of the the_index_larger_than_threshold', the_index_larger_than_threshold.size())
#    print('the size of the unsqueezed the_index_larger_than_threshold_one_more_dimension', the_index_larger_than_threshold_one_more_dimension.size())
#    print('unique values in lame1_vector', torch.unique(lame1_vector))
#    print('unique_values in lame2_vector', torch.unique(lame2_vector))
#    print('shape of lame1_vector', lame1_vector.size()) #(1024, 1)
#    print('shape of lame2_vector', lame2_vector.size()) #(1024, 1)
#    print('size of C11', C11.size()) #torch.Size([1024, 1])
#    print('size of C12', C12.size()) #torch.Size([1024, 1])
#    print('size of C33', C33.size()) #torch.Size([1024, 1])

    return lame1_vector, lame2_vector
    
    
def get_lame_vectors_torch_cuda1(source_point_set, lame1_version1=torch.tensor(82.21), lame2_version1=torch.tensor(1.68), lame1_version2=torch.tensor(8221.47),lame2_version2=torch.tensor(167.78)):
    '''
    Get the point-set-wise lame1 vector and lame2 vector; 
    source_point_set: [1024,3] 
    '''
    
    # assign the tensors to the cuda 0
    lame1_version1 = lame1_version1.to('cuda')
    lame2_version1 = lame2_version1.to('cuda')
    lame1_version2 = lame1_version2.to('cuda')
    lame2_version2 = lame2_version2.to('cuda')
        
    if torch.is_tensor(source_point_set):
       pass
    else:  #convert from numpy array to tensor
       source_point_set                         = torch.from_numpy(source_point_set)
    
    source_point_set = source_point_set.to('cuda')
        
    x_coordinates          = source_point_set[:,2] #2
    x_coordinates_sorted,_ = torch.sort(x_coordinates)
    
    the_distance_two_ends            = x_coordinates_sorted[-1] - x_coordinates_sorted[0]
    two_thirds_the_distance_two_ends = 2*the_distance_two_ends/3
    threshold                        = x_coordinates_sorted[0] + two_thirds_the_distance_two_ends

    the_index_smaller_than_threshold     = torch.zeros(source_point_set.shape[0])
    the_index_smaller_than_threshold     = x_coordinates<=threshold
    the_index_smaller_than_threshold_one_more_dimension =torch.unsqueeze(the_index_smaller_than_threshold, -1)

    lame1_vector = torch.where(the_index_smaller_than_threshold_one_more_dimension==True, lame1_version2, lame1_version1)
    lame2_vector = torch.where(the_index_smaller_than_threshold_one_more_dimension==True, lame2_version2, lame2_version1)


    return lame1_vector, lame2_vector    


def pinn_loss_lame_vector(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3,lame1_vector, lame2_vector):

    num_points  = ux_pred.shape[0]
    zero_vector = torch.zeros((num_points,1)).to('cuda:0')

    C11 = (2*lame2_vector + lame1_vector)
    C12 = lame1_vector
    C33 = 2*lame2_vector
    
    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = mse_loss(momentum_balance1, zero_vector) + mse_loss(momentum_balance2, zero_vector) + mse_loss(momentum_balance3, zero_vector)

    governing_equation1 = mse_loss(Sxx -(torch.mul(C11,Exx)+ torch.mul(C12,Eyy) + torch.mul(C12,Ezz)),  zero_vector)
    governing_equation2 = mse_loss(Syy -(torch.mul(C12,Exx)+ torch.mul(C11,Eyy) + torch.mul(C12,Ezz)),  zero_vector)    
    governing_equation3 = mse_loss(Szz -(torch.mul(C12,Exx)+ torch.mul(C12,Eyy) + torch.mul(C11,Ezz)),  zero_vector)        
    governing_equation4 = mse_loss(Sxy -torch.mul(C33,Exy),   zero_vector)     
    governing_equation5 = mse_loss(Sxz -torch.mul(C33,Exz),   zero_vector) 
    governing_equation6 = mse_loss(Syz -torch.mul(C33,Eyz),  zero_vector)

    elastic_energy      = 0.5*torch.sum(   torch.mul(Exx,Sxx) + torch.mul(Eyy,Syy) + torch.mul(Ezz,Szz) + 2.0*torch.mul(Exy,Sxy) + 2.0*torch.mul(Exz,Sxz) + 2.0*torch.mul(Eyz,Syz) )/num_points

    overall_loss =  governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance    
#    print('size of torch.mul(C11,Exx)', torch.mul(C11,Exx).size())
#    print('size of toch.mul(C12,Eyy)',  torch.mul(C12,Eyy).size())
#    print('size of toch.mul(C12,Ezz)',  torch.mul(C12,Ezz).size())
#    
#    print('size of governing_equation1', governing_equation1.size())    
#    print('size of governing_equation2', governing_equation2.size())
#    print('size of governing_equation3', governing_equation3.size())    
#    print('size of governing_equation4', governing_equation4.size())    
#    print('size of governing_equation5', governing_equation5.size())    
#    print('size of governing_equation6', governing_equation6.size()) 
#    print('size of elastic_energy',      elastic_energy.size())
    
    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance


def pinn_loss_lame_vector_cuda1(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3,lame1_vector, lame2_vector):

    num_points  = ux_pred.shape[0]
    zero_vector = torch.zeros((num_points,1)).to('cuda')

    #C11 = torch.tensor((2*lame2_vector + lame1_vector)).to('cuda')'
    #C12 = torch.tensor(lame1_vector).to('cuda')'
    #C33 = torch.tensor((2*lame2_vector)).to('cuda')'
    
    #print('C11 requires grad?', C11.requires_grad)
    #print('C12 requires grad?', C12.requires_grad)
    #print('C33 requires grad?', C33.requires_grad)
    
    
    C11 = (2*lame2_vector + lame1_vector).clone().detach().to('cuda')
    C12 = lame1_vector.clone().detach().to('cuda')
    C33 = (2*lame2_vector).clone().detach().to('cuda')
    #print('C11 requires grad?', C11.requires_grad)
    #print('C12 requires grad?', C12.requires_grad)
    #print('C33 requires grad?', C33.requires_grad)


    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = mse_loss(momentum_balance1, zero_vector) + mse_loss(momentum_balance2, zero_vector) + mse_loss(momentum_balance3, zero_vector)
    
    #print('Sxx shape', Sxx.size())
    #print('C11 shape', C11.size())
    #print('Exx shape', Exx.size())
    #print('C12 shape', C12.size())
    #print('Eyy shape', Eyy.size())
    #print('Ezz shape', Ezz.size())
    #print('zero_vector shape', zero_vector.size())

    governing_equation1 = mse_loss(Sxx -(torch.mul(C11,Exx)+ torch.mul(C12,Eyy) + torch.mul(C12,Ezz)),  zero_vector)
    governing_equation2 = mse_loss(Syy -(torch.mul(C12,Exx)+ torch.mul(C11,Eyy) + torch.mul(C12,Ezz)),  zero_vector)    
    governing_equation3 = mse_loss(Szz -(torch.mul(C12,Exx)+ torch.mul(C12,Eyy) + torch.mul(C11,Ezz)),  zero_vector)        
    governing_equation4 = mse_loss(Sxy -torch.mul(C33,Exy),   zero_vector)     
    governing_equation5 = mse_loss(Sxz -torch.mul(C33,Exz),   zero_vector) 
    governing_equation6 = mse_loss(Syz -torch.mul(C33,Eyz),   zero_vector)

    elastic_energy      = 0.5*torch.sum(   torch.mul(Exx,Sxx) + torch.mul(Eyy,Syy) + torch.mul(Ezz,Szz) + 2.0*torch.mul(Exy,Sxy) + 2.0*torch.mul(Exz,Sxz) + 2.0*torch.mul(Eyz,Syz) )/num_points

    overall_loss =  governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance    

    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance


def pinn_loss_lame_vector_cuda1_normalised(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3,lame1_vector, lame2_vector):
    '''
       This function is the same with the function "pinn_loss_lame_vector_cuda1", except:
            The loss terms  governing_equation1, governing_equation2, governing_equation3,
            governing_equation4, governing_equation5, governing_equation6 are normalised with 
            the lame parameters.
            The above is also the reason why this function is called "pinn_loss_lame_vector_cuda1_normalised". 
    '''

    num_points  = ux_pred.shape[0]
    zero_vector = torch.zeros((num_points,1)).to('cuda') 
    
    #print('lame1_vector', lame1_vector)
     
    C11 = torch.tensor((2*lame2_vector + lame1_vector)).to('cuda')
    C12 = torch.tensor(lame1_vector).to('cuda')
    C33 = torch.tensor((2*lame2_vector)).to('cuda')
#    print('C11', C11.size())
#    print('Exx', Exx.size())
    
        
    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = mse_loss(momentum_balance1, zero_vector) + mse_loss(momentum_balance2, zero_vector) + mse_loss(momentum_balance3, zero_vector)

    # normalise the loss terms of governing equations
    governing_equation1 = mse_loss(torch.div(Sxx -(torch.mul(C11,Exx)+ torch.mul(C12,Eyy) + torch.mul(C12,Ezz)),C11),zero_vector) 
    governing_equation2 = mse_loss(torch.div(Syy -(torch.mul(C12,Exx)+ torch.mul(C11,Eyy) + torch.mul(C12,Ezz)),C11),zero_vector)
    governing_equation3 = mse_loss(torch.div(Szz -(torch.mul(C12,Exx)+ torch.mul(C12,Eyy) + torch.mul(C11,Ezz)),C11),zero_vector)
    governing_equation4 = mse_loss(torch.div(Sxy -torch.mul(C33,Exy), C33), zero_vector)
    governing_equation5 = mse_loss(torch.div(Sxy -torch.mul(C33,Exy), C33), zero_vector)  
    governing_equation6 = mse_loss(torch.div(Syz -torch.mul(C33,Eyz), C33), zero_vector)  
    
    elastic_energy      = 0.5*torch.sum(   torch.mul(Exx,Sxx) + torch.mul(Eyy,Syy) + torch.mul(Ezz,Szz) + 2.0*torch.mul(Exy,Sxy) + 2.0*torch.mul(Exz,Sxz) + 2.0*torch.mul(Eyz,Syz) )/num_points

    overall_loss =  governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance    

    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance


    
def pinn_loss_cuda1(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3, lame1=82.21, lame2=1.68):
    '''
       ux_pred,   uy_pred,   uz_pred
       ux_target, uy_target, uz_target # they are not used if we only use one individual network to predict the displacement vectors.
       Used in the momentum balance: 
                                   (1) sxx_gradx, sxy_grady, sxz_gradz;  sxy_gradx, syy_grady, syz_gradz; sxz_gradx, syz_grady, szz_gradz;
                                   (2) we denote momentum_balance1 = sxx_gradx+ sxy_grady+ sxz_gradz;
                                                 momentum_balance2 = sxy_gradx+ syy_grady+ syz_gradz;
                                                 momentum_balance3 = sxz_gradx+ syz_grady+ szz_gradz. 
      num_points: the number of points
    '''
#    print('the shape of ux_pred', ux_pred.shape)
    num_points  = ux_pred.shape[0]
#    print('num_points', num_points) # 2048
    zero_vector = torch.zeros((num_points,1)).to('cuda')
    C11 = torch.tensor((2*lame2 + lame1)).to('cuda')
    C12 = torch.tensor(lame1).to('cuda')
    C33 = torch.tensor((2*lame2)).to('cuda')
    
#    print('is C11 on gpu?', C11.is_cuda)
#    print('is C12 on gpu?', C12.is_cuda)
#    print('is C33 on gpu?', C33.is_cuda)
    
    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = mse_loss(momentum_balance1, zero_vector) + mse_loss(momentum_balance2, zero_vector) + mse_loss(momentum_balance3, zero_vector)
    
#    print('is momentum_balance on GPU', torch.tensor(momentum_balance).is_cuda)
    
#    print('momentum_balance', momentum_balance)
#    print('the shape of momentum_balance', momentum_balance.shape)
     
    ''' If we use one network, then we would not need this term. '''
#    data_constraints    = mse_loss(ux_pred-ux_target, zero_vector) + mse_loss(uy_pred-uy_target, zero_vector) + mse_loss(uz_pred-uz_target, zero_vector)

#    print('data_constraints', data_constraints)
    
#    print('shape of Sxx -(C11*Exx+ C12*Eyy + C12*Ezz)', (Sxx -(C11*Exx+ C12*Eyy + C12*Ezz)).shape)
#    print('shape of zero_vector', zero_vector.shape)
    governing_equation1 = mse_loss(Sxx -(C11*Exx+ C12*Eyy + C12*Ezz),  zero_vector)
#    print('is governing_equation1 on GPU', torch.tensor(governing_equation1).is_cuda)
#    print('governing_equation1', governing_equation1.shape)
    governing_equation2 = mse_loss(Syy -(C12*Exx+ C11*Eyy + C12*Ezz),  zero_vector)
#    print('governing_equation2', governing_equation2)
    governing_equation3 = mse_loss(Szz -(C12*Exx+ C12*Eyy + C11*Ezz),  zero_vector)    
#    print('governing_equation3', governing_equation3)   
    governing_equation4 = mse_loss(Sxy -C33*Exy,   zero_vector)   
#    print('governing_equation4', governing_equation4)
    governing_equation5 = mse_loss(Sxz -C33*Exz,   zero_vector)  
#    print('governing_equation5', governing_equation5)     
    governing_equation6 = mse_loss(Syz -C33*Eyz,  zero_vector)
#    print('governing_equation6', governing_equation6)
#    print('num_points',num_points)
    elastic_energy      = sum(0.5*sum(Exx*Sxx + Eyy*Syy + Ezz*Szz + 2.0*Exy*Sxy + 2.0*Exz*Sxz + 2.0*Eyz*Syz))/num_points # the mean elastic energy
#    print('is elastic_energy on GPU', torch.tensor(elastic_energy).is_cuda)
#    print('elastic_energy', elastic_energy)
#    print('shape of elastic_energy', elastic_energy.shape)
    overall_loss =  governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance # the term 'data_constraints' is not used. 
#    print('overall_loss', overall_loss)
    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance
    



def pinn_loss(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3, lame1=82.21, lame2=1.68):
    '''
       ux_pred,   uy_pred,   uz_pred
       ux_target, uy_target, uz_target # they are not used if we only use one individual network to predict the displacement vectors.
       Used in the momentum balance: 
                                   (1) sxx_gradx, sxy_grady, sxz_gradz;  sxy_gradx, syy_grady, syz_gradz; sxz_gradx, syz_grady, szz_gradz;
                                   (2) we denote momentum_balance1 = sxx_gradx+ sxy_grady+ sxz_gradz;
                                                 momentum_balance2 = sxy_gradx+ syy_grady+ syz_gradz;
                                                 momentum_balance3 = sxz_gradx+ syz_grady+ szz_gradz. 
      num_points: the number of points
    '''
#    print('the shape of ux_pred', ux_pred.shape)
    num_points  = ux_pred.shape[0]
#    print('num_points', num_points) # 2048
    zero_vector = torch.zeros((num_points,1)).to('cuda:0')    
    C11 = (2*lame2 + lame1)
    C12 = lame1
    C33 = 2*lame2
    
    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = mse_loss(momentum_balance1, zero_vector) + mse_loss(momentum_balance2, zero_vector) + mse_loss(momentum_balance3, zero_vector)
#    print('momentum_balance', momentum_balance)
#    print('the shape of momentum_balance', momentum_balance.shape)
     
    ''' If we use one network, then we would not need this term. '''
#    data_constraints    = mse_loss(ux_pred-ux_target, zero_vector) + mse_loss(uy_pred-uy_target, zero_vector) + mse_loss(uz_pred-uz_target, zero_vector)

#    print('data_constraints', data_constraints)
    
#    print('shape of Sxx -(C11*Exx+ C12*Eyy + C12*Ezz)', (Sxx -(C11*Exx+ C12*Eyy + C12*Ezz)).shape)
#    print('shape of zero_vector', zero_vector.shape)
    governing_equation1 = mse_loss(Sxx -(C11*Exx+ C12*Eyy + C12*Ezz),  zero_vector)
#    print('governing_equation1', governing_equation1.shape)
    governing_equation2 = mse_loss(Syy -(C12*Exx+ C11*Eyy + C12*Ezz),  zero_vector)
#    print('governing_equation2', governing_equation2)
    governing_equation3 = mse_loss(Szz -(C12*Exx+ C12*Eyy + C11*Ezz),  zero_vector)    
#    print('governing_equation3', governing_equation3)   
    governing_equation4 = mse_loss(Sxy -C33*Exy,   zero_vector)   
#    print('governing_equation4', governing_equation4)
    governing_equation5 = mse_loss(Sxz -C33*Exz,   zero_vector)  
#    print('governing_equation5', governing_equation5)     
    governing_equation6 = mse_loss(Syz -C33*Eyz,  zero_vector)
#    print('governing_equation6', governing_equation6)
#    print('num_points',num_points)
    elastic_energy      = sum(0.5*sum(Exx*Sxx + Eyy*Syy + Ezz*Szz + 2.0*Exy*Sxy + 2.0*Exz*Sxz + 2.0*Eyz*Syz))/num_points # the mean elastic energy
#    print('elastic_energy', elastic_energy)
#    print('shape of elastic_energy', elastic_energy.shape)
    overall_loss =  governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance # the term 'data_constraints' is not used. 
#    print('overall_loss', overall_loss)
    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance
    
    
def pinn_loss_vector_used_for_checking_NOT_USED_FOR_TRAINING(ux_pred, uy_pred, uz_pred, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3, lame1=82.21, lame2=1.68):
    '''
       NOT_USED_FOR TRAINING, FOR CHECKING THE RESULTS
       ux_pred,   uy_pred,   uz_pred
       ux_target, uy_target, uz_target # they are not used if we only use one individual network to predict the displacement vectors.
       Used in the momentum balance: 
                                   (1) sxx_gradx, sxy_grady, sxz_gradz;  sxy_gradx, syy_grady, syz_gradz; sxz_gradx, syz_grady, szz_gradz;
                                   (2) we denote momentum_balance1 = sxx_gradx+ sxy_grady+ sxz_gradz;
                                                 momentum_balance2 = sxy_gradx+ syy_grady+ syz_gradz;
                                                 momentum_balance3 = sxz_gradx+ syz_grady+ szz_gradz. 
      num_points: the number of points
    '''
#    print('the shape of ux_pred', ux_pred.shape)
    num_points  = ux_pred.shape[0]
#    print('num_points', num_points) # 2048
    zero_vector = torch.zeros((num_points,1)).to('cuda:0')    
    C11 = (2*lame2 + lame1)
    C12 = lame1
    C33 = 2*lame2
    
    mse_loss            = torch.nn.MSELoss()
    momentum_balance    = momentum_balance1 + momentum_balance2 + momentum_balance3

    governing_equation1 = Sxx -(C11*Exx+ C12*Eyy + C12*Ezz)
    
    governing_equation2 = Syy -(C12*Exx+ C11*Eyy + C12*Ezz)

    governing_equation3 = Szz -(C12*Exx+ C12*Eyy + C11*Ezz) 
 
    governing_equation4 = Sxy -C33*Exy  

    governing_equation5 = Sxz -C33*Exz
     
    governing_equation6 = Syz -C33*Eyz

    elastic_energy      = 0.5*(Exx*Sxx + Eyy*Syy + Ezz*Szz + 2.0*Exy*Sxy + 2.0*Exz*Sxz + 2.0*Eyz*Syz) # the mean elastic energy

    overall_loss        = governing_equation1 + governing_equation2 + governing_equation3 + governing_equation4 + governing_equation5 + governing_equation6 + elastic_energy + momentum_balance # the term 
    
    return overall_loss, governing_equation1, governing_equation2,governing_equation3,governing_equation4, governing_equation5, governing_equation6, elastic_energy, momentum_balance    
    
    
num_neurons =[40]*4 

def pinns_layers(nch_input, num_neurons):
    # desired number/dimension of output is 1
    layers = []
    last   = nch_input
    for nLay, nNeuron in enumerate(num_neurons):
#        print('nLay',    nLay)
#        print('nNeuron', nNeuron)
#        layers.append(torch.nn.Linear(last, nNeuron))
        layers.append(torch.nn.Conv1d(last, nNeuron, 1))
        layers.append(torch.nn.Tanh())
        
        last = nNeuron
    
#    layers.append(torch.nn.Linear(last,1))
    layers.append(torch.nn.Conv1d(last, 1, 1))    
    return layers

class PINNsNet(torch.nn.Module):
      def __init__(self,nch_input, num_neurons):    
          super().__init__()
          list_layers = pinns_layers(nch_input, num_neurons)
          self.layers = torch.nn.Sequential(*list_layers)
      
      def forward(self, inp):
          out = self.layers(inp)
          return out 




''' 
# the codes that test the PINNsNet module
test = PINNsNet(3,num_neurons)   
print(test) #
random_torch_tensor = torch.rand(5,3)
print(test(random_torch_tensor))
print(test(random_torch_tensor).shape)
'''



# the codes that test the torch.autograd#
#---------------------------------------#
'''
x = torch.ones(6, 1, requires_grad=True) # some dummy data
x.requires_grad= True
#x.requires_grad_(True) # both ways seem work to setting the requires_grad to be True
print(x.grad)
print('x.requires_grad', x.requires_grad)
v = x + 2
print('v.requires_grad', v.requires_grad)
y = v ** 2
print('y.requires_grad', y.requires_grad)
print('the shape of y', y.shape)
#dy_dx = torch.autograd.grad(outputs=y, inputs=x) # this cannot work
dy_dx  = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
print(dy_dx)
print(x.grad)
'''
#-------------------------------------#

# another example that checks the backward function
#---------------------------------------#
'''
x = torch.tensor([0.0, 2.0, 8.0],   requires_grad = True)
y = torch.tensor([5.0 , 1.0 , 7.0], requires_grad = True)
z = x*y
print(z.backward(torch.FloatTensor([1.0, 1.0, 1.0]))))
'''
#---------------------------------------#

###---------The following class is not used------------###
###---------The following class is not used------------###
'''
class PhysicsInformed(torch.nn.Module):
   def __init__(self, nch_input, num_neurons): 
      # ux, uy, uz are the known displacements (of the boundary points)
      super().__init__()
      
      # Loss
      self.num_neurons = [40]*4    
      self.pinns       = PINNsNet(nch_input,num_neurons)     
   def forward(self, x, y, z):
       # x, y and z are the (x,y,z) coordinates of the source point set to be deformed
       Ux = self.pinns([x,y,z])
       Uy = self.pinns([x,y,z])
       Uz = self.pinns([x,y,z])
       
       Sxx = self.pinns([x,y,z])
       Syy = self.pinns([x,y,z])
       Szz = self.pinns([x,y,z])
       Sxy = self.pinns([x,y,z])
       Sxz = self.pinns([x,y,z])
       Syz = self.pinns([x,y,z])
       
       # set the x,y,z to be 
       x.requires_grad= True
       y.requires_grad= True       
       z.requires_grad= True       
       
       Exx = torch.autograd.grad(Ux, x, grad_outputs=torch.ones_like(Ux))
       Eyy = torch.autograd.grad(Uy, y, grad_outputs=torch.ones_like(Uy))
       Ezz = torch.autograd.grad(Uz, z, grad_outputs=torch.ones_like(Uz))

       Exy = 0.5*(torch.autograd.grad(Ux, y, grad_outputs=torch.ones_like(Ux)) + torch.autograd.grad(Uy,x, grad_outputs=torch.ones_like(Uy)))
       Exz = 0.5*(torch.autograd.grad(Ux, z, grad_outputs=torch.ones_like(Ux)) + torch.autograd.grad(Uz,x, grad_outputs=torch.ones_like(Uz)))
       Eyz = 0.5*(torch.autograd.grad(Uy, z, grad_outputs=torch.ones_like(Uy)) + torch.autograd.grad(Uz,y, grad_outputs=torch.ones_like(Uz)))
       
       momentum_balance1 = torch.autograd.grad(Sxx, x, grad_outputs=torch.ones_like(Sxx)) + torch.autograd.grad(Sxy, y, grad_outputs=torch.ones_like(Sxy)) + torch.autograd.grad(Sxz, z, grad_outputs=torch.ones_like(Sxz))
       momentum_balance2 = torch.autograd.grad(Sxy, x, grad_outputs=torch.ones_like(Sxy)) + torch.autograd.grad(Syy, y, grad_outputs=torch.ones_like(Syy)) + torch.autograd.grad(Syz, z, grad_outputs=torch.ones_like(Syz))
       momentum_balance3 = torch.autograd.grad(Sxz, x, grad_outputs=torch.ones_like(Sxz)) + torch.autograd.grad(Syz, y, grad_outputs=torch.ones_like(Syz)) + torch.autograd.grad(Szz, z, grad_outputs=torch.ones_like(Szz))
       
       PINNs_Loss =  pinn_loss(Ux, Uy, Uz, ux_target, uy_target, uz_target, Sxx,Syy,Szz,Sxy,Sxz,Syz,Exx,Eyy,Ezz,Exy,Exz,Eyz, momentum_balance1,momentum_balance2, momentum_balance3)
       
      
       return 1  
''' 
###---------The above class is not used------------###          
###---------The above class is not used------------###           
          
          
          
          
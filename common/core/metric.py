import torch
import torch.nn as nn

class JointMSELoss(nn.Module):

    def __init__(self) -> None:
        self.criterion = nn.MSELoss()

    def forward(self,pred,hmap,joints_vis):
        assert pred.shape == hmap.shape

        batch_size = pred.size(0)
        num_joints = pred.size(1)

        heatmaps_preds = pred.reshape((batch_size,num_joints,-1)).split(1,1) # size,dim
        heatmaps_label = hmap.reshape((batch_size,num_joints,-1)).split(1,1)

        loss = 0
        for idx in range(num_joints):
            heatmap_preds = heatmaps_preds[idx].squeeze()
            heatmap_label = heatmaps_label[idx].squeeze()

            loss += 0.5*self.criterion(
                torch.mul(heatmap_preds,joints_vis[:,idx]),
                torch.mul(heatmap_label,joints_vis[:,idx])
            )
        return loss/num_joints

def easy_calc_loss(pred,labels,joints_vis):
    loss = JointMSELoss()
    hmap = labels[:,-1,:,:,:]
    return loss(pred,hmap,joints_vis)

import torchmetrics as metrics



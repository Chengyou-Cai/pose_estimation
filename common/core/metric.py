import torch
import torch.nn as nn

class JointMSELoss(nn.Module):

    def __init__(self) -> None:
        self.criterion = nn.MSELoss()

    def forward(self,preds,label,joints_vis):
        assert preds.shape == label.shape

        batch_size = preds.size(0)
        num_joints = preds.size(1)

        heatmaps_preds = preds.reshape((batch_size,num_joints,-1)).split(1,1)
        heatmaps_label = label.reshape((batch_size,num_joints,-1)).split(1,1)

        loss = 0
        for idx in range(num_joints):
            heatmap_preds = heatmaps_preds[idx].squeeze()
            heatmap_label = heatmaps_label[idx].squeeze()

            loss += 0.5*self.criterion(
                torch.mul(heatmap_preds,joints_vis[:,idx]),
                torch.mul(heatmap_label,joints_vis[:,idx])
            )
        return loss/num_joints

def easy_calc_loss(preds,labels,joints_vis):
    loss = JointMSELoss()
    return loss(preds,labels[:,-1,:,:,:],joints_vis)



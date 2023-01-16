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

import numpy as np
from torchmetrics import Metric
from common.inference import get_maxv_coord

class PCKm(Metric):
    
    @staticmethod
    def calc_dist(preds, label, normalize):
        preds = preds.astype(np.float32)
        label = label.astype(np.float32)

        dists = np.zeros((preds.shape[0], preds.shape[1]))
        
        for b in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if label[b, c, 0] > 1 and label[b, c, 1] > 1:
                    normed_preds = preds[b, c, :] / normalize[b]
                    normed_label = label[b, c, :] / normalize[b]
                    dists[b, c] = np.linalg.norm(normed_preds - normed_label)
                else:
                    dists[b, c] = -1
        return dists

    @staticmethod
    def calc_pck(dists,threshold=0.5):

        def reduce_on_batch(dist_j, threshold=0.5):
            dist_idxs = np.not_equal(dist_j,-1)
            if dist_idxs.sum() > 0:
                return np.less(dist_j[dist_idxs],threshold).sum()*1.0/dist_idxs.sum()
            else:
                return -1

        pck_joints = np.zeros(dists.shape[0])
        pck_mean = 0
        cnt = 0

        for j in range(dists.shape[0]):
            pck_joints[j] = reduce_on_batch(dists[j],threshold)
            if pck_joints[j] >=0 :
                pck_mean += pck_joints[j]
                cnt += 1
        pck_mean = pck_mean/cnt if cnt!= 0  else 0
        return pck_mean
        
    def __init__(self, threshold=0.5) -> None:
        super(PCKm,self).__init__()
        self.threshold = threshold
        self.add_state("distance", default=torch.tensor(0), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, label: torch.Tensor):
        assert preds.shape == label.shape
        batch_size, num_joints, heatmap_h, heatmap_w= preds.shape

        coord_preds, _ = get_maxv_coord(preds,batch_size,num_joints,heatmap_w)
        coord_label, _ = get_maxv_coord(label,batch_size,num_joints,heatmap_w)

        norm = np.ones((batch_size, 2)) * np.array([heatmap_h, heatmap_w]) / 10
        self.distance = self.calc_dist(coord_preds,coord_label,norm)

    def compute(self):
        trans_distance = np.transpose(self.distance)
        pck_joints, pck_mean = self.calc_pck(
            dists=trans_distance,threshold=self.threshold
            )
        return pck_mean





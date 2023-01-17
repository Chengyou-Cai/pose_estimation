import numpy as np

def get_maxv_coord(batch_heatmaps, batch_size, num_joints, heatmap_w):

    batch_heatmaps_ = batch_heatmaps.reshape((batch_size, num_joints, -1))
    
    batch_idxs = np.argmax(batch_heatmaps_.detach().cpu().numpy(), 2)
    batch_maxv = np.amax(batch_heatmaps_.detach().cpu().numpy(), 2)

    batch_idxs = batch_idxs.reshape((batch_size, num_joints, 1))
    batch_maxv = batch_maxv.reshape((batch_size, num_joints, 1))

    batch_coords = np.tile(batch_idxs, (1, 1, 2)).astype(np.float32)

    batch_coords[:, :, 0] = (batch_coords[:, :, 0]) % heatmap_w # x
    batch_coords[:, :, 1] = (batch_coords[:, :, 1]) // heatmap_w # y

    pred_mask = np.tile(np.greater(batch_maxv, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    batch_coords *= pred_mask
    return batch_coords, batch_maxv

from common.transform import inv_affine_transform

def get_coord_preds(batch_heatmaps,center,scale):
    assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    heatmap_h = batch_heatmaps.shape[2]
    heatmap_w = batch_heatmaps.shape[3]

    batch_coords, batch_maxv = get_maxv_coord(batch_heatmaps,batch_size,num_joints,heatmap_w)
    
    preds = batch_coords.copy()
    
    for i in range(batch_size):
        preds[i] = inv_affine_transform(
            batch_coords[i], 
            center[i], scale[i],
            (heatmap_w, heatmap_h)
            )
    return preds, batch_maxv





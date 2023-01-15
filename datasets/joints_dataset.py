import cv2
import copy
import logging
import numpy as np
import random
from torch.utils.data import Dataset

from common.transform import flip_skeleton
from common.transform import fit_affine_transform
from common.transform import get_affine_transform

class JointsDataset(Dataset):

    def __init__(self,stage,config,transforms=None) -> None:
        assert stage in ('train', 'valid', 'test')
        super(JointsDataset,self).__init__()

        self.stage = stage
        self.config = config
        self.transforms = transforms

        self.db = None
        self.num_joints = None
        self.flip_pairs = None

    def __len__(self):
        return len(self.db)

    def __getitem__(self, idx):
        image_data = copy.deepcopy(self.db[idx])
        
        image_path = image_data["image_path"]

        src_image = cv2.imread(image_path,cv2.IMREAD_COLOR)
        src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)

        joints = image_data["joints_3d"][:,:2]
        joints_vis = image_data["joints_3d"][:,-1].reshape(-1,1)
        
        # score = image_data["score"] if "score" in image_data else 1

        center = image_data["center"]
        scale = image_data["scale"] # w,h
        rotat = 0
        flip_ = False

        if scale[0] > self.config.MODEL.ASPECT_RATIO * scale[1]:
            scale[1] = scale[0] * 1.0 / self.config.MODEL.ASPECT_RATIO
        else:
            scale[0] = scale[1] * 1.0 * self.config.MODEL.ASPECT_RATIO

        scale[0] *= (1+self.config.DATA.BASE_EXT)
        scale[1] *= (1+self.config.DATA.BASE_EXT)
        if self.stage == "train":
            r = np.random.rand(2) if self.config.DATA.RAND_EXT else (1.0,1.0)
            scale[0] *= (1+self.config.DATA.TRAIN_X_EXT * r[0])
            scale[1] *= (1+self.config.DATA.TRAIN_Y_EXT * r[1])
        else:
            scale[0] *= (1+self.config.DATA.EVAL_X_EXT)
            scale[1] *= (1+self.config.DATA.EVAL_Y_EXT)

        # augmentation
        if self.stage == 'train':
            # # half body
            # if (np.sum(joints_vis[:, 0] > 0) > self.num_keypoints_half_body
            #     and np.random.rand() < self.prob_half_body):
            #     c_half_body, s_half_body = self.half_body_transform(
            #         joints, joints_vis)

            #     if c_half_body is not None and s_half_body is not None:
            #         center, scale = c_half_body, s_half_body

            # scale
            if self.config.DATA.AUGM_SCALE and random.random()<self.config.DATA.PROB_SCALE:
                sf = self.config.DATA.SCALE_FACTOR
                scale *= np.clip(1+np.random.randn()*sf,1-sf,1+sf)

            # rotat
            if self.config.DATA.AUGM_ROTAT and random.random()<self.config.DATA.PROB_ROTAT:
                rf = self.config.DATA.ROTAT_FACTOR
                rotat = np.clip(np.random.randn()*rf, -rf*2, rf*2)
            
            # flip_
            if self.config.DATA.AUGM_FLIP and random.random()<self.config.DATA.PROB_FLIP:
                joints, joints_vis = flip_skeleton(joints,joints_vis,self.flip_pairs)
                flip_ = True

        # ----------
        trans = get_affine_transform(center, scale, rotat, flip_, self.config.MODEL.INPUT_SHAPE)

        input_image = cv2.warpAffine(
            src_image,
            trans,
            (int(self.config.MODEL.INPUT_SHAPE[1]), int(self.config.MODEL.INPUT_SHAPE[0])),
            flags=cv2.INTER_LINEAR)

        if self.transforms:
            input_image = self.transforms(input_image)
        # ----------

        if self.stage != "test":
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = fit_affine_transform(joints[i, 0:2], trans)
                    if joints[i, 0] < 0 or joints[i, 0] > self.config.MODEL.INPUT_SHAPE[1] - 1 \
                        or joints[i, 1] < 0 or joints[i, 1] > self.config.MODEL.INPUT_SHAPE[0] - 1:
                        joints_vis[i, 0] = 0

            num_labels = len(self.config.DATA.GAUSSIAN_KERNELS)
            labels = np.zeros((num_labels,self.num_joints,*self.config.MODEL.OUTPUT_SHAPE))
            for i in range(num_labels):
                labels[i] = self.generate_joints_heatmap(joints, joints_vis, self.config.DATA.GAUSSIAN_KERNELS[i])
            return image_path, center, scale, input_image, labels, joints, joints_vis
        else:
            return image_path, center, scale, input_image

    def generate_joints_heatmap(self,joints,joints_vis,kernel):
        input_shape =  self.config.MODEL.INPUT_SHAPE
        output_shape = self.config.MODEL.OUTPUT_SHAPE

        heatmaps = np.zeros(
                (self.num_joints, *output_shape), dtype='float32')

        for i in range(self.num_joints):
            if joints_vis[i] < 1:
                continue
            target_y = joints[i, 1] * output_shape[0] / input_shape[0]
            target_x = joints[i, 0] * output_shape[1] / input_shape[1]
            
            heatmaps[i, int(target_y), int(target_x)] = 1
            heatmaps[i] = cv2.GaussianBlur(heatmaps[i], kernel, 0)
            maxi = np.amax(heatmaps[i])
            if maxi <= 1e-8:
                continue
            heatmaps[i] /= maxi / 255

        return heatmaps
    
    def _load_db(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

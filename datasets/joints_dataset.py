import cv2
import copy
import logging
import numpy as np
import random
from torch.utils.data import Dataset

from common.transforms import flip_joints
from common.transforms import affine_transform
from common.transforms import get_affine_transform

class JointsDataset(Dataset):

    def __init__(self,stage,config,transforms) -> None:
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
        
        image_id = image_data["image_id"]
        image_path = image_data["image_path"]

        data_numpy = cv2.imread(image_path,cv2.IMREAD_COLOR)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = image_data["joints_3d"][:,:2]
        joints_vis = image_data["joints_3d"][:,-1].reshape(-1,1)
        
        center = image_data["center"]
        scale = image_data["scale"] # w,h
        score = image_data["score"] if score in image_data else 1
        rotation = 0

        if scale[0] > self.config.MODEL.ASPECT_RATIO * scale[1]:
            scale[1] = scale[0] * 1.0 / self.config.MODEL.ASPECT_RATIO
        else:
            scale[0] = scale[1] * 1.0 * self.config.MODEL.ASPECT_RATIO

        scale[0] *= (1+self.config.DATA.BASE_EXT)
        scale[1] *= (1+self.config.DATA.BASE_EXT)
        
        if self.stage == "train":
            r = np.random.rand() if self.DATA.TRAIN.RAND_EXT else (1.0,1.0)
            scale[0] *= (1+self.config.DATA.TRAIN.X_EXT * r[0])
            scale[1] *= (1+self.config.DATA.TRAIN.Y_EXT * r[1])
        else:
            scale[0] *= (1+self.config.DATA.EVAL.X_EXT)
            scale[1] *= (1+self.config.DATA.EVAL.Y_EXT)

        # augmentation
        if self.stage == 'train':
            # # half body
            # if (np.sum(joints_vis[:, 0] > 0) > self.num_keypoints_half_body
            #     and np.random.rand() < self.prob_half_body):
            #     c_half_body, s_half_body = self.half_body_transform(
            #         joints, joints_vis)

            #     if c_half_body is not None and s_half_body is not None:
            #         center, scale = c_half_body, s_half_body
            
            # flip
            if random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = flip_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

            # scale & rotation
            sf = self.config.DATA.SCALE_FACTOR
            scale *= np.clip(1+np.random.randn()*sf,1-sf,1+sf)

            if random.random() <= 0.5:
                rf = self.config.DATA.ROTAT_FACTOR
                rotation = np.clip(np.random.randn()*rf, -rf*2, rf*2)

        # ----------
        trans = get_affine_transform(center, scale, rotation, self.config.MODEL.INPUT_SHAPE)

        input_image = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.config.MODEL.INPUT_SHAPE[1]), int(self.config.MODEL.INPUT_SHAPE[0])),
            flags=cv2.INTER_LINEAR)

        if self.transforms:
            input_image = self.transforms(input_image)
        # ----------

        if self.stage == "train":
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0:
                    joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
                    if joints[i, 0] < 0 or joints[i, 0] > self.config.MODEL.INPUT_SHAPE[1] - 1 \
                        or joints[i, 1] < 0 or joints[i, 1] > self.config.MODEL.INPUT_SHAPE[0] - 1:
                        joints_vis[i, 0] = 0

            num_labels = len(self.config.DATA.GAUSSIAN_KERNELS)
            labels = np.zeros(num_labels,self.num_joints,*self.config.MODEL.OUTPUT_SHAPE)
            for i in range(num_labels):
                labels[i] = self.generate_joints_heatmap(joints, joints_vis, self.config.DATA.GAUSSIAN_KERNELS[i])
            return input_image, joints_vis, labels
        else:
            return input_image, score, center, scale, image_id

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

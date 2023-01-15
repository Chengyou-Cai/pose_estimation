import os
import json
import numpy as np
from datasets.joints_dataset import JointsDataset

class MPIIDataset(JointsDataset):
    @property
    def keypoint_list(self):
        return [
            "r ankle", "r knee", "r hip", "l hip", "l knee", "l ankle", 
            "pelvis", "thorax", "upper neck", "head top", "r wrist", 
            "r elbow", "r shoulder", "l shoulder", "l elbow", "l wrist"
        ]

    @property
    def skeleton_list(self):
        return np.array([
            [0, 1], [1, 2], [2, 6], [3, 4], [3, 6], [4, 5], [6, 7],
            [7, 8], [8, 9], [8, 12], [8, 13], [10, 11], [11, 12], [13, 14], 
            [14, 15]
        ])+1

    @property
    def kpgt_anns_file_path(self):
        return os.path.join(self.config.DATA.ROOT,"annots",f"{self.stage}.json")

    @staticmethod
    def json2obj(fpath):
        with open(fpath) as json_file:
            obj = json.load(json_file)
        return obj

    def __init__(self, stage, config, transforms=None) -> None:
        super(MPIIDataset,self).__init__(stage, config, transforms)

        self.anns = self.json2obj(self.kpgt_anns_file_path)

        self.db = self._load_db()
        self.num_joints = len(self.keypoint_list)
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    
    def get_image_path(self,fname):
        return os.path.join(self.config.DATA.ROOT,"images",fname)

    def _load_db(self):
        db = list()

        for a in self.anns:
            image_id = a["image"].split('.')[0]
            image_path = self.get_image_path(fname=a["image"])

            center = np.array(a['center'], dtype=np.float32)
            scale =  np.array([a['scale'], a['scale']], dtype=np.float32)
            
            if center[0] != -1:
                center[1] = center[1] + 15 * scale[1]
            center -= 1

            if self.stage == 'test':
                joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            else:
                joints = np.array(a['joints'], dtype=np.int32) - 1 # joints -= 1
                joints_vis = np.array(a['joints_vis'], dtype=np.int32).reshape(-1,1)*2 # joints_vis = joints_vis.reshape(-1, 1)*2
                joints_3d = np.concatenate((joints, joints_vis), axis=1)
            
            db.append({
                "image_id": image_id,
                "image_path": image_path,
                "joints_3d": joints_3d,
                "center": center,
                "scale": scale
            })
        return db

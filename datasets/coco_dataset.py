import os
import numpy as np
from datasets.joints_dataset import JointsDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class COCODataset(JointsDataset):
    
    @property
    def coco_keypoint_dict(self):
        '''
        "keypoints": {
            0: "nose",
            1: "left_eye",
            2: "right_eye",
            3: "left_ear",
            4: "right_ear",
            5: "left_shoulder",
            6: "right_shoulder",
            7: "left_elbow",
            8: "right_elbow",
            9: "left_wrist",
            10: "right_wrist",
            11: "left_hip",
            12: "right_hip",
            13: "left_knee",
            14: "right_knee",
            15: "left_ankle",
            16: "right_ankle"
        }
        '''
        return self.coco.loadCats(self.coco.getCatIds())[0]["keypoints"]

    @property
    def coco_skeleton_list(self):
        '''
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
            [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        '''
        return self.coco.loadCats(self.coco.getCatIds())[0]["skeleton"]

    @property
    def gtkp_anns_file_path(self):
        prefix = "person_keypoints" if self.stage!="test" else "image_info"
        return os.path.join(
            self.config.DATA.ROOT,"annotations",f"{prefix}_{self.stage}2017.json"
            )

    @property
    def gtkp_imgs_idxs(self):
        return self.coco.getImgIds()

    @property
    def gtkp_anns_idxs(self):
        return self.coco.getAnnIds(iscrowd=False)

    def __init__(self, stage, config, transforms) -> None:
        super(COCODataset,self).__init__(stage, config, transforms)

        self.coco = COCO(self.gtkp_anns_file_path)
        self.db = self._load_db()

        self.num_joints = len(self.coco_keypoint_dict)
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]

    def _load_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
         """ ground truth bbox and keypoints """
         gt_db = list()
         for image_id in self.gtkp_imgs_idxs:
            gt_db.extend(self._load_coco_keypoint_annotations_kernal(image_id=image_id))
    
    def _load_coco_keypoint_annotations_kernal(self,image_id):
        image_info = self.coco.loadImgs(ids=[image_id])[0]
        image_path = self.image_path_from_image_info(image_info)
        image_w = image_info["width"]
        image_h = image_info["height"]

        annids = self.coco.getAnnIds(ids=[image_id],iscrowd=False)
        anns = self.coco.loadAnns(annids)

        clean_anns = list()
        for ann in anns:
            if max(ann["keypoints"]) <= 0 or ann['num_keypoints'] == 0:
                continue
            x, y, w, h = ann["bbox"]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((image_w - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((image_h - 1, y1 + np.max((0, h - 1))))
            if ann['area'] > 0 and x2 >= x1 and y2 >= y1:
                ann['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                clean_anns.append(ann)
        anns = clean_anns

        store = list()
        for ann in anns:
            joints_3d = np.array(ann["keypoints"]).reshape(-1,3)
            # if np.sum(joints_3d[:,-1]>0) < self.load_min_num_keypoints:
            #     continue
            bbox = np.array(ann["clean_bbox"])
            center, scale = self._bbox_to_center_and_scale(bbox)

            store.append({
                "image_id": image_id,
                "image_path": image_path,
                "joints_3d": joints_3d,
                "center": center,
                "scale": scale
            })
        return store
    
    def image_path_from_image_info(self,image_info):
        image_name = image_info["file_name"]
        return os.path.join(self.config.DATA.ROOT,f"{self.stage}2017",image_name)

    @staticmethod
    def _bbox_to_center_and_scale(bbox,pixel_std=200):
        x, y, w, h = bbox[:4] # (x,y) upper-left

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],dtype=np.float32)

        return center, scale

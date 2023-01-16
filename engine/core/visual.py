import cv2
import numpy as np

def bbox2cs(bbox,pixel_std=200):
    x, y, w, h = bbox[:4] # (x,y) upper-left

    center = np.zeros(2, dtype=np.float32)
    center[0] = x + w / 2.0
    center[1] = y + h / 2.0

    scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std],dtype=np.float32)

    return center, scale

def cs2bbox(center,scale,pixel_std=200):
    x, y = center
    w, h = scale * pixel_std

    lt = (int(x - w / 2.0), int(y - h / 2.0))
    rb = (int(x + w / 2.0), int(y + h / 2.0))
    return np.array(lt),np.array(rb)

def draw_bounding_box(image_path,center=None,scale=None):
    src_image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
    lt, rb = cs2bbox(center,scale)

    cv2.rectangle(src_image, lt, rb, color=(0, 0, 255), thickness=5)

    return src_image

def draw_joints_heatmap(labels,map_size=None):

    label = labels[-1]
    joints_heatmap = np.zeros((label.shape[1],label.shape[2]),dtype="float32")

    for joint_heatmap in label:
        r, c = np.nonzero(joint_heatmap)
        joints_heatmap[r,c] = joint_heatmap[r,c]

    if map_size:
        joints_heatmap = cv2.resize(joints_heatmap, map_size, interpolation=cv2.INTER_NEAREST)
    
    return joints_heatmap

def draw_skeleton(image,joints,joints_vis,skeleton,score=None):
    
    num_joints = len(joints)
    color = np.random.randint(0,256,(num_joints,3)).tolist()
    for i  in range(num_joints):
        if joints_vis[i,0]>0 and (joints[i, 0] > 0 and joints[i, 1] > 0):
            cv2.circle(image, tuple(joints[i, :2]), 2, tuple(color[i]), 2)

    if score:
        cv2.putText(image, score, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,(128, 255, 0), 2)

    def draw_line(img, p1, p2):
        c = (0, 0, 255)
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv2.line(img, tuple(p1), tuple(p2), c, 2)

    for pair in skeleton:
        draw_line(image, joints[pair[0] - 1], joints[pair[1] - 1])

    return image


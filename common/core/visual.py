import cv2
import numpy as np

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


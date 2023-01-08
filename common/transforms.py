import numpy as np
import cv2

def flip_back(output_flipped, matched_pairs):
    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_pairs:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped

def flip_joints(joints, joints_vis, width, matched_pairs):
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in matched_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis

def flip_skeleton(joints, joints_vis, matched_pairs):

    for pair in matched_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis

def fit_affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_affine_transform(center, scale, rotat=0, flip_=False, output_size=None, pixel_std=200):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * pixel_std
    rotat_rad = np.pi * rotat / 180

    src_w, src_h = scale_tmp[0], scale_tmp[1]
    dst_w, dst_h = output_size[1], output_size[0]
    
    src_dir = get_dir([0, src_w * 0.5], rotat_rad)
    dst_dir = np.array([0, dst_w * 0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    
    src[0, :] = center #
    src[1, :] = center + src_dir
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])

    dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5]) #
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if flip_:
        dst[:,0] = dst_w - dst[:,0] - 1

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32) # 向量垂直

def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
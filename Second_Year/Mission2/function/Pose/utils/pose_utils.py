from .pose_gt import *


def manage_duplicate_pose(cad_name, pose_idx):
    # change pose index to smaller index if the renderings are equal
    try:
        duplicate_poses = DUPLICATE_POSES_DICT[cad_name]
        pose_idx = duplicate_poses[np.where(duplicate_poses == pose_idx)[0][0]][0]
        return pose_idx
    except:
        return pose_idx

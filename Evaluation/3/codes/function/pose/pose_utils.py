from scipy.spatial.transform import Rotation as Rot # quat (x, y, z, w)

from .pose_gt import *


def manage_duplicate_pose(cad_name, pose_idx):
    # change pose index to smaller index if the renderings are equal
    try:
        duplicate_poses = DUPLICATE_POSES_DICT[cad_name]
        pose_idx = duplicate_poses[np.where(duplicate_poses == pose_idx)[0][0]][0]
        return pose_idx
    except:
        return pose_idx


def logit_to_RT(logit):
	quaternion = logit[:4]
	translation = logit[4:]
	R_pred = Rot.from_quat(np.roll(quaternion, -1)).as_matrix()
	T_pred = translation.reshape(3, 1)
	RT_pred = np.hstack((R_pred, T_pred))
	return RT_pred
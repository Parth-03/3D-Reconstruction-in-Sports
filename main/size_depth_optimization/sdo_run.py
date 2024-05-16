import torch
import numpy as np
import tqdm

from util.depth import read_all_joints, project_joints_to_img

from losses import MSELoss, keypoint_loss

debug = False
debug1 = False

"""
multi_hmr_persons: person[]
person = {
            # Detection
            'scores': scores_det[i], # detection scores
            'loc': out['loc'][i], # 2d pixel location of the primary keypoints
            
            # SMPL-X params
            'transl': out['transl'][i], # from the primary keypoint i.e. the head
            'transl_pelvis': out['transl_pelvis'][i], # of the pelvis joint
            'rotvec': out['rotvec'][i],
            'expression': out['expression'][i],
            'shape': out['shape'][i],
            
            # SMPL-X meshs
            'verts_smplx': out['verts_smplx_cam'][i],
            'j3d_smplx': out['j3d'][i],
            'j2d_smplx': out['j2d'][i],
        }

Required format from original paper (output of precompute process):
        data = {
            # needed for get_relative_nomalized_plane_v3; not using for this project
            "predicted_disparity": predicted_disparity, 
            "road": road,
            "h": h,
            "w": w,
            
            # used in joint_opt_use_smpl_for_reproj
            "joint_names": joint_names, # json file names ~ j3d_smplx, transl
            
            # not used
            "glob_kpts_all": glob_kpts_all,  # these are reprojected from the estimations directly. shape=[num_people, num_points, 3]. ~ j2d_smplx
            "persons_mask": persons_mask, 
        }
"""


def run_size_depth_opt(multi_hmr_output, img_size):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # setup args from demo
    n_iters = 2000

    results = []

    num_images = len(multi_hmr_output)

    for ind, humans_arr in tqdm.tqdm(enumerate(multi_hmr_output), total=num_images):
        jt_arr = [
            {"joints_3d": h["j3d_smplx"], "translation": h["transl"]}
            for h in humans_arr
        ]

        result_obj = joint_opt_use_smpl_for_reproj(
            jt_arr,
            n_iters,
            img_size,
        )
        results.append(result_obj)
    print("finished")
    return results


def joint_opt_use_smpl_for_reproj(
    jt_arr,
    n_iters,
    img_size,
):
    """ """

    criterion = MSELoss(reduction="none")

    # img_shape = torch.Tensor([832, 512])[None]
    img_shape = torch.Tensor([img_size, img_size])[None]

    dummy_trans = torch.Tensor([[0.0, 0.0, 0.0]]).cuda()

    # read all joints
    _, joints_trans_all, trans_all = read_all_joints(jt_arr)
    joints_trans_all_t = torch.Tensor(joints_trans_all).cuda()
    all_joints_projected = project_joints_to_img(
        joints_trans_all_t, img_shape, dummy_trans
    )

    # a bit random
    ref_person = 1

    smpl_idxs_ = np.arange(0, len(all_joints_projected))
    idx_ref_person = np.where(smpl_idxs_ == ref_person)
    smpl_idxs = np.delete(smpl_idxs_, idx_ref_person)
    joints_smpl_3d_sorted = joints_trans_all_t[smpl_idxs]
    sorted_gt_kpts = all_joints_projected[smpl_idxs]
    ones_t = torch.ones_like(sorted_gt_kpts[:, :, 0])
    gt_keypoints = torch.cat([sorted_gt_kpts, ones_t[..., None]], 2)

    ankles = joints_smpl_3d_sorted[:, 0].cpu().numpy()

    # intersection btw trans line and plane, shift to plane
    trans_dir = ankles
    mod = np.linalg.norm(trans_dir, axis=1)
    trans_dir_ = trans_dir / mod[:, None]

    # model assumes 24 joints per person. Multi hmr has a lot more (over 120)
    root_joint_t = joints_smpl_3d_sorted[:, 14, None, :]
    joints_p1_t = joints_smpl_3d_sorted

    # this are the ones to optimize
    # trans_to_inter and trans_iter_t depend on the output of get_relative_nomalized_plane_v3, which we're omitting out
    # We just start from a random tensor, not from trans_to_inter and trans_iter_t
    ankles_shape = ankles.shape
    trans_to_inter = np.random.rand(*ankles_shape)
    trans_iter_t = torch.tensor(trans_to_inter, device="cuda")

    opt_trans = torch.tensor(
        trans_to_inter[:, None, :],
        requires_grad=True,
        device="cuda",
        dtype=torch.float32,
    )
    opt_scale = torch.ones_like(
        trans_iter_t[:, 0, None, None],
        requires_grad=True,
        device="cuda",
        dtype=torch.float32,
    )

    # create init optimizer
    lr = 0.01
    optimizer = torch.optim.Adam([opt_scale, opt_trans], lr=lr)
    print_every = 50000
    loss_reprojection = 10000
    loss = 10000
    max_iters = n_iters
    w_loss_plane = 0

    crit = 1.0
    n_people = len(opt_scale)
    i = 0

    force = True
    while loss > crit or force:
        i += 1
        optimizer.zero_grad()
        # scale and trans
        trans_joints_p1_t = joints_p1_t + opt_trans
        scaled_joints_p1_t = (
            opt_scale * (trans_joints_p1_t - root_joint_t) + root_joint_t
        )
        joints_projected = project_joints_to_img(
            scaled_joints_p1_t, img_shape, dummy_trans
        )

        # keypoint_loss assumes 24 joints. Multi hmr has a lot more (127)
        loss_reprojection, loss_raw = keypoint_loss(
            joints_projected, gt_keypoints, criterion
        )

        loss = loss_reprojection  # without loss from plane estimation

        # backprop
        loss.backward()
        optimizer.step()

        if i % print_every == 0:
            print(
                "loss={}, loss_plane={}, loss_reproj={}".format(
                    loss,
                    w_loss_plane,
                    loss_reprojection,
                )
            )

        if i > max_iters:
            break

    new_scale = opt_scale.clone().detach()
    new_trans = opt_trans.clone().detach()
    final_joints_p1_t = joints_p1_t + new_trans
    final_joints_p1_t = new_scale * (final_joints_p1_t - root_joint_t) + root_joint_t
    final_joints_projected = project_joints_to_img(
        final_joints_p1_t, img_shape, dummy_trans
    )

    # scales_one = np.ones([n_people + 1], dtype=np.float32)
    scales = np.ones([n_people + 1], dtype=np.float32)
    scales[smpl_idxs] = new_scale[:, 0, 0].cpu().numpy()

    translations = np.zeros([n_people + 1, 1, 3], dtype=np.float32)
    translations[smpl_idxs] = new_trans.cpu().numpy()

    # more than 24 joints from multi hmr
    num_joints = 127
    optim_joints_proj = np.zeros([n_people + 1, num_joints, 2], dtype=np.float32)
    optim_joints_proj[smpl_idxs] = final_joints_projected.cpu().numpy()
    all_joints_projected = all_joints_projected.cpu().numpy()
    optim_joints_proj[ref_person] = all_joints_projected[ref_person]
    optim_joints_3d = np.zeros([n_people + 1, num_joints, 3], dtype=np.float32)
    optim_joints_3d[smpl_idxs] = final_joints_p1_t.cpu().numpy()
    optim_joints_3d[ref_person] = joints_trans_all[ref_person]

    translations_final = trans_all + scales[:, None] * translations.squeeze()

    opt_info = {
        "scale": [scales.tolist()],
        "translations": translations_final.tolist(),
        "joints2d": optim_joints_proj.tolist(),
        "joints3d": optim_joints_3d.tolist(),
    }

    return opt_info

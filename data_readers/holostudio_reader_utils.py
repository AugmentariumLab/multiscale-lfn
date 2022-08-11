import os

import cv2
import numpy as np

def get_video_details(video_path):
    """Find details about a video.

    Args:
        video_path: Path of the video.

    Returns:
        Dictionary of video properties.

    """
    properties = {}
    cap = cv2.VideoCapture(video_path)
    properties["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    properties["fps"] = int(cap.get(cv2.CAP_PROP_FPS))
    properties["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    properties["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return properties


# Copied from colmap_read_model.py
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def _compute_near_far_depth(pts_file, poses, fallback=(1, 10)):
    if not os.path.exists(pts_file):
        print("PTS File does not exit, returning fallback depth")
        all_depths = np.zeros((poses.shape[-1], 2))
        all_depths[:, 0] = fallback[0]
        all_depths[:, 1] = fallback[1]
        return all_depths

    with open(pts_file, "r") as f:
        all_points = f.readlines()
    all_points = filter(lambda x: x[0] != "#", all_points)
    all_points = map(lambda x: x.split(), all_points)
    all_points = [list(map(float, x)) for x in all_points]

    vis_arr = []
    for k in range(len(all_points)):
        cams = [0] * poses.shape[-1]
        for ind in all_points[k][8::2]:
            if ind - 1 > len(cams):
                raise ValueError(
                    'ERROR: the correct camera poses for current points cannot be accessed')
            cams[int(ind - 1)] = 1
        vis_arr.append(cams)

    pts_arr = np.array([x[1:4] for x in all_points])
    vis_arr = np.array(vis_arr)

    zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2, 0, 1]
                                                         ) - poses[:3, 3:4, :]) * poses[:3, 2:3, :], 0)

    all_depths = np.zeros((poses.shape[-1], 2))
    for i in range(poses.shape[-1]):
        vis = vis_arr[:, i]
        zs = zvals[:, i]
        zs = zs[vis == 1]
        close_depth, inf_depth = np.percentile(zs, .1), np.percentile(zs, 99.9)
        all_depths[i, 0] = close_depth
        all_depths[i, 1] = inf_depth

    print("Depth range:", np.mean(all_depths, axis=0),
          np.min(all_depths), np.max(all_depths))
    return all_depths


def load_poses(calibration_dir, factor):
    camera_txt_path = os.path.join(calibration_dir, "cameras.txt")
    with open(camera_txt_path, "r") as f:
        cameras = f.readlines()
    cameras = filter(lambda x: x[0] != "#", cameras)
    cameras = map(lambda x: x.strip("\n"), cameras)
    cameras = map(lambda x: x.split(" "), cameras)
    cameras = list(cameras)
    h, w, f = float(cameras[0][3]), float(cameras[0][2]), float(cameras[0][4])
    hwf = np.array([h, w, f]).reshape([3, 1])

    images_txt_path = os.path.join(calibration_dir, "images.txt")
    with open(images_txt_path, "r") as f:
        images_txt = f.readlines()
    images_txt = filter(lambda x: x[0] != "#", images_txt)
    images_txt = map(lambda x: x.strip("\n"), images_txt)
    images_txt = map(lambda x: x.split(" "), images_txt)
    images_txt = list(images_txt)[::2]
    images_txt = sorted(images_txt, key=lambda x: x[9])
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    rotations = [[float(x[1]), float(x[2]), float(x[3]), float(x[4])]
                 for x in images_txt]
    rotations = np.stack([qvec2rotmat(x) for x in rotations], axis=0)
    # rotations = Rotation.from_quat(rotations).as_matrix()
    translations = [[float(x[5]), float(x[6]), float(x[7])]
                    for x in images_txt]
    translations = np.array(translations)
    w2c_mats = np.concatenate((rotations, translations[:, :, None]), axis=2)
    bottom = np.array([0, 0, 0, 1])
    w2c_mats = np.concatenate(
        (w2c_mats, np.tile(bottom[None, None, :], [w2c_mats.shape[0], 1, 1])), axis=1)

    # Copied from LLFF
    w2c_mats = np.stack(w2c_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)
    poses_arr = c2w_mats[:, :3, :4].transpose([1, 2, 0])
    poses_arr = np.concatenate(
        [poses_arr, np.tile(hwf[..., None], [1, 1, poses_arr.shape[-1]])], 1)

    # must switch to [-u, r, -t] from [r, -u, t], NOT [r, u, -t]
    poses_arr = np.concatenate(
        [poses_arr[:, 1:2, :], poses_arr[:, 0:1, :], -poses_arr[:, 2:3, :],
         poses_arr[:, 3:4, :], poses_arr[:, 4:5, :]],
        1)

    depths = _compute_near_far_depth(os.path.join(
        calibration_dir, "points3D.txt"), poses_arr)

    poses_arr = poses_arr.transpose((2, 0, 1))
    poses_arr = poses_arr.reshape(
        (poses_arr.shape[0], poses_arr.shape[1] * poses_arr.shape[2]))
    poses_arr = np.concatenate((poses_arr, depths), axis=1)

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    poses[:2, 4, :] = np.array([h // factor, w // factor]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor
    return poses, bds


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def look_at(camera, up, target):
    forward = normalize(camera - target)
    # up_normalized = normalize(up)
    right = normalize(np.cross(up, forward))
    up = normalize(np.cross(forward, right))
    m = np.stack([right, up, forward, camera], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N, target=None):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    center_position = c2w[:, 3]
    if target is None:
        target = center_position

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -
            np.sin(theta * zrate), 1.0]) * rads
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        # render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
        render_poses.append(np.concatenate([look_at(c, up, target), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    def add_row_to_homogenize_transform(p):
        r"""Add the last row to homogenize 3 x 4 transformation matrices."""
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]),
                        [p.shape[0], 1, 1])], 1
        )

    # p34_to_44 = lambda p: np.concatenate(
    #     [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    # )

    p34_to_44 = add_row_to_homogenize_transform

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array(
            [radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(
            poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def get_video_frame(video_path, frame_num, factor):
    attempts = 0
    capture = cv2.VideoCapture(video_path)
    ret = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    if not ret:
        print("Capture set failed", ret)
    ret, frame = capture.read()
    while not ret and attempts < 10:
        ret, frame = capture.read()
        attempts += 1
    if not ret:
        print("Error loading frame", ret)
        capture.release()
        exit(0)
    capture.release()
    frame = frame[:, :, ::-1] / 255
    if factor is not 1:
        frame = cv2.resize(frame, (0, 0), fx=1 / factor, fy=1 / factor)
    return frame


def frame_num_extractor(x): return int(os.path.splitext(os.path.basename(x))[0])

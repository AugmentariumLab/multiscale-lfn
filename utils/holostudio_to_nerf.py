import os
import json
import shutil

import tqdm
from distutils.dir_util import copy_tree

import numpy as np
import cv2

DATASET = "DavidDataset0113Config5v3"
CAMERA_FILENAMES = (
    "cameraParametersColmapLidarPriorCameraCoversPartiallyInstalled.txt",
    "cameraParametersColmapAlignedNew.txt",
)
yz_flip_3 = np.array(((1, 0, 0,),
                      (0, 0, 1,),
                      (0, -1, 0)))
yz_flip_3 = np.array(((1, 0, 0),
                      (0, -1, 0),
                      (0, 0, -1))) @ yz_flip_3
r1 = np.array(((1, 0, 0,),
               (0, -1, 0,),
               (0, 0, -1)))
translation_scale = 1.0 * np.identity(3)
factor = 8

original_dataset_path = f"datasets/{DATASET}"
nerf_dataset_path = f"datasets/m_nerf/{DATASET}"
images_path = os.path.join(nerf_dataset_path, "train")
despilled_frames_dir = os.path.join(original_dataset_path, "DespilledFrames")

if not os.path.isdir(original_dataset_path):
    raise ValueError("Original dataset not found")
transforms_path = [os.path.join(original_dataset_path, x)
                   for x in CAMERA_FILENAMES
                   if os.path.isfile(os.path.join(original_dataset_path, x))]
if not transforms_path:
    raise ValueError("Camera parameters not found")
while os.path.isdir(images_path):
    shutil.rmtree(images_path)
os.makedirs(images_path, exist_ok=True)
transforms_path = transforms_path[0]
with open(transforms_path, "r") as f:
    transforms = json.load(f)

focal = (transforms[0]["intrinsics"][0] + transforms[0]["intrinsics"][5]) / 2
input_path = os.path.join(despilled_frames_dir, "Board101Camera0", "com", "00000.png")
height, width, _ = cv2.imread(input_path).shape
camera_angle_x = 2.0 * np.arctan(.5 * width / focal)
new_transforms = {
    "fl_x": transforms[0]["intrinsics"][0],
    "fl_y": transforms[0]["intrinsics"][5],
    "cx": transforms[0]["intrinsics"][8],
    "cy": transforms[0]["intrinsics"][9],
    "w": width,
    "h": height,
    "frames": []
}
camera_names = sorted(os.listdir(despilled_frames_dir))
for i, camera in tqdm.tqdm(enumerate(camera_names), total=len(camera_names)):
    if i % 24 > 0:
        continue
    frame_num = i + 1
    input_path = os.path.join(despilled_frames_dir, camera, "com", "00000.png")
    output_path = os.path.join(images_path, "%04d.png" % (i + 1))
    if True or not os.path.isfile(output_path):
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, image)

    new_transform = np.array(transforms[i]["transform"], dtype=np.float32).reshape(
        (4, 4)).transpose()
    rotation = new_transform[:3, :3].transpose()
    camera_positions = -rotation @ new_transform[:3, 3]
    new_transform = np.concatenate(
        (rotation, camera_positions[:, None]), axis=1)
    new_transform = np.concatenate((new_transform, np.array(
        (0, 0, 0, 1), dtype=np.float32)[None]), axis=0)

    camera_rot = new_transform[:3, :3] @ r1
    camera_trans = new_transform[:3, 3]
    camera_rot = yz_flip_3 @ camera_rot
    camera_trans = translation_scale @ yz_flip_3 @ camera_trans
    new_matrix_world = np.concatenate((camera_rot, camera_trans[:, None]), axis=1)
    new_matrix_world = np.concatenate((new_matrix_world, np.array((0, 0, 0, 1))[None]), axis=0)
    new_transform = new_matrix_world

    intrinsics = np.array(transforms[i]["intrinsics"], dtype=np.float32).reshape(
        (4, 4)).transpose()[:3, :3]

    x_fov = 2.0 * (180 / np.pi) * np.arctan(width / (2.0 * intrinsics[0, 0]))
    y_fov = 2.0 * (180 / np.pi) * np.arctan(height / (2.0 * intrinsics[1, 1]))

    new_transform = new_transform.tolist()
    new_transforms["frames"].append({
        "file_path": os.path.join("train", "%04d.png" % (i + 1)),
        "transform_matrix": new_transform,
        "x_fov": x_fov,
        "y_fov": y_fov,
        "principal_x": intrinsics[0, 2] / width,
        "principal_y": intrinsics[1, 2] / height
    })
transforms_output_path = os.path.join(nerf_dataset_path, "transforms_train.json")
with open(transforms_output_path, "w") as f:
    json.dump(new_transforms, f, indent=2)

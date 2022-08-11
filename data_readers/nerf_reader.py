import glob
import json
import re
from typing import Tuple, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

from data_readers.holostudio_reader_utils import *

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')
IMAGES = "images"
VIDEO_FRAMES = "video_frames"
CAMERA_PARAMETER_FILENAMES = (
    "transforms_train.json",
)


class NerfSyntheticDataset(Dataset):

    def __init__(self, basedir: str, factor: int = 8, recenter: bool = True, bd_factor: float = 0.75,
                 spherify: bool = False, load_saliency: bool = True, max_frames: int = 30,
                 renderposes_height: float = 0.4,
                 resize_interpolation: int = cv2.INTER_AREA,
                 mip_factors: Sequence[int] = (), dropout_poses: Sequence[int] = (),
                 include_dropout_only: bool = False):
        """Initialize a holostudio dataset.

        Args:
            basedir: Base directory for the dataset.
            factor: Resize factor.
            recenter: Recenter the dataset.
            bd_factor: Unknown
            spherify: Unknown
            load_saliency: Load saliency files.
            max_frames: Max number of frames (for videos).
        """
        self.basedir = basedir
        self.factor = factor
        self.recenter = recenter
        self.bd_factor = bd_factor
        self.spherify = spherify
        self.load_saliency = load_saliency
        self.max_frames = max_frames
        self.dataset_type = None
        self.resize_interpolation = resize_interpolation
        self.mip_factors = mip_factors
        self.dropout_poses = set(dropout_poses)
        self.include_dropout_only = include_dropout_only
        self.renderposes_height = renderposes_height

        ret = self._parse_images()
        if ret is not None:
            self.dataset_type = IMAGES
            (self.len, self.images_dir, self.images_list, self.original_height, self.original_width) = ret
            self.num_frames = 1
            self.num_views = self.len
        if self.dataset_type is None:
            raise ValueError("Could not recognize dataset type")
        self.height = self.original_height // self.factor
        self.width = self.original_width // self.factor

        self.using_processed_camera_parameters = any(
            os.path.isfile(os.path.join(self.basedir, x)) for x in CAMERA_PARAMETER_FILENAMES)
        if self.using_processed_camera_parameters:
            self.transforms, self.intrinsics, self.distortions = self._load_calibration_file()
            self.camera_params_to_rays(torch.tensor(self._get_resized_intrinsics(0)),
                                       torch.tensor(self.transforms[0]),
                                       self.height, self.width)

            pca = PCA(2)
            pca.fit(self.transforms[:, :3, 3])
            up = -np.cross(pca.components_[0], pca.components_[1])
            up = up / np.linalg.norm(up)
            self.render_poses_up = up
            self.render_poses_target = np.mean(self.transforms[:, :3, 3], axis=0) - self.renderposes_height * up
            self.render_poses_rotation_mat = np.stack((pca.components_[0], pca.components_[1], up), axis=1)
        else:
            self.poses, self.bds = self._load_poses()
            hwf = self.poses[0, :3, -1]
            self.hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
            self.focal = hwf[2]
        self.render_poses = {}

    def _parse_images(self):
        image_dir_names = ["train"]
        images_dir = [os.path.join(self.basedir, x) for x in image_dir_names]
        images_dir = [x for x in images_dir if os.path.isdir(x)]
        if not images_dir:
            return None
        images_dir = images_dir[0]
        number_extract = re.compile(r'r_(\d+)')
        is_color_image = re.compile(r'r_(\d+)\.')
        images_dir_files = [x for x in os.listdir(images_dir) if os.path.splitext(x)[
            1] in IMAGE_EXTENSIONS and is_color_image.search(x)]
        if number_extract.search(images_dir_files[0]):
            images_list = sorted(images_dir_files, key=lambda x: int(
                number_extract.search(x)[1]))
        else:
            images_list = sorted(images_dir_files)
            raise ValueError("Not sorting by image number")
        if self.dropout_poses and self.include_dropout_only:
            images_list = [x for i, x in enumerate(images_list) if i in self.dropout_poses]
        elif self.dropout_poses:
            images_list = [x for i, x in enumerate(images_list) if i not in self.dropout_poses]
        self_len = len(images_list)
        if not images_list:
            return None

        image = plt.imread(os.path.join(images_dir, images_list[0]))
        height = image.shape[0]
        width = image.shape[1]
        return self_len, images_dir, images_list, height, width

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if self.dataset_type == IMAGES:
            return self._getitem_images(item)
        else:
            raise ValueError(f"Unknown dataset type {str(self.dataset_type)}")

    def _get_resized_intrinsics(self, item):
        intrinsics = np.copy(self.intrinsics[item])
        intrinsics[:2, :] = intrinsics[:2, :] / self.factor
        return intrinsics

    def _getitem_images(self, item):
        factor = self.factor
        image_name = self.images_list[item]
        image = full_image = plt.imread(os.path.join(self.images_dir, image_name)).astype(np.float32)
        if factor != 1:
            image = cv2.resize(image, (0, 0), fx=1 / factor, fy=1 / factor, interpolation=self.resize_interpolation)
        height, width, channels = image.shape
        return_data = {
            "index": item,
            "image": torch.tensor(image, dtype=torch.float32),
            "t": torch.tensor(0.0, dtype=torch.float32)
        }
        if channels == 4:
            mask = (image[:, :, 3] > 1e-5).astype(np.float32)
            return_data["mask"] = torch.tensor(mask, dtype=torch.float32)
        if self.using_processed_camera_parameters:
            return_data["transform"] = torch.tensor(self.transforms[item])
            return_data["intrinsics"] = torch.tensor(self._get_resized_intrinsics(item))
            if len(self.distortions) > 0:
                return_data["distortion"] = torch.tensor(self.distortions[item])
        else:
            return_data["pose"] = torch.tensor(self.poses[item])
        for i, f in enumerate(self.mip_factors):
            resized_image = cv2.resize(full_image, (0, 0), fx=1 / f, fy=1 / f, interpolation=self.resize_interpolation)
            return_data[f"mip_image_{i}"] = torch.tensor(resized_image, dtype=torch.float32)
        return return_data

    def _load_calibration_file(self):
        calibration_file = [os.path.join(self.basedir, x) for x in CAMERA_PARAMETER_FILENAMES]
        calibration_file = [x for x in calibration_file if os.path.isfile(x)][0]
        with open(calibration_file, "r") as f:
            calibration_file_data = json.load(f)
        all_transforms = []
        all_intrinsics = []
        all_distortion = []
        for camera in calibration_file_data["frames"]:
            transform = np.array(camera["transform_matrix"], dtype=np.float32).reshape((4, 4))

            camera_angle_x = float(calibration_file_data['camera_angle_x'])
            focal = .5 * self.original_width / np.tan(.5 * camera_angle_x)
            intrinsics = np.array([
                [focal, 0, self.original_width / 2],
                [0, focal, self.original_height / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            all_transforms.append(transform)
            all_intrinsics.append(intrinsics)
        if self.dropout_poses and self.include_dropout_only:
            all_transforms = [x for i, x in enumerate(all_transforms) if i in self.dropout_poses]
            all_intrinsics = [x for i, x in enumerate(all_intrinsics) if i in self.dropout_poses]
        elif self.dropout_poses:
            all_transforms = [x for i, x in enumerate(all_transforms) if i not in self.dropout_poses]
            all_intrinsics = [x for i, x in enumerate(all_intrinsics) if i not in self.dropout_poses]
        all_transforms = np.stack(all_transforms)
        all_intrinsics = np.stack(all_intrinsics)
        return all_transforms, all_intrinsics, all_distortion

    def update_factor(self, new_factor: int):
        self.factor = new_factor
        self.height = self.original_height // new_factor
        self.width = self.original_width // new_factor
        self.render_poses.clear()
        if not self.using_processed_camera_parameters:
            self.poses, self.bds = self._load_poses()
            hwf = self.poses[0, :3, -1]
            self.hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
            self.focal = hwf[2]

    def camera_params_to_rays(self, intrinsics: torch.Tensor, transforms: torch.Tensor, height: int, width: int
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts camera parameters to rays in world coordinates.

        Args:
            intrinsics: Intrinsics as (3, 3) tensor.
            transforms: Extrinsics as (4, 4) tensor.
            height: Image height.
            width: Image width.

        Returns:
            Tuple containing ray origins and ray directions.
        """
        dtype = intrinsics.dtype
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Wrong intrinsics shape {intrinsics.shape}")
        if transforms.shape != (4, 4):
            raise ValueError(f"Wrong transforms shape {transforms.shape}")
        device = intrinsics.device
        x = torch.arange(width, dtype=dtype, device=device)
        y = torch.arange(height, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        directions = torch.stack(
            [
                (xx - intrinsics[0, 2]) / intrinsics[0, 0],
                -(yy - intrinsics[1, 2]) / intrinsics[1, 1],
                -torch.ones_like(xx),
            ],
            dim=-1,
        )
        # directions = torch.einsum('ij,abj->abi', transforms[:3, :3], directions)
        directions = directions @ transforms[:3, :3].T
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        origins = transforms[None, None, :3, 3] * torch.ones_like(directions)
        return origins, directions

    def get_render_poses(self, num_views: int = 30):
        if num_views not in self.render_poses:
            if self.using_processed_camera_parameters:
                pca = PCA(2)
                pca.fit(self.transforms[:, :3, 3])
                up = -np.cross(pca.components_[0], pca.components_[1])
                up = up / np.linalg.norm(up)
                target = np.mean(self.transforms[:, :3, 3], axis=0) - self.renderposes_height * up
                rotation_mat = np.stack((pca.components_[0], pca.components_[1], up), axis=1)

                rots = 2
                zrate = 0.0
                transforms = []
                intrinsics = []
                for theta in np.linspace(0.0, 2.0 * np.pi * rots, num_views + 1)[:-1]:
                    position = target + 4.0 * rotation_mat @ np.array(
                        [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate)],
                        dtype=np.float32) + self.renderposes_height * up
                    rotation = look_at(position, -up if up[2] < 0 else up, target)
                    # rotation[:, 0] = -rotation[:, 0]
                    # rotation[:, 2] = -rotation[:, 2]
                    new_transform = np.concatenate((rotation, np.array((0, 0, 0, 1), dtype=np.float32)[None]),
                                                   axis=0)
                    transforms.append(new_transform)
                    intrinsics.append(self._get_resized_intrinsics(0))
                assert len(transforms) == num_views
                assert len(intrinsics) == num_views
                transforms = np.stack(transforms)
                intrinsics = np.stack(intrinsics)
                self.render_poses[num_views] = {
                    "transforms": torch.tensor(transforms),
                    "intrinsics": torch.tensor(intrinsics)
                }
            else:
                poses = self.poses
                bds = self.bds

                c2w = poses_avg(poses)

                # Get spiral
                # Get average pose
                pca = PCA(2)
                pca.fit(poses[:, :3, 3])
                up = -np.cross(pca.components_[0], pca.components_[1])
                up = up / np.linalg.norm(up)
                target = c2w[:, 3] - 1.0 * up

                # Find a reasonable "focus depth" for this dataset
                close_depth, inf_depth = bds.min() * 0.9, bds.max() * 5.0
                dt = 0.75
                mean_dz = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))
                focal = mean_dz

                # Get radii for spiral path
                shrink_factor = 0.8
                zdelta = close_depth * 0.2
                tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
                rads = np.percentile(np.abs(tt), 80, 0)
                c2w_path = c2w
                N_views = num_views
                N_rots = 2

                # Generate poses for spiral path
                render_poses = render_path_spiral(
                    c2w_path, up, rads, focal, zdelta,
                    zrate=0.0, rots=N_rots, N=N_views,
                    target=target
                )

                render_poses = np.array(render_poses).astype(np.float32)
                self.render_poses[num_views] = torch.tensor(render_poses)
        return self.render_poses[num_views]

import glob
import json
import re
from turtle import width
from typing import Tuple, Sequence, Optional

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
    "cameraParametersColmapAlignedNew.txt",
    "cameraParametersColmapAligned.txt",
    "cameraParametersColmapLidarPriorCameraCoversPartiallyInstalled.txt"
)


class HolostudioDataset(Dataset):

    def __init__(self, basedir: str, factor: int = 8, recenter: bool = True, bd_factor: float = 0.75,
                 spherify: bool = False, load_saliency: bool = True, max_frames: int = 30,
                 load_every_nth_view: int = 1, renderposes_height: float = 0.4,
                 resize_interpolation: int = cv2.INTER_AREA,
                 mip_factors: Sequence[int] = (), dropout_poses: Sequence[int] = (),
                 include_dropout_only: bool = False, renderposes_centeroffset: Optional[Sequence[float]] = None,
                 cache_lowres: bool = False):
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
        self.load_every_nth_view = load_every_nth_view
        self.max_frames = max_frames
        self.dataset_type = None
        self.resize_interpolation = resize_interpolation
        self.mip_factors = mip_factors
        self.dropout_poses = set(dropout_poses)
        self.include_dropout_only = include_dropout_only
        self.renderposes_height = renderposes_height
        self.renderposes_centeroffset = (
            np.array(renderposes_centeroffset, dtype=np.float32)
            if renderposes_centeroffset is not None
            else np.zeros(3, dtype=np.float32))
        self.cache_lowres = cache_lowres

        ret = self._parse_images()
        if ret is not None:
            self.dataset_type = IMAGES
            (self.len, self.images_dir, self.images_list, self.masks_dir,
             self.masks_list, self.saliency_dir, self.saliency_list,
             self.original_height, self.original_width) = ret
            self.num_frames = 1
            self.num_views = self.len
        if self.dataset_type is None:
            ret = self._parse_video_frames()
            if ret is not None:
                self.dataset_type = VIDEO_FRAMES
                (self.bgmv_dir, self.len, self.num_frames, self.num_views, self.images_list, self.masks_list,
                 self.original_height, self.original_width) = ret
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
            self.render_poses_target = (np.mean(self.transforms[:, :3, 3], axis=0) - self.renderposes_height * up +
                                        self.renderposes_centeroffset)
            self.render_poses_rotation_mat = np.stack(
                (pca.components_[0], pca.components_[1], up), axis=1)
        else:
            self.poses, self.bds = self._load_poses()
            hwf = self.poses[0, :3, -1]
            self.hwf = [int(hwf[0]), int(hwf[1]), hwf[2]]
            self.focal = hwf[2]
        self.render_poses = {}
        self.dataset_cache = {}

    def _parse_images(self):
        image_dir_names = ["images", "Extracted", "SingleFrameForeground"]
        images_dir = [os.path.join(self.basedir, x) for x in image_dir_names]
        images_dir = [x for x in images_dir if os.path.isdir(x)]
        if not images_dir:
            return None
        images_dir = images_dir[0]
        number_extract = re.compile(r'_(\d+)_(extracted|foreground|mask)')
        images_dir_files = [x for x in os.listdir(images_dir) if os.path.splitext(x)[
            1] in IMAGE_EXTENSIONS]
        if number_extract.search(images_dir_files[0]):
            images_list = sorted(images_dir_files, key=lambda x: int(
                number_extract.search(x)[1]))
        else:
            images_list = sorted(images_dir_files)
        if self.dropout_poses and self.include_dropout_only:
            images_list = [x for i, x in enumerate(
                images_list) if i in self.dropout_poses]
        elif self.dropout_poses:
            images_list = [x for i, x in enumerate(
                images_list) if i not in self.dropout_poses]
        if self.load_every_nth_view > 1:
            images_list = [x for i, x in enumerate(
                images_list) if i % self.load_every_nth_view == 0]
        self_len = len(images_list)

        mask_dir_names = ["Mask", "SingleFrameMasks"]
        masks_dir = [os.path.join(self.basedir, x) for x in mask_dir_names]
        masks_dir = [x for x in masks_dir if os.path.isdir(x)]
        self_masks_dir = None
        self_masks_list = None
        if masks_dir:
            masks_dir = masks_dir[0]
            self_masks_dir = masks_dir
            mask_dir_files = [x for x in os.listdir(masks_dir) if os.path.splitext(x)[
                1] in IMAGE_EXTENSIONS]
            if number_extract.search(mask_dir_files[0]):
                self_masks_list = sorted(
                    mask_dir_files, key=lambda x: int(number_extract.search(x)[1]))
            else:
                self_masks_list = sorted(mask_dir_files)
            if self.dropout_poses and self.include_dropout_only:
                self_masks_list = [x for i, x in enumerate(
                    self_masks_list) if i in self.dropout_poses]
            elif self.dropout_poses:
                self_masks_list = [x for i, x in enumerate(
                    self_masks_list) if i not in self.dropout_poses]
            if self.load_every_nth_view > 1:
                self_masks_list = [x for i, x in enumerate(
                    self_masks_list) if i % self.load_every_nth_view == 0]

        saliency_dir_names = ["saliency"]
        saliency_dir = [os.path.join(self.basedir, x)
                        for x in saliency_dir_names]
        saliency_dir = [x for x in saliency_dir if os.path.isdir(x)]
        self_saliency_dir = None
        self_saliency_list = None
        if saliency_dir and self.load_saliency:
            saliency_dir = saliency_dir[0]
            self_saliency_dir = saliency_dir
            saliency_dir_files = [x for x in os.listdir(
                saliency_dir) if os.path.splitext(x)[1] in IMAGE_EXTENSIONS]
            if number_extract.search(saliency_dir_files[0]):
                self_saliency_list = sorted(
                    saliency_dir_files, key=lambda x: int(number_extract.search(x)[1]))
            else:
                self_saliency_list = sorted(saliency_dir_files)
            if self.dropout_poses and self.include_dropout_only:
                self_saliency_list = [x for i, x in enumerate(
                    self_saliency_list) if i in self.dropout_poses]
            elif self.dropout_poses:
                self_saliency_list = [x for i, x in enumerate(
                    self_saliency_list) if i not in self.dropout_poses]

        image = plt.imread(os.path.join(images_dir, images_list[0]))
        height, width = image.shape[:2]

        return (self_len, images_dir, images_list, self_masks_dir, self_masks_list,
                self_saliency_dir, self_saliency_list, height, width)

    def _parse_video_frames(self):

        possible_dirs = ["DespilledFrames", "BGMV2OutputBK"]
        bgmv_dir = [os.path.join(self.basedir, x) for x in possible_dirs if
                    os.path.isdir(os.path.join(self.basedir, x))]
        if not bgmv_dir:
            return None
        bgmv_dir = bgmv_dir[0]

        unique_views = sorted(os.listdir(bgmv_dir))
        view_to_num = {v: i for i, v in enumerate(unique_views)}

        def extract_view_num(x):
            view_dir = x.split(os.path.sep)[-3]
            return view_to_num[view_dir]

        images_list = glob.glob(os.path.join(bgmv_dir, "*", "com", "*.png"))
        images_list = sorted(
            x for x in images_list if frame_num_extractor(x) < self.max_frames)
        if self.dropout_poses and self.include_dropout_only:
            images_list = [x for x in images_list if extract_view_num(
                x) in self.dropout_poses]
        elif self.dropout_poses:
            images_list = [x for x in images_list if extract_view_num(
                x) not in self.dropout_poses]
        if self.load_every_nth_view > 1:
            images_list = [x for i, x in enumerate(
                images_list) if i % self.load_every_nth_view == 0]
        masks_list = glob.glob(os.path.join(bgmv_dir, "*", "pha", "*.jpg"))
        masks_list = sorted(
            x for x in masks_list if frame_num_extractor(x) < self.max_frames)
        if self.dropout_poses and self.include_dropout_only:
            masks_list = [x for x in masks_list if extract_view_num(
                x) in self.dropout_poses]
        elif self.dropout_poses:
            masks_list = [x for x in masks_list if extract_view_num(
                x) not in self.dropout_poses]
        if self.load_every_nth_view > 1:
            masks_list = [x for i, x in enumerate(
                masks_list) if i % self.load_every_nth_view == 0]
        num_frames = 1 + max(frame_num_extractor(x) for x in images_list)
        num_views = len(
            np.unique([os.path.sep.join(x.split(os.path.sep)[:-2]) for x in images_list]))
        self_len = num_frames * num_views

        image = plt.imread(os.path.join(images_list[0]))
        height = image.shape[0]
        width = image.shape[1]

        return bgmv_dir, self_len, num_frames, num_views, images_list, masks_list, height, width

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if item in self.dataset_cache:
            return self._uncache_data(item)
        if self.dataset_type == IMAGES:
            return self._getitem_images(item)
        elif self.dataset_type == VIDEO_FRAMES:
            return self._getitem_videoframes(item)
        else:
            raise ValueError(f"Unknown dataset type {str(self.dataset_type)}")

    def get_transform(self, item):
        view_num = item // self.num_frames
        return self.transforms[view_num]

    def get_intrinsics(self, item):
        view_num = item // self.num_frames
        return self._get_resized_intrinsics(view_num)

    def _get_resized_intrinsics(self, item):
        intrinsics = np.copy(self.intrinsics[item])
        intrinsics[:2, :] = intrinsics[:2, :] / self.factor
        return intrinsics

    def _getitem_images(self, item):
        factor = self.factor
        image_name = self.images_list[item]
        image = full_image = plt.imread(os.path.join(
            self.images_dir, image_name)).astype(np.float32)
        if factor != 1:
            image = cv2.resize(image, (0, 0), fx=1 / factor, fy=1 /
                               factor, interpolation=self.resize_interpolation)
        height, width, channels = image.shape
        return_data = {
            "index": item,
            "image": torch.tensor(image, dtype=torch.float32),
            "t": torch.tensor(0.0, dtype=torch.float32)
        }
        if self.masks_dir:
            mask = plt.imread(os.path.join(
                self.masks_dir, self.masks_list[item])).astype(np.float32) / 255
            if factor != 1:
                mask = cv2.resize(mask, (0, 0), fx=1 / factor, fy=1 /
                                  factor, interpolation=self.resize_interpolation)
            return_data["mask"] = torch.tensor(mask, dtype=torch.float32)
        elif channels == 4:
            mask = (image[:, :, 3] > 1e-5).astype(np.float32)
            return_data["mask"] = torch.tensor(mask, dtype=torch.float32)
        if self.saliency_dir:
            saliency = plt.imread(os.path.join(
                self.saliency_dir, self.saliency_list[item])).astype(np.float32) / 255
            if factor != 1:
                saliency = cv2.resize(saliency, (0, 0), fx=1 / factor, fy=1 / factor,
                                      interpolation=self.resize_interpolation)
            return_data["saliency"] = saliency
        if self.using_processed_camera_parameters:
            return_data["transform"] = torch.tensor(self.transforms[item])
            return_data["intrinsics"] = torch.tensor(
                self._get_resized_intrinsics(item))
            if len(self.distortions) > 0:
                return_data["distortion"] = torch.tensor(
                    self.distortions[item])
        else:
            return_data["pose"] = torch.tensor(self.poses[item])
        for i, f in enumerate(self.mip_factors):
            resized_image = cv2.resize(
                full_image, (0, 0), fx=1 / f, fy=1 / f, interpolation=self.resize_interpolation)
            return_data[f"mip_image_{i}"] = torch.tensor(
                resized_image, dtype=torch.float32)
        return return_data

    def _getitem_videoframes(self, item):
        view_num = item // self.num_frames
        frame_num = item % self.num_frames
        image = full_image = plt.imread(os.path.join(
            self.images_list[item])).astype(np.float32)
        if self.factor != 1:
            image = cv2.resize(image, (0, 0), fx=1 / self.factor, fy=1 / self.factor,
                               interpolation=self.resize_interpolation)
        if image.shape[2] == 4:
            image[:, :, :3] = image[:, :, :3] * \
                (image[:, :, 3:4] > 0.001).astype(np.float32)
        return_data = {
            "index": item,
            "image": torch.tensor(image, dtype=torch.float32),
            "t": torch.tensor(frame_num, dtype=torch.float32)
        }
        if self.masks_list:
            mask = plt.imread(os.path.join(
                self.masks_list[item])).astype(np.float32) / 255
            if self.factor != 1:
                mask = cv2.resize(mask, (0, 0), fx=1 / self.factor, fy=1 / self.factor,
                                  interpolation=self.resize_interpolation)
            return_data["mask"] = mask
        if self.using_processed_camera_parameters:
            return_data["transform"] = torch.tensor(self.transforms[view_num])
            return_data["intrinsics"] = torch.tensor(
                self._get_resized_intrinsics(view_num))
            if len(self.distortions) > 0:
                return_data["distortion"] = torch.tensor(
                    self.distortions[view_num])
        else:
            return_data["pose"] = torch.tensor(self.poses[view_num])
        for i, f in enumerate(self.mip_factors):
            resized_image = cv2.resize(
                full_image, (0, 0), fx=1 / f, fy=1 / f, interpolation=self.resize_interpolation)
            return_data[f"mip_image_{i}"] = torch.tensor(
                resized_image, dtype=torch.float32)
        if self.cache_lowres and self.factor >= 4:
            self._cache_data(return_data)
        return return_data

    def _cache_data(self, return_data):
        index = return_data["index"]
        fg_pixels = return_data["mask"] > 0.1
        bg_pixels = np.logical_not(fg_pixels).nonzero()
        fg_pixels = fg_pixels.nonzero()
        bg_color = return_data["image"][bg_pixels][0]
        self.dataset_cache[index] = {
            "fg_pixels": fg_pixels,
            "fg_colors": return_data["image"][fg_pixels],
            "bg_color": bg_color,
            "height": return_data["image"].shape[0],
            "width": return_data["image"].shape[1],
        }
        other_params = ["transform", "intrinsics", "distortion", "pose", "t"]
        for param in other_params:
            if param in return_data:
                self.dataset_cache[index][param] = return_data[param]

    def _uncache_data(self, index):
        cached_item = self.dataset_cache[index]
        height = cached_item["height"]
        width = cached_item["width"]
        fg_colors = cached_item["fg_colors"]
        image = (torch.ones((height, width, 1), dtype=torch.float32) *
                 cached_item["bg_color"][None, None, :])
        image[cached_item["fg_pixels"]] = fg_colors
        mask = np.zeros((height, width))
        mask[cached_item["fg_pixels"]] = 1
        return_data = {
            "image": image,
            "mask": mask.astype(np.float32),
        }
        other_params = ["transform", "intrinsics", "distortion", "pose", "t"]
        for param in other_params:
            if param in cached_item:
                return_data[param] = cached_item[param]
        return return_data

    def _load_poses(self):
        calibration_dir = os.path.join(self.basedir, "calibration")
        factor = self.factor
        bd_factor = self.bd_factor
        recenter = self.recenter

        poses, bds = load_poses(calibration_dir, factor=factor)

        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate(
            [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Rescale if bd_factor is provided
        sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
        poses[:, :3, 3] *= sc
        bds *= sc

        if recenter:
            poses = recenter_poses(poses)

        print("Data:", poses.shape, bds.shape)

        if self.dropout_poses and self.include_dropout_only:
            raise NotImplementedError()
        elif self.dropout_poses:
            raise NotImplementedError()

        poses = poses.astype(np.float32)
        return poses, bds

    def _load_calibration_file(self):
        calibration_file = [os.path.join(self.basedir, x)
                            for x in CAMERA_PARAMETER_FILENAMES]
        calibration_file = [
            x for x in calibration_file if os.path.isfile(x)][0]
        with open(calibration_file, "r") as f:
            calibration_file_data = json.load(f)
        all_transforms = []
        all_intrinsics = []
        all_distortion = []
        for camera in calibration_file_data:
            transform = np.array(camera["transform"], dtype=np.float32).reshape(
                (4, 4)).transpose()
            rotation = transform[:3, :3].transpose()
            camera_positions = -rotation @ transform[:3, 3]
            transform = np.concatenate(
                (rotation, camera_positions[:, None]), axis=1)
            transform = np.concatenate((transform, np.array(
                (0, 0, 0, 1), dtype=np.float32)[None]), axis=0)

            intrinsics = np.array(camera["intrinsics"], dtype=np.float32).reshape(
                (4, 4)).transpose()[:3, :3]
            all_transforms.append(transform)
            all_intrinsics.append(intrinsics)
            if "distortion" in camera:
                distortion = np.array(camera["distortion"], dtype=np.float32)
                all_distortion.append(distortion)
        if self.dropout_poses and self.include_dropout_only:
            all_transforms = [x for i, x in enumerate(
                all_transforms) if i in self.dropout_poses]
            all_intrinsics = [x for i, x in enumerate(
                all_intrinsics) if i in self.dropout_poses]
            all_distortion = [x for i, x in enumerate(
                all_distortion) if i in self.dropout_poses]
        elif self.dropout_poses:
            all_transforms = [x for i, x in enumerate(
                all_transforms) if i not in self.dropout_poses]
            all_intrinsics = [x for i, x in enumerate(
                all_intrinsics) if i not in self.dropout_poses]
            all_distortion = [x for i, x in enumerate(
                all_distortion) if i not in self.dropout_poses]
        if self.load_every_nth_view > 1:
            all_transforms = [x for i, x in enumerate(
                all_transforms) if i % self.load_every_nth_view == 0]
            all_intrinsics = [x for i, x in enumerate(
                all_intrinsics) if i % self.load_every_nth_view == 0]
            all_distortion = [x for i, x in enumerate(
                all_distortion) if i % self.load_every_nth_view == 0]
        all_transforms = np.stack(all_transforms)
        all_intrinsics = np.stack(all_intrinsics)
        all_distortion = np.stack(all_distortion) if all_distortion else []
        return all_transforms, all_intrinsics, all_distortion

    def update_factor(self, new_factor: int):
        self.factor = new_factor
        self.height = self.original_height // new_factor
        self.width = self.original_width // new_factor
        self.render_poses.clear()
        self.dataset_cache.clear()
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
                (yy - intrinsics[1, 2]) / intrinsics[1, 1],
                torch.ones_like(xx),
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
                target = (np.mean(self.transforms[:, :3, 3], axis=0) - self.renderposes_height * up +
                          self.renderposes_centeroffset)
                rotation_mat = np.stack(
                    (pca.components_[0], pca.components_[1], up), axis=1)

                rots = 2
                zrate = 0.0
                transforms = []
                intrinsics = []
                for theta in np.linspace(0.0, 2.0 * np.pi * rots, num_views + 1)[:-1]:
                    position = target + 4.0 * rotation_mat @ np.array(
                        [np.cos(theta), -np.sin(theta), -
                         np.sin(theta * zrate)],
                        dtype=np.float32)
                    rotation = look_at(position, up, target)
                    rotation[:, 0] = -rotation[:, 0]
                    rotation[:, 2] = -rotation[:, 2]
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

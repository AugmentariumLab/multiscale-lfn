import os
import json

import cv2
import tqdm
import numpy as np


def main():
    alpha_threshold = 0.02
    base_dir = "datasets/MariaCamera0124Take2"
    despilled_dir = os.path.join(base_dir, "DespilledFrames")
    cameras = os.listdir(despilled_dir)
    # os.makedirs(os.path.join(base_dir, "crop_test"), exist_ok=True)
    global_h_min = 10e10
    global_h_max = 0
    global_w_min = 10e10
    global_w_max = 0
    for camera in tqdm.tqdm(cameras):
        mask_path = os.path.join(despilled_dir, camera, "pha", "00000.jpg")
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        valid_pixels = mask_image > alpha_threshold
        alpha_ok_h = np.where(np.any(valid_pixels, axis=1))[0]
        alpha_ok_w = np.where(np.any(valid_pixels, axis=0))[0]
        alpha_ok_h_min = np.min(alpha_ok_h)
        alpha_ok_h_max = np.max(alpha_ok_h) + 1
        alpha_ok_w_min = np.min(alpha_ok_w)
        alpha_ok_w_max = np.max(alpha_ok_w) + 1
        cropped_mask = mask_image[alpha_ok_h_min:alpha_ok_h_max, alpha_ok_w_min:alpha_ok_w_max]
        # cv2.imwrite(os.path.join(base_dir, "crop_test", camera + ".jpg"), cropped_mask)
        global_h_min = min(global_h_min, alpha_ok_h_min)
        global_h_max = max(global_h_max, alpha_ok_h_max)
        global_w_min = min(global_w_min, alpha_ok_w_min)
        global_w_max = max(global_w_max, alpha_ok_w_max)
    with open(os.path.join(base_dir, "global_crop.json"), "w") as f:
        json.dump({
            "crop_params": [int(global_w_min), int(global_w_max),
                            int(global_h_min), int(global_h_max)]
        }, f, indent=2)


if __name__ == "__main__":
    main()

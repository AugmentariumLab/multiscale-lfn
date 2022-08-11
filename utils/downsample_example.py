import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


def downsample_circles():
    image_filename = "assets/concentric_circles.png"
    image = cv2.imread(image_filename)
    print("image shape", image.shape)
    new_height = new_width = 64
    nearest_downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    area_downsampled_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    # cv2.imwrite("assets/downsampled_nearest.png", nearest_downsampled_image)
    # cv2.imwrite("assets/downsampled_area.png", area_downsampled_image)

    new_image = []
    mask_size = 10
    for channel in range(image.shape[2]):
        bgimage = image[:, :, channel]
        f = np.fft.fft2(bgimage)
        fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        rows, cols = bgimage.shape
        crow, ccol = rows // 2, cols // 2
        plt.imsave(f"assets/fourier_{channel}.png", 20 *
                   np.log(np.abs(fshift) + 1e-10), cmap="gray")

        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - mask_size:crow + mask_size,
             ccol - mask_size:ccol + mask_size] = 1
        # apply mask and inverse DFT
        fshift = fshift * mask
        plt.imsave(f"assets/fourier_pass_{channel}.png",
                   20 * np.log(np.abs(fshift) + 1e-10), cmap="gray")

        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)
        new_image.append(img_back)
    new_image = np.stack(new_image, axis=-1)
    cv2.imwrite("assets/downsampled_dft.png", new_image)


def downsample_image():
    base_dir = "datasets/DavidDataset0113Config5v4/frames/"
    images = [
        "Board103Camera1.png",
        "Board112Camera2.png",
    ]
    images = [os.path.join(base_dir, x) for x in images]
    crops=[
        [1723, 492, 2342, 2012],
        [1723, 857, 2521, 2352]
    ]
    for i, image_filename in enumerate(images):
        image = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)
        print("image shape", image.shape)
        crop=crops[i]
        for f in [1, 2, 4, 8]:
            if f != 1:
                area_downsampled_image = cv2.resize(
                    image, (0, 0), fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)
            else:
                area_downsampled_image = image.copy()
            crop_resized = [crop[0]//f, crop[1]//f, crop[2]//f, crop[3]//f]
            cropped_image = area_downsampled_image[crop_resized[1]:crop_resized[3], crop_resized[0]:crop_resized[2]]
            cv2.imwrite(f"temp/{i}_downsampled_area_f{f}.png", cropped_image)


if __name__ == "__main__":
    # downsample_circles()
    downsample_image()

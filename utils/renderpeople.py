import os
import json

import bpy
import numpy as np


def load_calibration_file(filepath):
    with open(filepath, "r") as f:
        calibration_file_data = json.load(f)
    all_transforms = []
    all_intrinsics = []
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
    all_transforms = np.stack(all_transforms)
    all_intrinsics = np.stack(all_intrinsics)
    return all_transforms, all_intrinsics


def render_to_file(filepath, width, height):
    bpy.context.scene.render.resolution_x = width
    bpy.context.scene.render.resolution_y = height
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def delete_all_cameras():
    for k, v in bpy.context.collection.objects.items():
        if isinstance(v.data, bpy.types.Camera):
            bpy.data.objects.remove(v, do_unlink=True)


def get_calibration_matrix_K_from_blender(mode='simple'):
    projection_matrix = bpy.context.scene.camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(), x=WIDTH, y=HEIGHT)
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    camera = bpy.context.scene.camera
    k = np.array((width * projection_matrix[0][0], 0, width / 2,
                  0, height * projection_matrix[1][1], height / 2,
                  0, 0, 1)).reshape(3, 3)
    return k


camera_params_file = "cameraParametersColmapAligned.txt"
transforms, intrinsics = load_calibration_file(camera_params_file)
print(os.getcwd())
print("transforms.shape", transforms.shape)

WIDTH = 4032
HEIGHT = 3040
output_dir = "images"

os.makedirs(output_dir, exist_ok=True)
main_object = bpy.context.collection.objects[0]
main_object.location.z = -1.5
delete_all_cameras()
yz_flip_3 = np.array(((1, 0, 0,),
                      (0, 0, 1,),
                      (0, -1, 0)))
yz_flip_3 = np.array(((1, 0, 0),
                      (0, -1, 0),
                      (0, 0, -1))) @ yz_flip_3
r1 = np.array(((1, 0, 0,),
               (0, -1, 0,),
               (0, 0, -1)))

print("intrinsics", intrinsics[0])

new_intrinsics = []
for i in range(len(transforms)):
    print(f"processing {i}/{len(transforms)}")
    delete_all_cameras()
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_data.angle = 1.2
    camera_object = bpy.data.objects.new('Camera', camera_data)
    camera_rot = transforms[i, :3, :3] @ r1
    camera_trans = transforms[i, :3, 3]
    camera_rot = yz_flip_3 @ camera_rot
    camera_trans = yz_flip_3 @ camera_trans
    new_matrix_world = np.concatenate((camera_rot, camera_trans[:, None]), axis=1)
    new_matrix_world = np.concatenate((new_matrix_world, np.array((0, 0, 0, 1))[None]), axis=0)
    camera_object.matrix_world = new_matrix_world.T
    bpy.context.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    render_to_file(os.path.join(output_dir, f"{i:04d}.png"), WIDTH, HEIGHT)
    k = get_calibration_matrix_K_from_blender()
    k = np.concatenate((k, np.zeros(3)[:, None]), axis=1)
    k = np.concatenate((k, np.array((0, 0, 0, 1))[None]), axis=0)
    new_intrinsics.append(k)

new_calibration = []
for t, i in zip(transforms, new_intrinsics):
    r = t[:3, :3]
    p = -np.linalg.inv(r) @ t[:3, 3]
    new_t = np.concatenate((r.T, p[:, None]), axis=1)
    new_t = np.concatenate((new_t, np.array((0, 0, 0, 1))[None]), axis=0)
    new_calibration.append({
        "transform": new_t.T.reshape(-1).tolist(),
        "intrinsics": i.T.reshape(-1).tolist()
    })
with open("cameraParametersColmapAlignedNew.txt", "w") as f:
    json.dump(new_calibration, f, indent=2)

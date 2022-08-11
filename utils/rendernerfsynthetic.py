# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.

import argparse, sys, os
import json
from math import radians
import bpy
import mathutils
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


yz_flip_3 = np.array(((1, 0, 0,),
                      (0, 0, 1,),
                      (0, -1, 0)))
yz_flip_3 = np.array(((1, 0, 0),
                      (0, -1, 0),
                      (0, 0, -1))) @ yz_flip_3
r1 = np.array(((1, 0, 0,),
               (0, -1, 0,),
               (0, 0, -1)))
translation_scale = 1.2 * np.identity(3)

camera_params_file = "cameraParametersColmapAligned.txt"
transforms, intrinsics = load_calibration_file(camera_params_file)

DEBUG = False

RESOLUTION = 800
RESULTS_PATH = 'drums_hs_240'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
CIRCLE_FIXED_START = (.3, 0, 0)
RENDER_DEPTH_NORMAL = False

fp = bpy.path.abspath(f"//{RESULTS_PATH}")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
# bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG and RENDER_DEPTH_NORMAL:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
        # Remap as other types can not represent the full range of depth.
        map = tree.nodes.new(type="CompositorNodeMapValue")
        # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
        map.offset = [-0.7]
        map.size = [DEPTH_SCALE]
        map.use_min = True
        map.min = [0]
        links.new(render_layers.outputs['Depth'], map.inputs[0])

        links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background


objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (0, 0, 0)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

rotation_mode = 'XYZ'

if not DEBUG and RENDER_DEPTH_NORMAL:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''

out_data['frames'] = []

for i in range(0, min(5, transforms.shape[0])):

    scene.render.filepath = fp + '/r_' + str(i)

    if RENDER_DEPTH_NORMAL:
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

    camera_rot = transforms[i, :3, :3] @ r1
    camera_trans = transforms[i, :3, 3]
    camera_rot = yz_flip_3 @ camera_rot
    camera_trans = translation_scale @ yz_flip_3 @ camera_trans
    new_matrix_world = np.concatenate((camera_rot, camera_trans[:, None]), axis=1)
    new_matrix_world = np.concatenate((new_matrix_world, np.array((0, 0, 0, 1))[None]), axis=0)
    cam.matrix_world = new_matrix_world.T

    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': scene.render.filepath,
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

if not DEBUG:
    with open(fp + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

import torch
import numpy as np
import open3d as o3d
from gedi import GeDi
from scipy.spatial.transform import Rotation as R, Slerp
import os
import copy

# Function to perform RANSAC registration
def run_ransac(reg_pcd0, reg_pcd1, pcd0_feature, pcd1_feature, voxel_size):
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        reg_pcd0,
        reg_pcd1,
        pcd0_feature,
        pcd1_feature,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 3,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 3)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.99)
    )
    ransac_matrix = result_ransac.transformation
    return ransac_matrix

# Function to perform ICP refinement
def run_icp(reg_pcd0, reg_pcd1, initial_transformation, voxel_size):
    icp_threshold = voxel_size * 0.4
    icp_result = o3d.pipelines.registration.registration_icp(
        reg_pcd0, reg_pcd1, icp_threshold, initial_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    icp_matrix = icp_result.transformation
    return icp_matrix

# Function to create the registration animation
def create_registration_animation(pcd0, pcd1, transformation, output_dir, steps=100, camera_params_file=None):
    """
    Creates an animation of the registration process by interpolating between 
    the initial and final transformation matrices.

    Args:
        pcd0 (open3d.geometry.PointCloud): Source point cloud.
        pcd1 (open3d.geometry.PointCloud): Target point cloud.
        transformation (numpy.ndarray): Final transformation matrix after ICP.
        output_dir (str): Directory to save the animation frames.
        steps (int): Number of interpolation steps (frames).
        camera_params_file (str, optional): Path to the camera parameters JSON file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Registration Animation')
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd0)
    
    # Set custom camera parameters if provided
    if camera_params_file and os.path.isfile(camera_params_file):
        cam_params = o3d.io.read_pinhole_camera_parameters(camera_params_file)
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(cam_params)
        print(f"Applied custom camera parameters from '{camera_params_file}'")
    else:
        print("No custom camera parameters provided. Using default view.")
    
    # Decompose the final transformation matrix
    transformation = np.asarray(transformation).copy()
    rotation_final = transformation[:3, :3].copy()  # Make a writable copy
    translation_final = transformation[:3, 3].copy()
    
    # Initial rotation (identity) and translation (zeros)
    rotation_initial = np.eye(3)
    translation_initial = np.zeros(3)
    
    # Convert rotations to quaternions
    quat_initial = R.from_matrix(rotation_initial).as_quat()
    quat_final = R.from_matrix(rotation_final).as_quat()
    
    # Setup Slerp
    key_times = [0, 1]
    key_rotations = R.from_quat([quat_initial, quat_final])
    slerp = Slerp(key_times, key_rotations)
    
    # Interpolation steps
    for i in range(steps + 1):
        t = i / steps  # Interpolation parameter from 0 to 1
        
        # Interpolate rotation using SLERP
        rot_interp = slerp(t)
        quat_interp = rot_interp.as_quat()
        rotation_interp = rot_interp.as_matrix()
        
        # Interpolate translation linearly
        translation_interp = (1 - t) * translation_initial + t * translation_final
        
        # Create transformation matrix
        transformation_interp = np.eye(4)
        transformation_interp[:3, :3] = rotation_interp
        transformation_interp[:3, 3] = translation_interp
        
        # Apply transformation to a copy of the source point cloud
        pcd0_temp = copy.deepcopy(pcd0)
        pcd0_temp.transform(transformation_interp)
        
        # Update geometry in the visualizer
        vis.clear_geometries()
        vis.add_geometry(pcd1)
        vis.add_geometry(pcd0_temp)
        
        # Apply custom camera parameters at each step if needed
        if camera_params_file and os.path.isfile(camera_params_file):
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(cam_params)
        
        vis.poll_events()
        vis.update_renderer()
        
        # Capture frame
        image = vis.capture_screen_float_buffer(False)
        image = (np.asarray(image) * 255).astype(np.uint8)
        image_path = os.path.join(output_dir, f'frame_{i:04d}.png')
        o3d.io.write_image(image_path, o3d.geometry.Image(image))
        
    vis.destroy_window()
    print(f"Animation frames saved to '{output_dir}'")

config = {'dim': 32,                                            # descriptor output dimension
          'samples_per_batch': 500,                             # batches to process the data on GPU
          'samples_per_patch_lrf': 4000,                        # num. of point to process with LRF
          'samples_per_patch_out': 512,                         # num. of points to sample for pointnet++
          'r_lrf': .5,                                          # LRF radius
          'fchkpt_gedi_net': 'data/chkpts/3dmatch/chkpt.tar'}   # path to checkpoint

voxel_size = .01
patches_per_pair = 5000

# initializing GeDi
gedi = GeDi(config=config)

# loading point clouds
pcd0 = o3d.io.read_point_cloud('/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply')
pcd1 = o3d.io.read_point_cloud('/home/martyn/Thesis/pose-estimation/data/scenes/scene_01/point_cloud_cropped.ply')

# estimating normals (only for visualization)
pcd0.estimate_normals()
pcd1.estimate_normals()

# Visualize before registration
print("Displaying point clouds before registration...")
o3d.visualization.draw_geometries([pcd0, pcd1])

# randomly sampling some points from the point cloud
inds0 = np.random.choice(np.asarray(pcd0.points).shape[0], patches_per_pair, replace=False)
inds1 = np.random.choice(np.asarray(pcd1.points).shape[0], patches_per_pair, replace=False)

pts0 = torch.tensor(np.asarray(pcd0.points)[inds0]).float()
pts1 = torch.tensor(np.asarray(pcd1.points)[inds1]).float()

# applying voxelization to the point cloud
pcd0_voxel = pcd0.voxel_down_sample(voxel_size)
pcd1_voxel = pcd1.voxel_down_sample(voxel_size)

_pcd0 = torch.tensor(np.asarray(pcd0_voxel.points)).float()
_pcd1 = torch.tensor(np.asarray(pcd1_voxel.points)).float()

# computing descriptors
pcd0_desc = gedi.compute(pts=pts0, pcd=_pcd0)
pcd1_desc = gedi.compute(pts=pts1, pcd=_pcd1)

# preparing format for open3d ransac
pcd0_dsdv = o3d.pipelines.registration.Feature()
pcd1_dsdv = o3d.pipelines.registration.Feature()

pcd0_dsdv.data = pcd0_desc.T
pcd1_dsdv.data = pcd1_desc.T

_pcd0_pts = o3d.geometry.PointCloud()
_pcd0_pts.points = o3d.utility.Vector3dVector(pts0.numpy())
_pcd1_pts = o3d.geometry.PointCloud()
_pcd1_pts.points = o3d.utility.Vector3dVector(pts1.numpy())

# Run RANSAC
ransac_matrix = run_ransac(_pcd0_pts, _pcd1_pts, pcd0_dsdv, pcd1_dsdv, voxel_size)

# Run ICP refinement
icp_matrix = run_icp(_pcd0_pts, _pcd1_pts, ransac_matrix, voxel_size)

# Specify the path to your saved camera parameters
camera_params_file = "camera_params.json"  # Update this path if necessary

# Create animation
output_dir = "registration_animation_frames"
create_registration_animation(pcd0, pcd1, icp_matrix, output_dir, steps=100, camera_params_file=camera_params_file)

# Optionally, create a video from the frames using FFmpeg
# After running the script, you can use the following command in your terminal:
# ffmpeg -framerate 30 -i registration_animation_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p registration_animation.mp4

# Visualize after registration
print("Displaying point clouds after registration...")
pcd0.transform(icp_matrix)
o3d.visualization.draw_geometries([pcd0, pcd1])

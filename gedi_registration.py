import open3d as o3d
import numpy as np
import copy
import cv2
import os
import time
import torch
from gedi import GeDi

# Function to compute descriptors using GeDi
def compute_gedi_features(gedi, pcd, pcd_voxel, num_samples):
    # Randomly sample points
    inds = np.random.choice(len(pcd.points), num_samples, replace=False)
    pts = torch.tensor(np.asarray(pcd.points)[inds]).float()

    # pcd_voxel is already provided
    _pcd = torch.tensor(np.asarray(pcd_voxel.points)).float()

    # Compute descriptors
    descriptors = gedi.compute(pts=pts, pcd=_pcd)

    # Prepare Open3D Feature object
    feature = o3d.pipelines.registration.Feature()
    feature.data = descriptors.T  # descriptors is already a numpy array

    # Prepare point cloud
    reg_pcd = o3d.geometry.PointCloud()
    reg_pcd.points = o3d.utility.Vector3dVector(pts.numpy())

    return reg_pcd, feature

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

# Function to project 3D points to 2D using intrinsic matrix
def project_points_to_image_plane(points_3d, intrinsic_matrix, transformation_matrix):
    # Transform points with the transformation matrix
    points_3d_hom = np.hstack(
        (points_3d, np.ones((points_3d.shape[0], 1)))
    )
    transformed_points = (
        (transformation_matrix @ points_3d_hom.T).T[:, :3]
    )

    # Project transformed points to 2D
    projected_points = intrinsic_matrix @ transformed_points.T
    projected_points = (projected_points[:2] / projected_points[2]).T

    return projected_points

def overlay_point_cloud_on_image(
    source, image, transformation_matrix, intrinsic_matrix
):
    points_3d = np.asarray(source.points)

    # Project points to 2D
    projected_points = project_points_to_image_plane(
        points_3d, intrinsic_matrix, transformation_matrix
    )

    # Create an overlay layer
    overlay = image.copy()

    # Draw points on the overlay
    for pt in projected_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
            cv2.circle(
                overlay, (x, y), radius=2, color=(0, 0, 255), thickness=-1
            )

    # Blend the overlay with the original image
    alpha = 0.50
    blended_image = cv2.addWeighted(
        overlay, alpha, image, 1 - alpha, 0
    )

    return blended_image  # Return the blended image instead of saving it

def load_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    intrinsic_matrix = []
    for line in lines:
        numbers = [float(num) for num in line.strip().split()]
        intrinsic_matrix.append(numbers)
    intrinsic_matrix = np.array(intrinsic_matrix)
    return intrinsic_matrix

# Function to compute evaluation metrics
def compute_metrics(estimated_matrix, ground_truth_matrix):
    translation_est = estimated_matrix[:3, 3]
    translation_gt = ground_truth_matrix[:3, 3]
    # Convert translation vectors from meters to millimeters
    translation_error_m = np.linalg.norm(translation_est - translation_gt)
    translation_error_mm = translation_error_m * 1000  # Convert to mm
    
    rotation_est = estimated_matrix[:3, :3]
    rotation_gt = ground_truth_matrix[:3, :3]
    rotation_diff = np.dot(rotation_est, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = np.clip((trace - 1) / 2, -1.0, 1.0)
    rotation_error = np.arccos(trace)
    return translation_error_mm, np.degrees(rotation_error)

def main():
    voxel_size = 0.01
    scenes_root_dir = "/home/martyn/Thesis/pose-estimation/data/scenes_axle/scenes_single/"
    gedi_checkpoint = "data/chkpts/3dmatch/chkpt.tar"

    # Load intrinsic matrix (assuming it's the same for all scenes)
    intrinsic_matrix = load_intrinsic_matrix("/home/martyn/Thesis/pose-estimation/data/cam_K.txt")
    source_path = "/home/martyn/Thesis/pose-estimation/data/point-clouds/point_cloud_medium.ply"
    source = o3d.io.read_point_cloud(source_path)

    # Error handling for source point cloud
    if source.is_empty():
        print("Error: Source point cloud is empty or failed to load.")
        return

    # Initialize GeDi
    print("Initializing GeDi...")
    config = {
        'dim': 32,                                           # Descriptor output dimension
        'samples_per_batch': 2000,                           # Batches to process data on GPU
        'samples_per_patch_lrf': 1500,                       # Num. of points to process with LRF
        'samples_per_patch_out': 256,                        # Num. of points to sample for PointNet++
        'r_lrf': 0.5,                                        # LRF radius
        'fchkpt_gedi_net': gedi_checkpoint                   # Path to checkpoint
    }
    gedi = GeDi(config=config)

    # Parameters
    num_samples = 2500  # Number of points to sample for feature computation

    # Specify the scene number directly
    scene_num = 1  # Change this number to select a different scene
    scene_name = f"scene_{scene_num:02d}"
    scene_dir = os.path.join(scenes_root_dir, scene_name)
    # Remove output directory since we are not saving results
    # output_dir = os.path.join(output_root_dir, scene_name)

    # Load scene-specific data
    target_path = os.path.join(scene_dir, "point_cloud_cropped.ply")
    #ground_truth_path = os.path.join(scene_dir, "tf_ground_truth.txt")
    rgb_image_path = os.path.join(scene_dir, "rgb.png")

    target = o3d.io.read_point_cloud(target_path)
    #ground_truth = np.loadtxt(ground_truth_path)
    image = cv2.imread(rgb_image_path)

    # Error handling
    if target.is_empty():
        print(f"Error: Target point cloud is empty or failed to load for {scene_name}.")
        return
    #if ground_truth.size == 0:
    #    print(f"Error: Ground truth transformation matrix failed to load for {scene_name}.")
    #    return
    if image is None:
        print(f"Error: Failed to load image for {scene_name}.")
        return

    # Voxelization (performed once per scene)
    source_voxel = source.voxel_down_sample(voxel_size)
    target_voxel = target.voxel_down_sample(voxel_size)

    print(f"\n{scene_name} - Single Run")

    # Start timing the total process for this run (starts before descriptor computation)
    total_start_time = time.time()

    # Compute descriptors using GeDi for source and target
    print("Computing descriptors using GeDi...")
    descriptor_start_time = time.time()
    reg_pcd0, pcd0_feature = compute_gedi_features(gedi, source, source_voxel, num_samples)
    reg_pcd1, pcd1_feature = compute_gedi_features(gedi, target, target_voxel, num_samples)
    descriptor_time = time.time() - descriptor_start_time
    print(f"Descriptor computation time: {descriptor_time:.2f} seconds")

    # Run RANSAC
    ransac_start_time = time.time()
    ransac_matrix = run_ransac(reg_pcd0, reg_pcd1, pcd0_feature, pcd1_feature, voxel_size)
    ransac_time = time.time() - ransac_start_time
    print(f"RANSAC registration time: {ransac_time:.2f} seconds")
    print("RANSAC Transformation:\n", ransac_matrix)

    # Run ICP refinement
    icp_start_time = time.time()
    icp_matrix = run_icp(reg_pcd0, reg_pcd1, ransac_matrix, voxel_size)
    icp_time = time.time() - icp_start_time
    print(f"ICP refinement time: {icp_time:.2f} seconds")
    print("ICP Transformation:\n", icp_matrix)

    # End timing the total process for this run (ends after ICP refinement)
    total_runtime = time.time() - total_start_time

    # Compute metrics
    #ransac_translation_error, ransac_rotation_error = compute_metrics(
    #    ransac_matrix, ground_truth
    #)
    #icp_translation_error, icp_rotation_error = compute_metrics(
    #    icp_matrix, ground_truth
    #)

    # Print metrics for this run
    #print("\nMetrics for this run:")
    #print(f"RANSAC Translation Error: {ransac_translation_error:.6f} mm")
    #print(f"RANSAC Rotation Error: {ransac_rotation_error:.6f} deg")
    #print(f"ICP Translation Error: {icp_translation_error:.6f} mm")
    #print(f"ICP Rotation Error: {icp_rotation_error:.6f} deg")

    # Create and visualize overlay image
    print("Creating overlay image...")
    blended_image = overlay_point_cloud_on_image(
        source, image, icp_matrix, intrinsic_matrix
    )

    # Display the overlay image
    cv2.imshow(f"Overlay Image - {scene_name}", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
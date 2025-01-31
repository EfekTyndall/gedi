import open3d as o3d
import numpy as np
import copy
import cv2
import os
import time
import pandas as pd
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
    source, image, transformation_matrix, intrinsic_matrix, output_path
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

    # Save the blended image
    cv2.imwrite(output_path, blended_image)

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


def add_metric(model_points, R_gt, t_gt, R_pred, t_pred):
    """
    Calculates the Average Distance of Model Points (ADD) metric.
    """
    # Transform points using ground truth
    transformed_gt = (R_gt @ model_points.T).T + t_gt

    # Transform points using predictions
    transformed_pred = (R_pred @ model_points.T).T + t_pred

    # Compute average distance
    add = np.mean(np.linalg.norm(transformed_gt - transformed_pred, axis=1))
    return add



def main():
    voxel_size = 0.01
    scenes_root_dir = "/home/martyn/Thesis/pose-estimation/data/scenes/"
    output_root_dir = "/home/martyn/Thesis/pose-estimation/results/methods/gedi/"
    gedi_checkpoint = "/home/martyn/Thesis/pose-estimation/methods/gedi/data/chkpts/3dmatch/chkpt.tar"

    # Ensure output root directory exists
    os.makedirs(output_root_dir, exist_ok=True)

    # Load intrinsic matrix (assuming it's the same for all scenes)
    intrinsic_matrix = load_intrinsic_matrix("/home/martyn/Thesis/pose-estimation/data/cam_K.txt")
    source_path = "/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply"
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

    # Prepare lists to collect metrics over all scenes
    all_metrics = []

    # Parameters
    num_samples = 2500  # Number of points to sample for feature computation
    num_runs = 5        # Number of runs per scene

    for scene_num in range(1, 11):  # scenes from 1 to 10
        scene_name = f"scene_{scene_num:02d}"
        scene_dir = os.path.join(scenes_root_dir, scene_name)
        output_dir = os.path.join(output_root_dir, scene_name)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load scene-specific data
        target_path = os.path.join(scene_dir, "point_cloud_cropped.ply")
        ground_truth_path = os.path.join(scene_dir, "tf_ground_truth.txt")
        rgb_image_path = os.path.join(scene_dir, "rgb.png")

        target = o3d.io.read_point_cloud(target_path)
        ground_truth = np.loadtxt(ground_truth_path)
        image = cv2.imread(rgb_image_path)

        # Error handling
        if target.is_empty():
            print(f"Error: Target point cloud is empty or failed to load for {scene_name}.")
            continue
        if ground_truth.size == 0:
            print(f"Error: Ground truth transformation matrix failed to load for {scene_name}.")
            continue
        if image is None:
            print(f"Error: Failed to load image for {scene_name}.")
            continue

        # Voxelization (performed once per scene)
        source_voxel = source.voxel_down_sample(voxel_size)
        target_voxel = target.voxel_down_sample(voxel_size)

        # Initialize lists to store metrics and runtimes for this scene
        ransac_translation_errors = []
        ransac_rotation_errors = []
        icp_translation_errors = []
        icp_rotation_errors = []
        descriptor_times = []      # List to store descriptor computation times
        ransac_runtimes = []
        icp_runtimes = []
        total_runtimes = []
        add_metrics = []

        for i in range(num_runs):
            print(f"\n{scene_name} - Run {i+1}/{num_runs}")

            # Create a directory for this run
            run_dir = os.path.join(output_dir, f"run_{i+1:02d}")
            os.makedirs(run_dir, exist_ok=True)

            # Start timing the total process for this run (starts before descriptor computation)
            total_start_time = time.time()

            # Compute descriptors using GeDi for source and target
            print("Computing descriptors using GeDi...")
            descriptor_start_time = time.time()
            reg_pcd0, pcd0_feature = compute_gedi_features(gedi, source, source_voxel, num_samples)
            reg_pcd1, pcd1_feature = compute_gedi_features(gedi, target, target_voxel, num_samples)
            descriptor_time = time.time() - descriptor_start_time
            print(f"Descriptor computation time: {descriptor_time:.2f} seconds")
            descriptor_times.append(descriptor_time)

            # Run RANSAC
            ransac_start_time = time.time()
            ransac_matrix = run_ransac(reg_pcd0, reg_pcd1, pcd0_feature, pcd1_feature, voxel_size)
            ransac_time = time.time() - ransac_start_time
            print(f"RANSAC registration time: {ransac_time:.2f} seconds")
            print("RANSAC Transformation:\n", ransac_matrix)

            # Save RANSAC results for this run
            np.savetxt(os.path.join(run_dir, "ransac_transformation.txt"), ransac_matrix)

            # Run ICP refinement
            icp_start_time = time.time()
            icp_matrix = run_icp(reg_pcd0, reg_pcd1, ransac_matrix, voxel_size)
            icp_time = time.time() - icp_start_time
            print(f"ICP refinement time: {icp_time:.2f} seconds")
            print("ICP Transformation:\n", icp_matrix)

            # Save ICP results for this run
            np.savetxt(os.path.join(run_dir, "icp_transformation.txt"), icp_matrix)

            # End timing the total process for this run (ends after ICP refinement)
            total_runtime = time.time() - total_start_time
            total_runtimes.append(total_runtime)

            # Compute metrics
            ransac_translation_error, ransac_rotation_error = compute_metrics(
                ransac_matrix, ground_truth
            )
            icp_translation_error, icp_rotation_error = compute_metrics(
                icp_matrix, ground_truth
            )

            # Compute ADD metrics
            R_pred = icp_matrix[:3, :3]
            t_pred = icp_matrix[:3, 3]
            R_gt = ground_truth[:3, :3]
            t_gt = ground_truth[:3, 3]
            add_error = add_metric(np.asarray(source.points), R_gt, t_gt, R_pred, t_pred)

            # Save metrics to lists
            ransac_translation_errors.append(ransac_translation_error)
            ransac_rotation_errors.append(ransac_rotation_error)
            icp_translation_errors.append(icp_translation_error)
            icp_rotation_errors.append(icp_rotation_error)
            ransac_runtimes.append(ransac_time)
            icp_runtimes.append(icp_time)
            add_metrics.append(add_error)

            # Save runtime information for this run
            with open(os.path.join(run_dir, "runtime.txt"), "w") as f:
                f.write(f"Total Runtime: {total_runtime:.4f} seconds\n")
                f.write(f"Descriptor Computation Time: {descriptor_time:.4f} seconds\n")
                f.write(f"RANSAC Runtime: {ransac_time:.4f} seconds\n")
                f.write(f"ICP Runtime: {icp_time:.4f} seconds\n")

            # Save metrics for this run
            with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
                f.write(f"RANSAC Translation Error: {ransac_translation_error:.6f} mm\n")
                f.write(f"RANSAC Rotation Error: {ransac_rotation_error:.6f} deg\n")
                f.write(f"ICP Translation Error: {icp_translation_error:.6f} mm\n")
                f.write(f"ICP Rotation Error: {icp_rotation_error:.6f} deg\n")
                f.write(f"ADD: {add_error:.6f} deg\n")

            # Overlay 3D points on 2D image and save (excluded from total runtime)
            print("Creating overlay image...")
            overlay_point_cloud_on_image(
                source, image, icp_matrix, intrinsic_matrix,
                os.path.join(run_dir, "overlay.png")
            )

            print(f"{scene_name} - Run {i+1} results saved in {run_dir}")

        # Save overall metrics to CSV for this scene
        metrics = {
            "Run": list(range(1, num_runs+1)),
            "RANSAC Translation Error (mm)": ransac_translation_errors,
            "RANSAC Rotation Error (deg)": ransac_rotation_errors,
            "ICP Translation Error (mm)": icp_translation_errors,
            "ICP Rotation Error (deg)": icp_rotation_errors,
            "ADD (mm)": add_metrics,
            "Descriptor Computation Time (s)": descriptor_times,
            "RANSAC Runtime (s)": ransac_runtimes,
            "ICP Runtime (s)": icp_runtimes,
            "Total Runtime (s)": total_runtimes
        }
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(os.path.join(output_dir, "metrics_over_runs.csv"), index=False)
        print(f"{scene_name} - Metrics saved to:", os.path.join(output_dir, "metrics_over_runs.csv"))

        # -------------------------------------------------
        # Compute means (averages) and standard deviations
        # -------------------------------------------------
        avg_total_runtime = np.mean(total_runtimes)
        sd_total_runtime = np.std(total_runtimes)

        avg_desc_time = np.mean(descriptor_times)
        sd_desc_time = np.std(descriptor_times)

        avg_ransac_runtime = np.mean(ransac_runtimes)
        sd_ransac_runtime = np.std(ransac_runtimes)

        avg_icp_runtime = np.mean(icp_runtimes)
        sd_icp_runtime = np.std(icp_runtimes)

        avg_ransac_trans_err = np.mean(ransac_translation_errors)
        sd_ransac_trans_err = np.std(ransac_translation_errors)

        avg_ransac_rot_err = np.mean(ransac_rotation_errors)
        sd_ransac_rot_err = np.std(ransac_rotation_errors)

        avg_icp_trans_err = np.mean(icp_translation_errors)
        sd_icp_trans_err = np.std(icp_translation_errors)

        avg_icp_rot_err = np.mean(icp_rotation_errors)
        sd_icp_rot_err = np.std(icp_rotation_errors)

        avg_add = np.mean(add_metrics)
        sd_add = np.std(add_metrics)

        # Print scene-level summary
        print(f"\n{scene_name} - Averages over runs:")
        print(f"Average Total Runtime (s): {avg_total_runtime:.6f}")
        print(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}")
        print(f"Average Registration Runtime (s): {avg_ransac_runtime:.6f}")
        print(f"Average Refinement Runtime (s): {avg_icp_runtime:.6f}")
        print(f"Average Registration Translation Error (mm): {avg_ransac_trans_err:.6f}")
        print(f"Average Registration Rotation Error (deg): {avg_ransac_rot_err:.6f}")
        print(f"Average Refinement Translation Error (mm): {avg_icp_trans_err:.6f}")
        print(f"Average Refinement Rotation Error (deg): {avg_icp_rot_err:.6f}")
        print(f"Average ADD (mm): {avg_add:.6f}")

        print(f"\n{scene_name} - Standard Deviation over runs:")
        print(f"Total Runtime (s) SD: {sd_total_runtime:.6f}")
        print(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}")
        print(f"Registration Runtime (s) SD: {sd_ransac_runtime:.6f}")
        print(f"Refinement Runtime (s) SD: {sd_icp_runtime:.6f}")
        print(f"Registration Translation Error (mm) SD: {sd_ransac_trans_err:.6f}")
        print(f"Registration Rotation Error (deg) SD: {sd_ransac_rot_err:.6f}")
        print(f"Refinement Translation Error (mm) SD: {sd_icp_trans_err:.6f}")
        print(f"Refinement Rotation Error (deg) SD: {sd_icp_rot_err:.6f}")
        print(f"ADD (mm) SD: {sd_add:.6f}")

        # -------------------------------------------------
        # Save average & SD metrics to average_metrics.txt
        # -------------------------------------------------
        avg_metrics_file = os.path.join(output_dir, "average_metrics.txt")
        with open(avg_metrics_file, "w") as f:
            f.write(f"Average Metrics over runs for {scene_name}:\n")
            f.write(f"Average Total Runtime (s): {avg_total_runtime:.6f}\n")
            f.write(f"Average Descriptor Computation Time (s): {avg_desc_time:.6f}\n")
            f.write(f"Average Registration Runtime (s): {avg_ransac_runtime:.6f}\n")
            f.write(f"Average Refinement Runtime (s): {avg_icp_runtime:.6f}\n")
            f.write(f"Average Registration Translation Error (mm): {avg_ransac_trans_err:.6f}\n")
            f.write(f"Average Registration Rotation Error (deg): {avg_ransac_rot_err:.6f}\n")
            f.write(f"Average Refinement Translation Error (mm): {avg_icp_trans_err:.6f}\n")
            f.write(f"Average Refinement Rotation Error (deg): {avg_icp_rot_err:.6f}\n")
            f.write(f"Average ADD (mm): {avg_add:.6f}\n")

            f.write("\nStandard Deviation over runs:\n")
            f.write(f"Total Runtime (s) SD: {sd_total_runtime:.6f}\n")
            f.write(f"Descriptor Computation Time (s) SD: {sd_desc_time:.6f}\n")
            f.write(f"Registration Runtime (s) SD: {sd_ransac_runtime:.6f}\n")
            f.write(f"Refinement Runtime (s) SD: {sd_icp_runtime:.6f}\n")
            f.write(f"Registration Translation Error (mm) SD: {sd_ransac_trans_err:.6f}\n")
            f.write(f"Registration Rotation Error (deg) SD: {sd_ransac_rot_err:.6f}\n")
            f.write(f"Refinement Translation Error (mm) SD: {sd_icp_trans_err:.6f}\n")
            f.write(f"Refinement Rotation Error (deg) SD: {sd_icp_rot_err:.6f}\n")
            f.write(f"ADD (mm) SD: {sd_add:.6f}\n")

        print(f"{scene_name} - Averages and SD saved to: {avg_metrics_file}")

        # -------------------------------------------------
        # Create a dictionary including averages & SD for final CSV
        # -------------------------------------------------
        scene_metrics = {
            "Scene": scene_name,
            "Total Runtime (s) Mean": avg_total_runtime,
            "Total Runtime (s) SD": sd_total_runtime,

            "Descriptor Computation Time (s) Mean": avg_desc_time,
            "Descriptor Computation Time (s) SD": sd_desc_time,

            "Registration Runtime (s) Mean": avg_ransac_runtime,
            "Registration Runtime (s) SD": sd_ransac_runtime,

            "Refinement Runtime (s) Mean": avg_icp_runtime,
            "Refinement Runtime (s) SD": sd_icp_runtime,

            "Registration Translation Error (mm) Mean": avg_ransac_trans_err,
            "Registration Translation Error (mm) SD": sd_ransac_trans_err,

            "Registration Rotation Error (deg) Mean": avg_ransac_rot_err,
            "Registration Rotation Error (deg) SD": sd_ransac_rot_err,

            "Refinement Translation Error (mm) Mean": avg_icp_trans_err,
            "Refinement Translation Error (mm) SD": sd_icp_trans_err,

            "Refinement Rotation Error (deg) Mean": avg_icp_rot_err,
            "Refinement Rotation Error (deg) SD": sd_icp_rot_err,

            "ADD (mm) Mean": avg_add,
            "ADD (mm) SD": sd_add
        }

        # Append scene_metrics to all_metrics
        all_metrics.append(scene_metrics)

    # After processing all scenes, save all average metrics to a CSV
    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv(os.path.join(output_root_dir, "all_scenes_average_metrics.csv"), index=False)
    print("All scenes average metrics saved to:", os.path.join(output_root_dir, "all_scenes_average_metrics.csv"))

if __name__ == "__main__":
    main()

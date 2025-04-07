import torch
import os
import open3d as o3d

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# run VGGT to get the 3D model from the images
def create_model():
    image_dir_path = '/playpen-nas-ssd4/nofrahm/ModelGen/stable-virtual-camera/work_dirs/demo/img2trajvid_s-prob/vggt_prepped/flower/first-pass/samples-rgb'
    # images = './stable-virtual-camera/work_dirs/demo/img2trajvid_s-prob/vggt_prepped/flower/first-pass/samples-rgb'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

    # Load and preprocess example images (replace with your own image paths)
    # image_names = [image_path]
    image_names = [os.path.join(image_dir_path, image_name) for image_name in os.listdir(image_dir_path)]
    images = load_and_preprocess_images(image_names).to(device)

    # with torch.no_grad():
    #     with torch.cuda.amp.autocast(dtype=dtype):
    #         # Predict attributes including cameras, depth maps, and point maps.
    #         predictions = model(images)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        breakpoint()

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            
        # Construct 3D Points from Depth Maps and Cameras
        # which usually leads to more accurate 3D points than point map branch
        point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                    extrinsic.squeeze(0), 
                                                                    intrinsic.squeeze(0))
        
        save_point_cloud(point_map_by_unprojection, filename="output.ply")

        # Predict Tracks
        # choose your own points to track, with shape (N, 2) for one scene
        query_points = torch.FloatTensor([[100.0, 200.0], 
                                            [60.72, 259.94]]).to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])


def save_point_cloud(points_tensor, filename="output.ply"):
    # points_tensor: numpy array of shape (S, H, W, 3)
    # Flatten the tensor to shape (S*H*W, 3)
    points = points_tensor.reshape(-1, 3)
    
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")


def main():
    # generate_views()
    create_model()


if __name__ == "__main__":
    main()

# python /playpen-nas-ssd4/nofrahm/ModelGen/vggt/custom_run.py
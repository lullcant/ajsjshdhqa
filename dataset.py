import os
import torch.utils
import torch.utils.data
import trimesh
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points
import h5py
import pyvista as pv
import re
def volume_to_point_cloud_tensor(
    volume: torch.Tensor, 
    voxel_size=(0.2, 0.2, 0.2), 
    origin=None
):
    """
    将体积数据转换为点云格式，全部使用 PyTorch 张量。
    当 origin 是 [B, 3] 形状时，为每个 batch 提供单独的原点坐标。

    Args:
        volume (torch.Tensor): 形状为 [B, C, D, H, W] 的体积张量。
                              其中 B 是 batch 维度, C 是通道数(通常为 1)，
                              D/H/W 分别对应深度/高度/宽度。
        voxel_size (tuple or list): 每个体素的物理尺寸 (dx, dy, dz)。
                                    也可以是一个 shape=[3] 的 torch.Tensor。
        origin (torch.Tensor): 如果是 shape=[B, 3]，表示每个 batch 的原点。
                               如果是 shape=[3]，则对所有 batch 使用相同原点。

    Returns:
        torch.Tensor: 形状为 [N, 4] 的点云张量，其中 N 为总点数。  
                      最后的 4 个维度含义为 [batch_index,x, y, z]。
    """
    device = volume.device
    B, C, D, H, W = volume.shape
    assert C == 1, "输入体素的通道数应为 1"

    # 构造 voxel_size_tensor
    if not torch.is_tensor(voxel_size):
        voxel_size_tensor = torch.tensor(voxel_size, 
                                         device=device, 
                                         dtype=torch.float32)  # [3]
    else:
        voxel_size_tensor = voxel_size.to(device=device, dtype=torch.float32)

    # 处理 origin：可能是 [3] 或 [B, 3]
    if origin is None:
        # 如果不传 origin，就默认是 0
        origin_tensor = torch.zeros((B, 3), device=device, dtype=torch.float32)
    else:
        origin_tensor = origin.to(device=device, dtype=torch.float32)
        # 若只给了一个全局 origin = [3]，则需要扩展到 [B, 3]
        if origin_tensor.ndim == 1:
            origin_tensor = origin_tensor.unsqueeze(0).expand(B, -1)  # [B, 3]
        assert origin_tensor.shape == (B, 3), \
            "origin 应该是 [B, 3] 或者 [3]，并在内部扩展为 [B, 3]"

    # 前景点筛选：将体素值先做 sigmoid，再用阈值 0.5 筛选
    foreground_prob = torch.sigmoid(volume)                     # [B, 1, D, H, W]
    foreground_mask = (foreground_prob > 0.5).squeeze(1)        # [B, D, H, W]
    # 取出前景位置索引
    nonzero_indices = torch.nonzero(foreground_mask, as_tuple=False)
  
    # 依次取出对应维度
    b = nonzero_indices[:, 0].long()  # batch index
    d = nonzero_indices[:, 1].long()  # depth index
    h = nonzero_indices[:, 2].long()  # height index
    w = nonzero_indices[:, 3].long()  # width index

    coords_dhw = torch.stack([d ,h, w], dim=-1).float()  # [N, 3]
    # 对于每个点，根据 batch index 去加各自的 origin
    coords_xyz = origin_tensor[b] + coords_dhw * voxel_size_tensor  # [N, 3]

    
    b_float = b.float().unsqueeze(-1)  # [N, 1]
    point_cloud = torch.cat([b_float,coords_xyz], dim=-1)  # [N, 4]

    return point_cloud



import numpy as np

def create_voxelwithnormal_grid(vertices, min_bound, max_bound, voxel_size=(0.2, 0.2, 0.2)):
    """
    创建体素网格，并将点、法向量及曲率映射到网格上。
    
    Args:
        vertices (np.ndarray): 点云顶点数组，形状为 (N, 7)，前3列为点坐标，接下来3列为法向量，最后1列为曲率。
        min_bound (np.ndarray): 裁剪区域的最小边界，形状为 (3,)。
        max_bound (np.ndarray): 裁剪区域的最大边界，形状为 (3,)。
        voxel_size (tuple): 每个体素的大小，形状为 (3,)。
    
    Returns:
        np.ndarray: 体素网格，形状为 [5, D, H, W]。
                    通道0为占用信息，通道1-3为法向量平均值，通道4为曲率平均值。
    """
    # 分离点坐标、法向量和曲率
    points = vertices[:, :3]
    normals = vertices[:, 3:6]
    curvatures = vertices[:, 6]
    
    # 计算网格大小
    voxel_size = np.array(voxel_size)
    grid_shape = np.ceil((max_bound - min_bound) / voxel_size).astype(int)
    D, H, W = grid_shape  # 深度、高度、宽度
    
    # 初始化占用通道、法向量通道和曲率通道
    occupancy = np.zeros(grid_shape, dtype=np.uint8)
    normals_sum = np.zeros((3, D * H * W), dtype=np.float32)
    curvature_sum = np.zeros(D * H * W, dtype=np.float32)
    counts = np.zeros(D * H * W, dtype=np.int32)
    
    # 将点映射到体素网格
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    
    # 确保索引在体素网格范围内
    valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < grid_shape), axis=1)
    voxel_indices = voxel_indices[valid_mask]
    valid_normals = normals[valid_mask]
    valid_curvatures = curvatures[valid_mask]
    
    if voxel_indices.size == 0:
        # 如果没有有效的点，返回空的体素网格
        voxel_grid = np.zeros((5, D, H, W), dtype=np.float32)
        return voxel_grid
    
    # 计算线性索引
    linear_indices = voxel_indices[:, 0] * (H * W) + voxel_indices[:, 1] * W + voxel_indices[:, 2]
    
    # 设置占用信息
    occupancy.flat[linear_indices] = 1
    
    # 累加法向量
    for i in range(3):
        normals_sum[i].flat[linear_indices] += valid_normals[:, i]
    
    # 累加曲率
    np.add.at(curvature_sum, linear_indices, valid_curvatures)
    
    # 统计每个体素中的点数
    np.add.at(counts, linear_indices, 1)
    
    # 避免除以零
    counts_nonzero = counts.copy()
    counts_nonzero[counts_nonzero == 0] = 1
    
    # 计算平均法向量
    normals_avg = normals_sum / counts_nonzero
    normals_avg = normals_avg.reshape(3, D, H, W)
    
    # 计算平均曲率
    average_curvature = np.abs(curvature_sum / counts_nonzero).reshape(D,H,W)
    
    # 组装最终的体素网格
    voxel_grid = np.zeros((5, D, H, W), dtype=np.float32)
    voxel_grid[0] = occupancy  # 占用通道
    voxel_grid[1:4] = normals_avg  # 法向量通道
    voxel_grid[4] = average_curvature  # 曲率通道
    
    return voxel_grid


def create_voxel_grid(vertices, min_bound, max_bound, voxel_size=(0.2,0.2,0.2)):
    """
    创建体素网格。
    
    Args:
        vertices (np.ndarray): 点云顶点数组，形状为 (N, 3)。
        min_bound (np.ndarray): 裁剪区域的最小边界。
        max_bound (np.ndarray): 裁剪区域的最大边界。
        voxel_size (tuple): 每个体素的大小，形状为 (3,)。
    
    Returns:
        np.ndarray: 体素网格，形状由裁剪范围和体素大小决定。
    """
    # 计算网格大小
    grid_shape = np.ceil((max_bound - min_bound) / np.array(voxel_size)).astype(int)
    
    # 初始化空体素网格
    voxel_grid = np.zeros(grid_shape, dtype=np.uint8)
    
    # 将顶点映射到体素网格
    voxel_indices = np.floor((vertices - min_bound) / np.array(voxel_size)).astype(int)
    
    # 确保索引在体素网格范围内
    valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < grid_shape), axis=1)
    voxel_indices = voxel_indices[valid_mask]
    
    # 设置对应的体素为 1
    voxel_grid[tuple(voxel_indices.T)] = 1
    
    return voxel_grid


# class IOS_Dataset(Dataset):
#        '''
#         旧实现，没有返回点云
#        ''''''
#     def __init__(self, root_dir, is_train = True, crop_size=(2.56, 2.56, 2.56), voxel_size=(0.2, 0.2, 0.2)):
#         """
#         Initialize the dataset.
        
#         Args:
#             root_dir (str): The root directory containing the subdirectories.
#             crop_size (tuple): The size of the cropping region (in cm).
#             voxel_size (tuple): The voxel size (in mm).
#         """
#         self.root_dir = Path(root_dir)
#         self.crop_size = crop_size
#         self.voxel_size = voxel_size
        
#         # Define the subdirectories (11, 12, 21, etc.)
#         self.subdirs = ['11', '12', '21', '22', '31', '32', '41', '42']
        
#         # Prepare the list of data paths
#         self.data_paths = []
#         for subdir in self.subdirs:
#             subdir_path = self.root_dir / subdir
#             if is_train:
#                 case_dir = subdir_path / 'train' 
#             else:
#                 case_dir = subdir_path / 'test'
#             for case in os.listdir(case_dir):
#                 abs_case = os.path.join(case_dir,case)
#                 crown_file = os.path.join(abs_case, 'crown.ply')
#                 pna_crop_file = os.path.join(abs_case ,'pna_crop.ply')
#                 self.data_paths.append((case, crown_file, pna_crop_file))
#         #print(self.data_paths)
#     def __len__(self):
#         return len(self.data_paths)
    
#     def __getitem__(self, idx):
#         # Get the directory name and file paths for this index
#         dirpath, crown_file, pna_crop_file = self.data_paths[idx]
        
#         # Read the meshes
#         crown_mesh = trimesh.load(crown_file)
#         pna_crop_mesh = trimesh.load(pna_crop_file)
        
#         # Get the center for cropping (use crown mesh center as reference)
#         crown_min_bound, crown_max_bound = crown_mesh.bounds
#         crown_center = (crown_min_bound + crown_max_bound) / 2
        
#         # Create voxel grids for both meshes
#         crown_voxel_grid = create_voxel_grid(crown_mesh, crown_center, crop_size=self.crop_size, voxel_size=self.voxel_size)
#         pna_crop_voxel_grid = create_voxel_grid(pna_crop_mesh, crown_center, crop_size=self.crop_size, voxel_size=self.voxel_size)
        
#         # Convert numpy arrays to tensors
#         crown_tensor = torch.tensor(crown_voxel_grid, dtype=torch.float32).unsqueeze(0)
#         pna_crop_tensor = torch.tensor(pna_crop_voxel_grid, dtype=torch.float32).unsqueeze(0)
    
#         # Return both voxel grids as a tuple, along with the parent directory name
#         return pna_crop_tensor,crown_tensor, dirpath

class IOS_Dataset(Dataset):
    def __init__(self, root_dir, is_train=True, crop_size=(2.00, 2.00, 2.00), voxel_size=(0.15625, 0.15625, 0.15625)):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): The root directory containing the subdirectories.
            crop_size (tuple): The size of the cropping region (in cm).
            voxel_size (tuple): The voxel size (in mm).
        """
        self.root_dir = Path(root_dir)
        self.crop_size = crop_size
        self.voxel_size = voxel_size
        # Define the subdirectories (11, 12, 21, etc.)
        #self.subdirs = ['11', '12', '21', '22', '31', '32', '41', '42']
        self.subdirs = ['11', '12', '13','14', '15', '16', '17', 
 '21', '22','23', '24', '25', '26', '27', 
 '31', '32','33','34', '35', '36', '37', 
 '41', '42', '43','44', '45', '46', '47']
        # Prepare the list of data paths
        self.data_paths = []
        for subdir in self.subdirs:
            subdir_path = self.root_dir / subdir
            if is_train:
                case_dir = subdir_path / 'train' 
            else:
                case_dir = subdir_path / 'test'
            for case in os.listdir(case_dir):
                abs_case = os.path.join(case_dir, case)
                crown_file = os.path.join(abs_case, 'crown.h5')
                pna_crop_file = os.path.join(abs_case, 'pna_crop.h5')
                self.data_paths.append((abs_case, crown_file, pna_crop_file))

    def crop_mesh(self, mesh, center, crop_size):
        """
        Crop the mesh to a specific region around the center.
        
        Args:
            mesh (trimesh.Trimesh): The mesh to crop.
            center (numpy.ndarray): The center point for cropping.
            crop_size (tuple): The size of the cropping region (in number of points or voxel grid units).
        
        Returns:
            numpy.ndarray: The cropped point cloud.
        """
        # Convert mesh to numpy array of points
        points = mesh
        
        # Compute the crop bounds
        half_crop_size = 10 * np.array(crop_size) / 2  # Half of the crop size
        min_bound = center - half_crop_size
        max_bound = center + half_crop_size
        
        # Select points within the bounds
        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        cropped_points = points[mask]
        
        return cropped_points
    
    def __len__(self):
        return len(self.data_paths)
    
   
    def __getitem__(self, idx):
        # Get the directory name and file paths for this index
        dirpath, crown_file, pna_crop_file = self.data_paths[idx]
       
        # Read crown vertices using h5py
        with h5py.File(crown_file, 'r') as f:
            crown_vertices = np.array(f['vertices'])  # 读取 crown 的顶点信息
            crown_normals = np.array(f['normals'])
            crown_curvatures = np.array(f['curvatures']).reshape(-1,1)
        # Read pna_crop vertices using h5py
        with h5py.File(pna_crop_file, 'r') as f:
            pna_crop_vertices = np.array(f['vertices'])  # 读取 pna_crop 的顶点信息
        
        # Calculate bounds and center for crown vertices
        crown_min_bound = crown_vertices.min(axis=0)
        crown_max_bound = crown_vertices.max(axis=0)
        crown_center = (crown_min_bound + crown_max_bound) / 2  # 中心点
        
        # Calculate crop bounds
        half_crop_size = np.array(self.crop_size) * 10 / 2  # Convert cm to mm and divide by 2
        min_bound_crop = crown_center - half_crop_size
        max_bound_crop = crown_center + half_crop_size
        
        # Create voxel grids for both meshes
       
        crown_voxel_grid = create_voxelwithnormal_grid(np.concatenate([crown_vertices,crown_normals,crown_curvatures],axis=1), min_bound_crop, max_bound_crop, self.voxel_size)
        pna_crop_voxel_grid = create_voxel_grid(pna_crop_vertices, min_bound_crop, max_bound_crop, self.voxel_size)
        
        # Crop crown vertices to a point cloud
        crown_mesh_crop = self.crop_mesh(crown_vertices, crown_center, self.crop_size)
        point_cloud_crown_tensor = torch.tensor(crown_mesh_crop, dtype=torch.float32)
        point_cloud_crown_tensor = sample_farthest_points(point_cloud_crown_tensor[None,:],K=2048)[0].squeeze()
        # Convert voxel grids to tensors
        crown_tensor = torch.tensor(crown_voxel_grid, dtype=torch.float32)
        
        pna_crop_tensor = torch.tensor(pna_crop_voxel_grid, dtype=torch.float32).unsqueeze(0)
        
        # Return data along with min_bound_crop
        return pna_crop_tensor, crown_tensor, point_cloud_crown_tensor, min_bound_crop, dirpath
'''
src\model\est_coord.py
'''

from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn

from ..config import Config
from ..vis import Vis


class EstCoordNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Estimate the coordinates in the object frame for each object point.
        """
        super().__init__()
        self.config = config


        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_1024=nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.ReLU()
        )
        # concat 1024 and 64
        self.mlp2 = nn.Sequential(
            nn.Linear(1024 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 is the coordinate
        )



    def forward(
        self, pc: torch.Tensor, coord: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstCoordNet

        Parameters
        ----------
        pc: torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        coord: torch.Tensor
            Ground truth coordinates in the object frame, shape \(B, N, 3\)

        Returns
        -------
        float
            The loss value according to ground truth coordinates
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        N=pc.shape[1]
        pc = pc.transpose(1, 2)
        # (B,3,N)
        mlp1_64_output=self.mlp1(pc)
        x=self.conv_1024(mlp1_64_output)
        pooling_x = torch.max(x, dim=2, keepdim=True)[0]  # (B, 1024, 1)
        pooling_x = pooling_x.expand(-1, -1, N)          # (B, 1024, N)
        concat_x = torch.cat([mlp1_64_output, pooling_x], dim=1)  # (B, 64+1024, N)

        concat_x = concat_x.transpose(1, 2)  # (B, N, 1088)
        pred_coord=self.mlp2(concat_x)
        
        # (B,N,3)
        loss = torch.norm(pred_coord - coord, dim=2).sum()*(1/20)
        metric = dict(
            loss=loss,
            # additional metrics you want to log
        )
        return loss, metric


    def est(self, pc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate translation and rotation in the camera frame

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)

        Returns
        -------
        trans: torch.Tensor
            Estimated translation vector in camera frame, shape \(B, 3\)
        rot: torch.Tensor
            Estimated rotation matrix in camera frame, shape \(B, 3, 3\)

        Note
        ----
        The rotation matrix should satisfy the requirement of orthogonality and determinant 1.

        We don't have a strict limit on the running time, so you can use for loops and numpy instead of batch processing and torch.

        The only requirement is that the input and output should be torch tensors on the same device and with the same dtype.
        """
        N=pc.shape[1]
        pc = pc.transpose(1, 2)
        # (B,3,N)
        mlp1_64_output=self.mlp1(pc)
        pc = pc.transpose(1, 2)
        x=self.conv_1024(mlp1_64_output)
        pooling_x = torch.max(x, dim=2, keepdim=True)[0]  # (B, 1024, 1)
        pooling_x = pooling_x.expand(-1, -1, N)          # (B, 1024, N)
        concat_x = torch.cat([mlp1_64_output, pooling_x], dim=1)  # (B, 64+1024, N)
        concat_x = concat_x.transpose(1, 2)  # (B, N, 1088)
        
        # coordinates in the object frame
        pred_coord=self.mlp2(concat_x)
        
        '''"Using RANSAC!!!!"
        --------- 
        RANSAC 部分（纯 torch 实现，无循环） 
        ----------
        '''
        
        '''
        --- Parameters---
        M: points cloud in camera frame (B,N,3)
        N: predict point cloud in object frame (B,N,3)
        K: Number of candicate
        inlier threshold
        '''
        M = pc          # (B, N, 3)
        N_pts = pred_coord  # (B, N, 3)
        B=M.shape[0]
        K = 100  
        inlier_thresh = 0.03  


        '''
        --- Prepare for K candidates---
        1. for each candidate, sample 3 points
        2. for 3 points: calculate centers and Rot, Trans.
        Goal min ||M^T-RN^T||
        '''
        rand_idx = torch.randint(0, N, (B, K, 3), device=M.device)
        batch_idx = torch.arange(B, device=M.device).view(B, 1, 1).expand(B, K, 3)
        M_samples = M[batch_idx, rand_idx, :]   
        N_samples = N_pts[batch_idx, rand_idx, :] 

        centroid_M = M_samples.mean(dim=2, keepdim=True)# (B, K, 1, 3)
        centroid_N = N_samples.mean(dim=2, keepdim=True)

        M_centered = M_samples - centroid_M  # (B, K, 3, 3)
        N_centered = N_samples - centroid_N  # (B, K, 3, 3)

        # H = N_centered^T @ M_centered 
        # H: the last dim is coordinate
        H = torch.matmul(N_centered.transpose(-2, -1), M_centered)  # (B, K, 3, 3)
        U, _, V = torch.svd(H)   # U, Vh: (B, K, 3, 3)

        R_candidates = torch.matmul(V, U.transpose(-2, -1))  # (B, K, 3, 3)

        # Check and validate Rotation matrix
        det = torch.det(R_candidates)   # (B, K)
        mask_det = det < 0                      # shape (B, K)
        D = torch.eye(3, device=M.device).unsqueeze(0).unsqueeze(0).expand(B, K, 3, 3).clone()
        b_idx, k_idx = torch.where(mask_det)   # shape (N,), (N,)
        D[b_idx, k_idx, 2, 2] = -1
        R_candidates = torch.matmul(V, torch.matmul(D, U.transpose(-2, -1)))  # (B, K, 3, 3)
        # t = centroid_M - R * centroid_N
        t_candidates = centroid_M.transpose(-2, -1) - torch.matmul(R_candidates, centroid_N.transpose(-2, -1))  # (B, K, 3, 1)

        '''
        --- Evaluation---
        1. Transformation for parallel computing
        2. Use broadcast
        3. projection and compute error
        '''
        N_pts_exp = N_pts.unsqueeze(1).unsqueeze(-1) # (B, 1, N, 3, 1)
        R_exp = R_candidates.unsqueeze(2)  # (B, K, 3, 3)->(B, K, 1, 3, 3)
        t_exp = t_candidates.unsqueeze(2)   # (B, K, 3, 1)->(B, K, 1, 3, 1)
        # x' = R * N + t
        # (B, K, N, 3, 1) = R_exp (B, K,1,3,3) @ N_pts_exp_ (B,1,N,3,1)
        transformed_N = torch.matmul(R_exp, N_pts_exp)  # (B, K, N, 3, 1)
        transformed_N = (transformed_N + t_exp).squeeze(-1)  # (B, K, N, 3)
        # print("transformed_N shape :",transformed_N.shape) # torch.Size([16, 100, 1024, 3])
        M_exp = M.unsqueeze(1)  # (B, 1, N, 3)
        errors = torch.norm(M_exp-transformed_N, dim=-1)  # (B, K, N)
        # count inliers
        inlier_counts = (errors < inlier_thresh).float().sum(dim=-1)  # (B, K)

        '''
        For each batch, get the best_idx, find all inlier points defined by R and T 
        '''
        best_idx = inlier_counts.argmax(dim=1)  # (B,)
        #print(best_idx)
        points_mask=errors < inlier_thresh # (B, K, N) inliers: 1; outliers: 0
        best_mask = points_mask[torch.arange(B), best_idx, :].unsqueeze(-1).float() # (B, N, 1)
        
        '''mask outliers: 
        If the point is outliers, delete the points (set coordinate to zero) of the source and target PC simultaneously
        '''
        mask = best_mask.squeeze(-1) #(B,N)
        counts = mask.sum(dim=1, keepdim=True) # (B, 1)
        # Center the point clouds: use inliers to calculate centers
        src_pc=N_pts
        target_pc=M
        cent_tar = (target_pc * mask.unsqueeze(-1)).sum(dim=1, keepdim=True)  / counts.unsqueeze(-1) # (B, 1, 3)
        cent_src = (src_pc * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / counts.unsqueeze(-1)
        tar_cend = (target_pc - cent_tar) * mask.unsqueeze(-1)   # (B, N, 3)
        src_cend = (src_pc - cent_src) * mask.unsqueeze(-1)   # (B, N, 3)
        
        H = torch.matmul(src_cend.transpose(1, 2),tar_cend)  # (B, 3, 3)
        # SVD on H
        U, _, V = torch.svd(H) # (B, 3, 3)
        # if det=-1, modify the R(3,3) by multiply -1
        det=torch.det(torch.matmul(V,U.transpose(1,2)))
        D = torch.eye(3, device=M.device).unsqueeze(0).repeat(B, 1, 1)
        D[det < 0, 2, 2] = -1
        rot=torch.matmul(V, torch.matmul(D, U.transpose(1,2)))
        trans = cent_tar.squeeze(1) - torch.matmul(rot, cent_src.transpose(1, 2)).squeeze(-1)
        return trans,rot


    

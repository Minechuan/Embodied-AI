from typing import Tuple, Dict
import torch
from torch import nn

from ..config import Config


class EstPoseNet(nn.Module):

    config: Config

    def __init__(self, config: Config):
        """
        Directly estimate the translation vector and rotation matrix.
        """
        super().__init__()
        self.config = config

        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # 预测 6 个数，即旋转矩阵的前两列。最后一列可以求出来。
            nn.Linear(256, 9)  # For example: output 3 for 3D translation + 6 rotation matrix（first 2 columns）
        )



    def forward(
        self, pc: torch.Tensor, trans: torch.Tensor, rot: torch.Tensor, **kwargs
    ) -> Tuple[float, Dict[str, float]]:
        """
        Forward of EstPoseNet

        Parameters
        ----------
        pc : torch.Tensor
            Point cloud in camera frame, shape \(B, N, 3\)
        trans : torch.Tensor
            Ground truth translation vector in camera frame, shape \(B, 3\)
        rot : torch.Tensor
            Ground truth rotation matrix in camera frame, shape \(B, 3, 3\)

        Returns
        -------
        float
            The loss value according to ground truth translation and rotation
        Dict[str, float]
            A dictionary containing additional metrics you want to log
        """
        # raise NotImplementedError("You need to implement the forward function")
        pc = pc.transpose(1, 2)
        
        
        latent_x=self.mlp1(pc)
        # max pooling 
        pooling_x = torch.max(latent_x, dim=2)[0]
        net_output=self.mlp2(pooling_x)
        T_pred=net_output[:,0:3]

        R_cols = net_output[:, 3:].view(-1, 3, 2)  # (B, 3, 2)
        col1 = R_cols[:, :, 0]  # (B, 3)
        col2 = R_cols[:, :, 1]  # (B, 3)

        # Gram-Schmidt
        col1 = nn.functional.normalize(col1, dim=1)
        col2 = col2 - (col1 * (col1 * col2).sum(dim=1, keepdim=True)) 
        col2 = nn.functional.normalize(col2, dim=1)
        col3 = torch.cross(col1, col2, dim=1)  

        R_pred = torch.stack([col1, col2, col3], dim=2)  # (B, 3, 3)
        
        def so3_dist(R_pred,R_gt):
            # calculate SO3 distance: in utils I have numpy dist, but here is tensor 
            R_relative = torch.matmul(R_pred.transpose(1, 2), R_gt)
            trace = R_relative.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1)
            trace = torch.clamp((trace - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)
            return torch.acos(trace)
        


        # Use the distance of two rotation matrix Geodesic loss
        T_loss=nn.functional.mse_loss(T_pred, trans)
        R_loss=so3_dist(R_pred,rot).mean()
        loss = T_loss+R_loss
        metric = dict(
            loss=loss.detach(),
            # additional metrics you want to log
            T_loss = T_loss.detach(),
            R_loss = R_loss.detach(),
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
        """
        # raise NotImplementedError("You need to implement the est function")
        pc = pc.transpose(1, 2)
        
        with torch.no_grad():
            latent_x=self.mlp1(pc)
            pooling_x = torch.max(latent_x, dim=2)[0]
            # (B,1024,)
            net_output=self.mlp2(pooling_x)
            
            trans=net_output[:,0:3]
            R_cols = net_output[:, 3:].view(-1, 3, 2)  # (B, 3, 2)
            col1 = R_cols[:, :, 0]  # (B, 3)
            col2 = R_cols[:, :, 1]  # (B, 3)

            # Gram-Schmidt
            col1 = nn.functional.normalize(col1, dim=1)
            col2 = col2 - (col1 * (col1 * col2).sum(dim=1, keepdim=True)) 
            col2 = nn.functional.normalize(col2, dim=1)
            col3 = torch.cross(col1, col2, dim=1)  

            rot = torch.stack([col1, col2, col3], dim=2)  # (B, 3, 3))

        return (trans,rot)




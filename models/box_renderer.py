import torch
import torch.nn as nn

class BoxRenderer(nn.Module):
    """Draw bounding boxes onto image tensors using differentiable operations.
    
    Uses sigmoid-based soft masks for edge rendering, enabling
    gradient flow through bbox coordinate parameters.
    """
    
    def __init__(self, color=(1.0, 0.0, 0.0), line_width=2, sigma=1.0):
        """
        Args:
            color: RGB color tuple for box edges, values in [0, 1] range
            line_width: Width of box edges in pixels
            sigma: Softness parameter for sigmoid edge function (higher = softer edges)
        """
        super().__init__()
        self.color = color
        self.line_width = line_width
        self.sigma = sigma
    
    def forward(self, images, bboxes):
        """Draw bounding boxes on images.
        
        Args:
            images: (B, 3, H, W) float tensor, values in [0, 1]
            bboxes: (B, N, 4) float tensor, values [cx, cy, w, h] normalized [0, 1]
            
        Returns:
            (B, 3, H, W) float tensor with boxes drawn
        """
        B, C, H, W = images.shape
        
        # Check if all bboxes are zero -> early exit, return images unchanged
        if torch.all(bboxes == 0.0):
            return images
        
        device = images.device
        
        # Create coordinate grids
        y_grid = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
        
        combined_mask = torch.zeros(B, H, W, device=device)
        
        for n in range(bboxes.shape[1]):
            # Extract bbox params [cx, cy, w, h] normalized -> pixel coords
            cx = bboxes[:, n, 0] * W  # (B,)
            cy = bboxes[:, n, 1] * H
            bw = bboxes[:, n, 2] * W
            bh = bboxes[:, n, 3] * H
            
            # Compute x1, y1, x2, y2 (pixel coords)
            x1 = cx - bw / 2  # (B,)
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            
            # Reshape for broadcasting: (B,) -> (B, 1, 1)
            x1 = x1.view(B, 1, 1)
            y1 = y1.view(B, 1, 1)
            x2 = x2.view(B, 1, 1)
            y2 = y2.view(B, 1, 1)
            
            # Create soft masks for each edge using sigmoid functions
            # Left edge: activation toward x1, deactivation after x1+line_width
            left_edge = torch.sigmoid((x_grid - x1) / self.sigma)
            left_bound = 1.0 - torch.sigmoid((x_grid - (x1 + self.line_width)) / self.sigma)
            
            # Right edge: activation toward x2-line_width, deactivation after x2
            right_edge = torch.sigmoid((x_grid - (x2 - self.line_width)) / self.sigma)
            right_bound = 1.0 - torch.sigmoid((x_grid - x2) / self.sigma)
            
            # Top edge
            top_edge = torch.sigmoid((y_grid - y1) / self.sigma)
            top_bound = 1.0 - torch.sigmoid((y_grid - (y1 + self.line_width)) / self.sigma)
            
            # Bottom edge
            bottom_edge = torch.sigmoid((y_grid - (y2 - self.line_width)) / self.sigma)
            bottom_bound = 1.0 - torch.sigmoid((y_grid - y2) / self.sigma)
            
            # Vertical edges mask: pixel is on left OR right edge
            vertical_mask = left_edge * left_bound + right_edge * right_bound
            
            # Horizontal edges mask: pixel is on top OR bottom edge
            horizontal_mask = top_edge * top_bound + bottom_edge * bottom_bound
            
            # Full mask: pixel is on any edge (clamp to [0,1])
            edge_mask = torch.clamp(horizontal_mask + vertical_mask - 
                                    horizontal_mask * vertical_mask, 0.0, 1.0)
            
            # Only apply mask for non-zero bboxes
            is_valid = (bboxes[:, n].abs().sum(dim=1) > 0).float().view(B, 1, 1)
            combined_mask = combined_mask + edge_mask * is_valid
        
        # Clamp final mask to [0, 1]
        combined_mask = torch.clamp(combined_mask, 0.0, 1.0)
        
        # Blend: for each channel, mix image with box color
        result = images.clone()
        for c in range(C):
            result[:, c] = images[:, c] * (1.0 - combined_mask) + self.color[c] * combined_mask
        
        return result

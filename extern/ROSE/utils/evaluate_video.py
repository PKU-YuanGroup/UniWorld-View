import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from tqdm import tqdm
import cv2

class VideoQualityEvaluator:
    def __init__(self, device='cuda'):
        """Initialize video quality evaluator with specified computation device
        
        Args:
            device (str): Computation device ('cuda' or 'cpu')
        """
        self.device = device
        # Initialize LPIPS model (perceptual similarity metric)
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
    
    def _preprocess_frame(self, frame):
        """Convert frame to standardized format for evaluation
        
        Args:
            frame: Input frame (numpy array or torch tensor)
            
        Returns:
            Processed frame in HWC format with values in [0,1]
        """
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        
        # Normalize to [0,1] if needed
        if frame.max() > 1:
            frame = frame / 255.0
        # Convert CHW to HWC if needed
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        return frame
    
    def calculate_psnr(self, vid1, vid2):
        """Calculate average PSNR between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean PSNR value across all frames
        """
        psnrs = []
        for f1, f2 in zip(vid1, vid2):
            f1 = self._preprocess_frame(f1)
            f2 = self._preprocess_frame(f2)
            # Calculate PSNR for this frame pair
            psnrs.append(psnr(f1, f2, data_range=1.0))
        return np.mean(psnrs)
    
    def calculate_ssim(self, vid1, vid2):
        """Calculate average SSIM between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean SSIM value across all frames
        """
        ssims = []
        for f1, f2 in zip(vid1, vid2):
            f1 = self._preprocess_frame(f1)
            f2 = self._preprocess_frame(f2)
            # Calculate SSIM for this frame pair (multichannel for color images)
            ssims.append(ssim(f1, f2, channel_axis=2, data_range=1.0))
        return np.mean(ssims)
    
    def calculate_lpips(self, vid1, vid2):
        """Calculate average LPIPS (perceptual similarity) between two videos
        
        Args:
            vid1: First video (list/array of frames)
            vid2: Second video (list/array of frames)
            
        Returns:
            Mean LPIPS value across all frames (lower is better)
        """
        lpips_values = []
        for f1, f2 in zip(vid1, vid2):
            # Convert to torch tensor if needed
            if not isinstance(f1, torch.Tensor):
                f1 = torch.from_numpy(f1).permute(2, 0, 1).unsqueeze(0).float()  # HWC -> 1CHW
                f2 = torch.from_numpy(f2).permute(2, 0, 1).unsqueeze(0).float()
            
            # Normalize to [-1,1] if needed
            if f1.max() > 1:
                f1 = f1 / 127.5 - 1.0
                f2 = f2 / 127.5 - 1.0
            
            f1 = f1.to(self.device)
            f2 = f2.to(self.device)
            
            # Calculate LPIPS with no gradients
            with torch.no_grad():
                lpips_values.append(self.lpips_model(f1, f2).item())
        return np.mean(lpips_values)
    
    def evaluate_videos(self, generated_video, reference_video, metrics=['psnr','lpips','ssim']):
        """Comprehensive video quality evaluation between generated and reference videos
        
        Args:
            generated_video: Model-generated video [T,H,W,C] or [T,C,H,W]
            reference_video: Ground truth reference video [T,H,W,C] or [T,C,H,W]
            metrics: List of metrics to compute ('psnr', 'ssim', 'lpips')
            
        Returns:
            Dictionary containing computed metric values
        """
        results = {}
        
        # Verify video lengths match
        assert len(generated_video) == len(reference_video), "Videos must have same number of frames"
        
        # Calculate requested metrics
        if 'psnr' in metrics:
            results['psnr'] = self.calculate_psnr(generated_video, reference_video)
        
        if 'ssim' in metrics:
            results['ssim'] = self.calculate_ssim(generated_video, reference_video)
        
        if 'lpips' in metrics:
            results['lpips'] = self.calculate_lpips(generated_video, reference_video)
        
        return results


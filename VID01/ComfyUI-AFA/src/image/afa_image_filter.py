import torch
import numpy as np
import cv2

class AFAImageFilter:
    CATEGORY = "AFA"
    @classmethod    
    def INPUT_TYPES(s):
        return { "required":  {
            "a": ("IMAGE",),
            "kernel_size": ("INT", {
                "default": 5,
                "min": 3,      
                "max": 21,    
                "step": 2,     
                "display": "slider"
            }),
            "canny_low_thresh": ("INT", {
                "default": 50,
                "min": 0,      
                "max": 255,    
                "step": 10,     
                "display": "slider"
            }),
            "canny_high_thresh": ("INT", {
                "default": 200,
                "min": 0,      
                "max": 255,    
                "step": 10,     
                "display": "slider"
            }),
            "noise_strength": ("FLOAT", {
                "default": 1,
                "min": 0,      
                "max": 2,    
                "step": 0.1,     
                "display": "slider"
            }),

        }}
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("SMOOTH IMAGE", "CANNY IMAGE", "NOISE IMAGE")
    FUNCTION = "do_filter"
    
    def __init__(self):
        super().__init__()

    def do_filter(self, a, kernel_size, canny_low_thresh, canny_high_thresh, noise_strength):
        
        np_image = a[0].cpu().numpy()

        smooth_image = cv2.GaussianBlur(np_image,  (kernel_size, kernel_size), 0)

        gray_scale_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

        canny_image = cv2.Canny((gray_scale_image*255).astype(np.uint8),
                                canny_low_thresh,
                                canny_high_thresh)
        
        noise = np.random.normal(loc=0.0, scale=noise_strength, size=np_image.shape)

        noisey_image = np_image + noise
        noisey_image = np.clip(noisey_image, 0.0, 1.0)

        return (
            torch.from_numpy( smooth_image ).float().unsqueeze(0),
            torch.from_numpy( canny_image ).float().unsqueeze(0),
            torch.from_numpy( noisey_image ).float().unsqueeze(0),
        )
    


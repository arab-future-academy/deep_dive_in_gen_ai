

import torch
import torch.nn as nn

from glob import glob
import re
import os

def resume_checkpoint(
    models_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    certain_check_point:int=-1
) -> tuple[int, float]:
    
    checkpoint_paths = glob(os.path.join(models_dir, "*.pth"))

    # Parse epoch numbers from filenames like 'checkpoint_10.pth'
    model_epochs = []
    for path in checkpoint_paths:
        filename = os.path.basename(path)
        match = re.search(r'_(\d+)\.pth$', filename)
        if match:
            epoch_num = int(match.group(1))
            model_epochs.append((epoch_num, path))

    if model_epochs:
        # Pick checkpoint with highest epoch
        
        if certain_check_point != -1:
            last_epoch = certain_check_point
            last_checkpoint = dict(model_epochs)[certain_check_point]
        else:
            last_epoch, last_checkpoint = max(model_epochs, key=lambda x: x[0])

        print(f"Loading last checkpoint: {last_checkpoint}")

        checkpoint = torch.load(last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] 
                              if 'model_state_dict' in checkpoint else checkpoint)
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return last_epoch, last_checkpoint
    else:
        print("No checkpoint found. Starting from scratch.")
        if certain_check_point > 1:
            raise Exception("No checkpoint found. Starting from scratch.")
        return 0, None
    

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_dir: str = "models",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"checkpoint_{epoch}.pth"
    
    checkpoint_path = os.path.join(save_dir, filename)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss": loss
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    return checkpoint_path
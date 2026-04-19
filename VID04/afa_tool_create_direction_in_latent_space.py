# this file is created by gemini
import sys
import torch
import torchvision.utils as vutils
import torchvision.transforms as T
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QListWidget, QFileDialog, QFrame)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# --- Your trained model imports ---
from afa_faces_vae import VAE
from afa_save_load import resume_checkpoint

# --- Experiment settings ---
experiment = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 500
image_size = 128 

# --- Transformation ---
transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
])

# --- Load Model ---
model = VAE(input_channels=3, encoder_feature_size=128, decoder_feature_size=128, latent_dim=latent_dim).to(device)
optimizer = None
resume_checkpoint(f"./models{experiment}", model, optimizer, device, 15)
model.eval()

class LatentGui(QMainWindow):
    def __init__(self, model, device, transform):
        super().__init__()
        self.model = model
        self.device = device
        self.transform = transform
        self.setWindowTitle("VAE Latent Vector Explorer")
        self.resize(1100, 750)
        
        self.saved_vectors = {}  
        self.current_mean = None 
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- Left Panel: Controls ---
        controls = QVBoxLayout()
        
        btn_browse = QPushButton("Select Images (Pick 15)")
        btn_browse.setStyleSheet("background-color: #2c3e50; color: white; height: 40px; font-weight: bold;")
        btn_browse.clicked.connect(self.browse_images)
        controls.addWidget(btn_browse)

        controls.addWidget(QFrame()) # Spacer

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Vector Name (e.g., 'Smiling')")
        controls.addWidget(QLabel("Step 1: Save Current Mean"))
        controls.addWidget(self.name_input)
        
        btn_save = QPushButton("Save Mean Vector")
        btn_save.clicked.connect(self.save_vector)
        controls.addWidget(btn_save)

        line = QFrame(); line.setFrameShape(QFrame.HLine); controls.addWidget(line)

        self.vector_list = QListWidget()
        self.vector_list.setSelectionMode(QListWidget.MultiSelection)
        controls.addWidget(QLabel("Step 2: Select 2 to Subtract"))
        controls.addWidget(self.vector_list)

        btn_diff = QPushButton("Compute Difference (V2 - V1)")
        btn_diff.setStyleSheet("background-color: #e67e22; color: white;")
        btn_diff.clicked.connect(self.compute_difference)
        controls.addWidget(btn_diff)

        btn_export = QPushButton("Export All to .pt")
        btn_export.clicked.connect(self.export_data)
        controls.addWidget(btn_export)

        layout.addLayout(controls, 1)

        # --- Right Panel: Image Display ---
        display_layout = QVBoxLayout()
        
        self.orig_img = QLabel("Originals"); self.orig_img.setAlignment(Qt.AlignCenter)
        self.recon_img = QLabel("Reconstructions"); self.recon_img.setAlignment(Qt.AlignCenter)

        display_layout.addWidget(QLabel("<b>Original Input Batch:</b>"))
        display_layout.addWidget(self.orig_img)
        display_layout.addWidget(QLabel("<b>VAE Reconstruction:</b>"))
        display_layout.addWidget(self.recon_img)
        
        layout.addLayout(display_layout, 3)

    def browse_images(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select 15 Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_paths:
            self.process_selected_images(file_paths[:15])

    def process_selected_images(self, paths):
        try:
            tensor_list = []
            for p in paths:
                # Open image and ensure it is RGB and C-Contiguous
                with Image.open(p) as img:
                    img = img.convert("RGB")
                    # We convert to numpy and back to PIL to strip any weird 
                    # metadata or non-contiguous memory layouts
                    img_np = np.array(img)
                    img_fixed = Image.fromarray(np.ascontiguousarray(img_np))
                    
                    tensor_list.append(self.transform(img_fixed))
            
            imgs = torch.stack(tensor_list).to(self.device)
            
            with torch.no_grad():
                mu, _ = self.model.encode(imgs)
                recon = self.model.decode(mu)
                self.current_mean = mu.mean(dim=0, keepdim=True).cpu()

            # Create Visualization Grids
            orig_grid = vutils.make_grid(imgs.cpu(), nrow=5, normalize=True)
            recon_grid = vutils.make_grid(recon.cpu(), nrow=5, normalize=True)

            self.set_pixmap(self.orig_img, orig_grid)
            self.set_pixmap(self.recon_img, recon_grid)
            
        except Exception as e:
            # Print full error to console for debugging
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")

    def set_pixmap(self, label, tensor_img):
        img = tensor_img.permute(1, 2, 0).numpy()
        img = (img * 255).astype('uint8')
        h, w, c = img.shape
        # Use .copy() here too to ensure PySide handles the buffer safely
        qimg = QImage(img.copy().data, w, h, w * c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def save_vector(self):
        name = self.name_input.text().strip()
        if name and self.current_mean is not None:
            self.saved_vectors[name] = self.current_mean
            if not self.vector_list.findItems(name, Qt.MatchExactly):
                self.vector_list.addItem(name)
            self.name_input.clear()

    def compute_difference(self):
        items = self.vector_list.selectedItems()
        if len(items) == 2:
            # Note: Items are processed in order of selection usually
            name1, name2 = items[0].text(), items[1].text()
            v1, v2 = self.saved_vectors[name1], self.saved_vectors[name2]
            
            diff_name = f"diff_{name2}_minus_{name1}"
            self.saved_vectors[diff_name] = v2 - v1
            
            if not self.vector_list.findItems(diff_name, Qt.MatchExactly):
                self.vector_list.addItem(diff_name)

    def export_data(self):
        if not self.saved_vectors: return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "vectors.pt", "Torch Files (*.pt)")
        if path:
            torch.save(self.saved_vectors, path)
            print(f"Exported to {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LatentGui(model, device, transform)
    win.show()
    sys.exit(app.exec())
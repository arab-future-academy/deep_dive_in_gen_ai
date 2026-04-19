# this file is created by gemeni
import sys
import torch
import torchvision.utils as vutils
import torchvision.transforms as T
import numpy as np
from PIL import Image
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QFrame, QSlider, QListWidget, QAbstractItemView, QScrollArea)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt

# --- Your trained model imports ---
from afa_faces_vae import VAE
from afa_save_load import resume_checkpoint

# --- Settings ---
experiment = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 500
image_size = 128 

transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
])

# --- Load Model ---
model = VAE(input_channels=3, encoder_feature_size=128, decoder_feature_size=128, latent_dim=latent_dim).to(device)
resume_checkpoint(f"./models{experiment}", model, None, device, 15)
model.eval()

class LatentEditorGui(QMainWindow):
    def __init__(self, model, device, transform):
        super().__init__()
        self.model = model
        self.device = device
        self.transform = transform
        self.setWindowTitle("VAE Attribute Editor")
        self.resize(1400, 900) # Made slightly wider for side-by-side view
        
        self.loaded_vectors = {}  
        self.base_latents = None   
        self.active_sliders = {}   
        
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # --- Left Panel: File & Vector Selection ---
        left_panel = QVBoxLayout()
        
        btn_load_pt = QPushButton("1. Open Vectors (.pt)")
        btn_load_pt.clicked.connect(self.load_vector_file)
        left_panel.addWidget(btn_load_pt)

        self.vector_list = QListWidget()
        self.vector_list.setSelectionMode(QAbstractItemView.MultiSelection)
        left_panel.addWidget(QLabel("Select Vectors:"))
        left_panel.addWidget(self.vector_list)

        btn_confirm_vectors = QPushButton("2. Create Sliders")
        btn_confirm_vectors.clicked.connect(self.setup_sliders)
        left_panel.addWidget(btn_confirm_vectors)

        line = QFrame(); line.setFrameShape(QFrame.HLine); left_panel.addWidget(line)

        btn_load_imgs = QPushButton("3. Select Images to Edit")
        btn_load_imgs.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
        btn_load_imgs.clicked.connect(self.load_images)
        left_panel.addWidget(btn_load_imgs)

        layout.addLayout(left_panel, 1)

        # --- Middle Panel: Dynamic Sliders ---
        self.slider_layout = QVBoxLayout()
        slider_widget = QWidget()
        slider_widget.setLayout(self.slider_layout)
        
        scroll = QScrollArea() # Added scroll in case you have many sliders
        scroll.setWidgetResizable(True)
        scroll.setWidget(slider_widget)
        layout.addWidget(scroll, 1)

        # --- Right Panel: Image Display ---
        display_panel = QVBoxLayout()
        
        # Original Image Display
        display_panel.addWidget(QLabel("<b>Original Images:</b>"))
        self.orig_img = QLabel("No images loaded")
        self.orig_img.setAlignment(Qt.AlignCenter)
        self.orig_img.setMinimumHeight(300)
        display_panel.addWidget(self.orig_img)
        
        display_panel.addWidget(QFrame(frameShape=QFrame.HLine))

        # Reconstructed/Edited Display
        display_panel.addWidget(QLabel("<b>Edited Reconstructions:</b>"))
        self.recon_img = QLabel("Reconstructions will appear here")
        self.recon_img.setAlignment(Qt.AlignCenter)
        self.recon_img.setMinimumHeight(300)
        display_panel.addWidget(self.recon_img)
        
        layout.addLayout(display_panel, 3)

    def load_vector_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Latent Vectors", "", "Torch Files (*.pt)")
        if path:
            self.loaded_vectors = torch.load(path)
            self.vector_list.clear()
            for name in self.loaded_vectors.keys():
                self.vector_list.addItem(name)

    def setup_sliders(self):
        for i in reversed(range(self.slider_layout.count())): 
            self.slider_layout.itemAt(i).widget().setParent(None)
        self.active_sliders = {}

        for item in self.vector_list.selectedItems():
            name = item.text()
            row = QHBoxLayout()
            label = QLabel(f"{name}: 0.0")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-30, 30)
            slider.setValue(0)
            slider.valueChanged.connect(lambda v, n=name, l=label: self.on_slider_move(v, n, l))
            
            row.addWidget(label, 1)
            row.addWidget(slider, 2)
            
            container = QWidget()
            container.setLayout(row)
            self.slider_layout.addWidget(container)
            self.active_sliders[name] = slider

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.png *.jpg *.jpeg)")
        if paths:
            tensor_list = []
            # We process exactly 15 or fewer if less are selected
            selected_paths = paths[:15]
            for p in selected_paths:
                with Image.open(p) as img:
                    img_np = np.array(img.convert("RGB"))
                    img_fixed = Image.fromarray(np.ascontiguousarray(img_np))
                    tensor_list.append(self.transform(img_fixed))
            
            batch = torch.stack(tensor_list).to(self.device)
            
            # Show the Originals immediately
            orig_grid = vutils.make_grid(batch.cpu(), nrow=5, normalize=True)
            self.set_pixmap(self.orig_img, orig_grid)

            with torch.no_grad():
                mu, _ = self.model.encode(batch)
                self.base_latents = mu.cpu()
            
            self.update_reconstruction()

    def on_slider_move(self, value, name, label):
        label.setText(f"{name}: {value / 10.0:.1f}")
        self.update_reconstruction()

    def update_reconstruction(self):
        if self.base_latents is None: return

        offset = torch.zeros_like(self.base_latents)
        for name, slider in self.active_sliders.items():
            offset += (slider.value() / 10.0) * self.loaded_vectors[name].cpu()
        if len(self.active_sliders) > 0:
            offset /= len(self.active_sliders)
        with torch.no_grad():
            recon = self.model.decode((self.base_latents + offset).to(self.device))
        
        grid = vutils.make_grid(recon.cpu(), nrow=5, normalize=True)
        self.set_pixmap(self.recon_img, grid)

    def set_pixmap(self, label, tensor_img):
        img = (tensor_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
        h, w, c = img.shape
        qimg = QImage(img.copy().data, w, h, w * c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = LatentEditorGui(model, device, transform)
    win.show()
    sys.exit(app.exec())
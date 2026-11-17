# MRI Project
### Created by Eldon Whiteley, Gail Kinder, and Dylan Bjorum

An MRI segmentation project designed to process, visualize, and train models on 3D volumetric MRI scans (`.im`) and their corresponding multi-channel ground truth segmentation masks (`.seg`).

---

## Environment Setup

You can set up your local development environment using **Conda** (recommended) or **pip / venv**.

### Option 1: Conda Setup (Recommended)

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/Eldonwhiteley/MRI_Project.git
   cd MRI_Project
   ```

2. **Create the Conda environment from `environment.yml`:**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment:**
   ```bash
   conda activate mri_env
   ```

4. **Register the environment kernel with Jupyter:**
   ```bash
   python -m ipykernel install --user --name mri_env --display-name "Python (mri_env)"
   ```
   *When opening notebooks in VS Code or JupyterLab, select the kernel named **`Python (mri_env)`**.*

---

### Option 2: Python `venv` + `pip` Setup

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Register the kernel with Jupyter:**
   ```bash
   python -m ipykernel install --user --name mri_venv --display-name "Python (.venv)"
   ```

---

## Dataset Overview

The dataset consists of 3D knee MRI scans from the Osteoarthritis Initiative (OAI) stored as HDF5 containers:

* **`.im` files:** Original 3D MRI volumes with shape `(384, 384, 160)` of `float32` intensities.
* **`.seg` files:** 3D multi-channel segmentation masks with shape `(384, 384, 160, 6)` of `uint8` binary values (`0` or `1`).
* **Dataset Key:** In both files, array data is stored under the HDF5 key `"/data"`.

### Segmentation Channels:
* **Channel 0:** Femoral Cartilage
* **Channel 1:** Lateral Tibial Cartilage
* **Channel 2:** Medial Tibial Cartilage
* **Channel 3:** Patellar Cartilage
* **Channel 4:** Lateral Meniscus
* **Channel 5:** Medial Meniscus

---

## Quick Start & Visualization

### 1. Interactive Jupyter Notebook
Open [`src/HDF5_Viewing.ipynb`](src/HDF5_Viewing.ipynb) in JupyterLab, VS Code, or Google Colab:
* Includes an interactive slice slider (`ipywidgets`) to scrub through all 160 slices in real time.
* Displays composite multi-color overlays and individual tissue breakdowns.
* Includes orthogonal planar reslicing (Sagittal, Coronal, Axial).

### 2. Using `HDF5View` in Python Scripts
You can use the [`HDF5View`](src/hdf5_view.py) class in your own code:

```python
from src.hdf5_view import HDF5View

# 1. Automatically find matching (.im, .seg) file pairs
pairs = HDF5View.find_pairs("data/train-shifted/train-shifted")
sample_im, sample_seg = pairs[0]

# 2. Initialize viewer
viewer = HDF5View(sample_im, sample_seg)

# 3. Static multi-panel plot of slice 95
viewer.plot_slice(slice_idx=95, alpha=0.55)

# 4. Interactive slice scrubber (in notebook)
viewer.interactive_viewer(default_slice=75)

# 5. Orthogonal reslices (Sagittal, Coronal, Axial)
viewer.plot_orthogonal()
```

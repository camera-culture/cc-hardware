# CC-Hardware SPAD + Gantry Capture & Processing

This repository contains tools for driving a SPAD sensor mounted on a stepper-motor gantry, capturing top-level data, and then processing/training models on that data.

---

## 1. Get Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/camera-culture/cc-hardware.git
   ```

2. **Install in editable mode**  
   ```bash
   cd cc_hardware
   pip install -e .
   ```

3. **Switch to the `cc_urops` branch**  
   ```bash
   git checkout cc_urops
   ```

---

## 2. Run a Capture

> **Before you begin:**  
>  • Make sure your SPAD + gantry are physically connected and powered.  
>  • Press the gantry’s button until the blue LED is on.  

1. **Edit the script constants**  
   At the top of `run_spad_capture.py`, set:
   ```python
   X_SAMPLES      = 2          # number of X steps
   Y_SAMPLES      = 2          # number of Y steps
   OBJECT_NAME    = "arrow"    # object identifier
   SPAD_POSITION  = [0.1, 0.4, 0.5]  # in inches, [x, y, z]
   ```
   
2. **Run the capture**  
   ```bash
   python run_spad_capture.py
   ```
   - The capture CLI will launch automatically with all the right Hydra overrides.
   - **At the end**, you’ll see a **big, bold green** message with the full path to your saved `<object>_data.pkl`.
   - **Copy** that `.pkl` path—you’ll need it in the next step.

---

## 3. Process & Train a Model

1. **Open the notebook**  
Use the below or Colab/any Jupyter processing environment. 
   ```bash
   jupyter lab run_capture_process.ipynb
   ```

2. **Paste in your `.pkl` path**  
   At the top of the notebook set:
   ```python
   PKL_PATH = "/full/path/to/arrow_data.pkl"
   ```

3. **Run all cells**  
   Follow the instructions in each section. The notebook will load your captured data, preprocess it, and walk you through training a model.

---

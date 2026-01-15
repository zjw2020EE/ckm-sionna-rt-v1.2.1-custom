# Sionna-CKM-RT: Advanced Channel Knowledge Mapping Extension

**Sionna-CKM-RT** is a customized extension of [Sionna RT v1.2.1](https://nvlabs.github.io/sionna-rt), the high-performance ray tracing package for [Sionna™](https://github.com/NVlabs/sionna). 

This version introduces the `CKMapSolver`, which extends the standard ray tracing payload to support 12+ advanced physical metrics, including high-precision Time of Arrival (ToA) and directional characteristics (DoD/DoA).

---

## 🚀 Key Extensions

While the original `RadioMapSolver` focuses on energy coverage (Path Gain), our **`CKMapSolver`** enables comprehensive spatial-temporal analysis:

- **High-Precision Temporal Metrics**: Time of Arrival (ToA) with nanosecond ($ns$) precision and RMS Delay Spread.
- **Full Directional Mapping**: Mean Departure (DoD) and Arrival (DoA) vectors, including Azimuth, Elevation, and Angular Spread.
- **Environment Sensing**: Native Line-of-Sight (LoS) visibility identification.
- **CUDA Optimized**: Implements stable `atomicMin` operations via bit-casting for hardware-level precision in symbolic Dr.Jit loops.

---

## 📊 Mathematical Definitions

The `CKMapSolver` aggregates all propagation paths $i \in \{1, \dots, N\}$ in a spatial cell. Each path carries power $P_i$, delay $\tau_i$, arrival direction $\mathbf{v}\_{a,i}$, and departure direction $\mathbf{v}\_{d,i}$.

### 1. Temporal Metrics

* **Path Gain ($G$):** $$G = \sum_{i=1}^{N} P_i$$

* **Time of Arrival ($\tau_{\text{ToA}}$):** The minimum flight time, reflecting the first-arrival component.  
  $$\tau_{\text{ToA}} = \min_{i} (\tau_i)$$

* **RMS Delay Spread ($\sigma_{\tau}$):** $$\sigma_{\tau} = \sqrt{\frac{\sum_{i=1}^{N} P_i \tau_i^2}{\sum_{i=1}^{N} P_i} - \left( \frac{\sum_{i=1}^{N} P_i \tau_i}{\sum_{i=1}^{N} P_i} \right)^2}$$

---

### 2. Directional Metrics (DoA & DoD)

We derive the **Power-Weighted Mean Direction Vector** $\bar{\mathbf{v}} = [\bar{x}, \bar{y}, \bar{z}]^T$ for both arrival and departure: 

$$\bar{\mathbf{v}} = \frac{\sum_{i=1}^{N} P_i \mathbf{v}_i}{\sum_{i=1}^{N} P_i}$$

* **Azimuth ($\phi$):** $$\phi = \text{atan2}(\bar{y}, \bar{x})$$

* **Elevation ($\theta$):** $$\theta = \text{atan2}(\bar{z}, \sqrt{\bar{x}^2 + \bar{y}^2})$$

* **Angular Spread ($\sigma_{\text{AS}}$):** Measures angular dispersion (used for DSA and DSD).  
  $$\sigma_{\text{AS}} = \sqrt{1 - \|\bar{\mathbf{v}}\|^2}$$
---

## 📋 Supported Metrics Catalog

The `CKMapSolver` provides a total of 12+ metrics. Below is the detailed specification for each parameter.

### 1. Standard & Temporal Metrics
| Metric Name | Description | Unit | Range / Value |
| :--- | :--- | :--- | :--- |
| **`path_gain`** | Total accumulated channel power | Linear | $[0, \infty)$ |
| **`toa`** | Time of Arrival (First component) | Nanoseconds ($ns$) | $[0, \infty)$, $-1$ for No Signal |
| **`rms_ds`** | RMS Delay Spread | Nanoseconds ($ns$) | $[0, \infty)$ |
| **`is_los`** | Line-of-Sight Visibility | Boolean | $1$ (LoS), $0$ (NLoS) |
| **`rss`** | Received Signal Strength | $dBm$ | $(-\infty, 0]$ |
| **`sinr`** | Signal-to-Interference-plus-Noise Ratio| $dB$ | $(-\infty, \infty)$ |

### 2. Directional Metrics (Arrival - DoA)
| Metric Name | Description | Unit | Range / Value |
| :--- | :--- | :--- | :--- |
| **`doa_azi`** | Mean Arrival Azimuth | Radians ($rad$) | $[-\pi, \pi]$ |
| **`doa_ele`** | Mean Arrival Elevation | Radians ($rad$) | $[-\pi/2, \pi/2]$ |
| **`dsa`** | Directional Spread of Arrival | Index | $[0, 1]$ ($0$: Directional, $1$: Diffuse) |

### 3. Directional Metrics (Departure - DoD)
| Metric Name | Description | Unit | Range / Value |
| :--- | :--- | :--- | :--- |
| **`dod_azi`** | Mean Departure Azimuth | Radians ($rad$) | $[-\pi, \pi]$ |
| **`dod_ele`** | Mean Departure Elevation | Radians ($rad$) | $[-\pi/2, \pi/2]$ |
| **`dsd`** | Directional Spread of Departure | Index | $[0, 1]$ ($0$: Directional, $1$: Diffuse) |

---
**Note on Units**: Angular metrics are provided in Radians by default to maintain compatibility with standard trigonometric functions in Python/NumPy. Temporal metrics are scaled to nanoseconds ($ns$) for high-resolution analysis.

---

## 🛠 Installation

1. Clone this repository to your local machine.
2. Navigate to the project root directory.
3. Install in **editable mode** to ensure the path is correctly mapped. It is recommended to install this package in **editable mode** to allow for custom modifications:


```bash
git clone [https://github.com/YourUsername/ckm-sionna-rt-v1.2.1-custom.git](https://github.com/YourUsername/ckm-sionna-rt-v1.2.1-custom.git)
cd ckm-sionna-rt-v1.2.1-custom
pip install -e .
```
### ⚠️ Troubleshooting
> [!IMPORTANT]
> **If the `sionna.rt` import fails in a Jupyter Notebook:**
> 1. Ensure you have executed `python -m pip install -e .` in the correct environment.
> 2. **Restart the Jupyter Kernel** (Press `0,0` or use the 'Kernel' menu) to refresh the paths.

If the `sionna.rt` import fails, please run the following command in your terminal from the project root:

```bash
python -m pip install -e .
```

To run on CPU, [LLVM](https://llvm.org) is required. For GPU acceleration, ensure a compatible NVIDIA driver is installed.

---

## 📖 Getting Started (Tutorial)

We provide a comprehensive tutorial in the root directory: **`tutorial_ckmap.ipynb`**. 

This tutorial demonstrates how to:
* **Load a high-fidelity urban scene** and configure antenna arrays.
* **Launch a high-precision simulation** with $10^8$ samples to achieve "gold standard" accuracy.
* **Post-process raw tensors** into physical metrics (ToA, Angles, Spreads) using `utils.py`.
* **Visualize advanced metrics** like `toa_ns`, `dod_azi`, and `dsd` via customized heatmaps.
* **Interactive 3D inspection**: Use `scene.preview(ck_map=rm, rm_metric="rms_ds")` to visualize the radio map overlaid on the 3D environment.



---

## 📂 Project Structure

* **`src/sionna_rt/radio_map_solvers/`**: Contains the core implementation of the new solvers (`CKMapSolver`, `PlanarCKMap`, and `MeshCKMap`).
* **`utils.py`**: A dedicated utility script with English comments for metric extraction, unit conversion (e.g., seconds to nanoseconds), and angular masking.
* **`tutorial_ckmap.ipynb`**: A step-by-step Jupyter Notebook guide for end-to-end CKM generation and visualization.

---

## License and Citation

Sionna RT is Apache-2.0 licensed. If you use this customized version for research, please cite the original Sionna software and this extension.

```bibtex
@software{sionna,
 title = {Sionna},
 author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
 note = {[https://nvlabs.github.io/sionna/](https://nvlabs.github.io/sionna/)},
 year = {2022},
 version = {1.2.1}
}
```

---


# PaCo demo

This repository provides the inference demo of PaCo.

## Getting Started

You can run the demo in one of two ways:

- **Google Colab** (Recommended): No installation needed; get started instantly in the cloud.
- **Local Linux Environment**: Set up the environment locally for GPU-accelerated inference.

### Google Colab

[<img src="https://colab.research.google.com/assets/colab-badge.svg" height="32"/>](https://colab.research.google.com/github/PaCo-X/PaCo-demo/blob/main/demo.ipynb)

To run the demo on Google Colab, simply click the badge above ([link](https://colab.research.google.com/github/PaCo-X/PaCo-demo/blob/main/demo.ipynb)). Follow the instructions and execute the cells sequentially. Please note that setting up the environment and installing dependencies will take about 1-2 minutes.

---

### Local Linux Environment

> [!NOTE]
A GPU is required to run the inference process locally.

1. Clone the repository:

   ```bash
   git clone https://github.com/PaCo-X/PaCo-demo.git && cd PaCo-demo && git lfs pull
   ```

2. Create a conda environment with all dependencies (this will take approximately 5 minutes):

   ```bash
   conda env create -f build/environment.yml && conda activate paco
   chmod +x PolyFit/polyfit
   ```

3. Run the inference code and inspect the results in the `data/obj` directory:

   ```bash
   python inference.py --pc_file data/pc/00205251_3.npy
   ```

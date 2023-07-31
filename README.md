# Progressive Multi-Scale Light Field Networks

Codebase for _Progressive Multi-Scale Light Field Networks_ (3DV 2022).

## Getting Started

1. Download [our datasets](https://drive.google.com/drive/folders/16rtVRySPl5mujoEaowMoU2b71btljUPz?usp=sharing) and extract them to `datasets` directory.
2. Setup a PyTorch environment and install `requirements.txt`.
3. To train, run `python app.py -c configs/multiscale_jon.txt`. \
   Alternatively, download our [trained LFNs](https://drive.google.com/drive/folders/16rtVRySPl5mujoEaowMoU2b71btljUPz?usp=sharing) to `runs`.


## Interactive Viewer

To use the viewer on Ubuntu, run the following:
```bash
sudo apt install libmesa-dev libglfw3
# Required to install pycuda with OpenGL support
echo "CUDA_ENABLE_GL = True" > ~/.aksetup-defaults.py
pip install pycuda glumpy pyopengl
rm ~/.aksetup-defaults.py

python app.py -c configs/multiscale_jon.txt --script-mode=viewer
```

If you get `CUBLAS_STATUS_EXECUTION_FAILED` while opening the viewer, try running with `CUBLAS_WORKSPACE_CONFIG=:0:0`. ([PyTorch Issue](https://github.com/pytorch/pytorch/issues/54975)).


## Citation

```bibtex
@inproceedings{Li2022Progressive,
  author={Li, David and Varshney, Amitabh},
  booktitle={2022 International Conference on 3D Vision (3DV)}, 
  title={Progressive Multi-Scale Light Field Networks}, 
  year={2022},
  volume={},
  number={},
  pages={231-241},
  doi={10.1109/3DV57658.2022.00035}}
}
```

## Acknowledgments

- `utils/nerf_utils.py` is borrowed from `krrish94/nerf-pytorch`.

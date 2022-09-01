# Progressive Multi-scale Light Field Networks

Codebase for _Progressive Multi-scale Light Field Networks_ (3DV 2022).

## Getting Started

1. Download [our datasets](https://drive.google.com/drive/folders/16rtVRySPl5mujoEaowMoU2b71btljUPz?usp=sharing) and extract them to `datasets` directory.
2. Setup a PyTorch environment and install `requirements.txt`.
3. To train, run `python app.py -c configs/multiscale_jon.txt`. \
   Alternatively, download our [trained LFNs](https://drive.google.com/drive/folders/16rtVRySPl5mujoEaowMoU2b71btljUPz?usp=sharing) to `runs`.
4. To open the interactive viewer, run `python app.py -c configs/multiscale_jon.txt --script-mode=viewer`.

## Troubleshooting
If you get `CUBLAS_STATUS_EXECUTION_FAILED` while opening the viewer, try running with `CUBLAS_WORKSPACE_CONFIG=:0:0`. ([PyTorch Issue](https://github.com/pytorch/pytorch/issues/54975)).

For the viewer you will need the following:
* Ubuntu (WSL2 is not supported) with `sudo apt install libmesa-dev libglfw3`
* [`pycuda`](https://documen.tician.de/pycuda/) compiled with OpenGL
* `pip install glumpy pyopengl`


## Citation

```bibtex
@inproceedings{Li2022Progressive,
  author={Li, David and Varshney, Amitabh},
  booktitle={2022 International Conference on 3D Vision (3DV)},
  title={Progressive Multi-scale Light Field Networks},
  year={2022},
  volume={},
  number={},
  pages={},
  doi={}
}
```

## Acknowledgments

- `utils/nerf_utils.py` is borrowed from `krrish94/nerf-pytorch`.

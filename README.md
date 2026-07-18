# GPU Programming

Learning GPU programming and CUDA by writing kernels & pretending I know what I'm doing

## Contents

- `kernels/` — CUDA kernels (primary learning material)
- `notes.md` — personal notes on C++/CUDA concepts
- `running_cuda_in_kaggle_or_colab.ipynb` — poor man's way to runing CUDA in either Google Colab or Kaggle

## Running

On a local CUDA machine, compiled binaries go into the ignored `build/` directory:

```sh
make run SRC=pmpp/add.cu
```

After installing Modal and running `modal setup`, `make modal` defaults to `pmpp/add.cu` on an
RTX PRO 6000:

```sh
make modal
make modal SRC=pmpp/matvecmul.cu
make modal SRC=pmpp/matvecmul.cu GPU=L40S
```

Check active containers and today's cost with:

```sh
modal container list
modal environment billing report --for today --show-resources
```

## Resources

- [GPU MODE resource-stream](https://github.com/gpu-mode/resource-stream) <3
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) <3
- NVIDIA blog posts (e.g., [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/))

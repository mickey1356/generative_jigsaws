# Generative Whimsy Jigsaws

Generate jigsaw puzzles whose overall shape and individual pieces each look like a text prompt. Stage 1 optimizes the puzzle. Stage 2 turns it into printable geometry.

## Requirements

Linux with an NVIDIA GPU. CUDA toolkit with nvcc on PATH (tested: CUDA 12.4). CMake and Ninja. Python 3.11-3.13 (tested: 3.13).

## Install

1. Make an environment.

   ```
   conda create -n puzzle python=3.13 -y
   conda activate puzzle
   ```

2. Install PyTorch for your CUDA first.

   ```
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
   ```

3. Install the Python dependencies.

   ```
   pip install -r requirements.txt
   ```

4. Build the cufsm CUDA extension.

   ```
   pip install ./cufsm_lib
   ```

   The build assumes GPU arch 7.5. For other GPUs, override it (e.g. RTX 30 = 86, RTX 40 = 89).

   ```
   pip install ./cufsm_lib --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=86
   ```

5. Install pydiffvg (stage 2 only). It is not on PyPI. Build from source.

   ```
   git clone --recursive https://github.com/BachiLi/diffvg
   cd diffvg && python setup.py install && cd ..
   ```

6. Get diffusion model access. Stage 1 uses DeepFloyd IF, which is gated. Accept the license at https://huggingface.co/DeepFloyd/IF-I-L-v1.0

   ```
   huggingface-cli login
   ```

Usage

Stage 1 optimizes a puzzle from prompts. Defaults are in configs/base.toml. Flags override them.

```
python stage1.py --prompts "cat" --pieces 12 --name cats_12
python stage1.py --overall_prompt "tree" --prompts "bird" --name tree_of_birds
```

Output goes to experiments/<name>/. Run `python stage1.py -h` for all flags.

Stage 2 post-processes a stage 1 folder into geometry.

```
python stage2.py -f experiments/cats_12
python stage2.py -m experiments -o experiments/_final
```

Output includes vector polygons, SVGs, and OBJ meshes.

Notes

Stage 1 needs only cufsm and the diffusion stack. pydiffvg, nvdiffrast, gmsh, libigl, and triangle are stage 2 only.

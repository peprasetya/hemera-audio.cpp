# AMD XDNA2 NPU (Ryzen AI) Matmul Path

Runs the large matrix multiplies of the CPU backend on the NPU of Strix Point, Strix Halo and Krackan
Point chips, through XRT on Linux. Select it with `--backend npu`. Everything else stays on the CPU, so
the GPU is not used at all, and a language model can generate on the GPU at full speed at the same time.

It is not a separate ggml backend: there is no graph splitting and no scheduler. The CPU graph loop gives
an eligible `MUL_MAT` to the NPU on thread 0 and carries on (`external/ggml/src/ggml-cpu/xdna/`). Any model
benefits whose weights are 8-bit or float and whose multiplies are large enough; nothing in a model has to
change.

## Measured

Ryzen AI Max+ 395 (Radeon 8060S, `RyzenAI-npu5`, firmware 1.1.2.65), Ubuntu 26.04, kernel 7.0, XRT 2.21.75,
VoxCPM2 `q8_0`, streaming 14 s of Indonesian speech with a described voice:

| | Speed | First audio | Dropouts | LLM on the GPU | CPU |
|---|---|---|---|---|---|
| `--backend vulkan`, alone | 2.4x realtime | 0.15 s | 0 s | | |
| `--backend vulkan`, next to Qwen3.8 27B | 0.58x | | 7.1 s | 20 tokens/s | |
| `--backend npu`, alone | 1.76x | 0.28 s | 0 s | | 3.5 cores |
| `--backend npu`, next to Qwen3.8 27B | 1.0-1.25x | 0.3-0.6 s | about 0 s | 20-21 tokens/s | 3.3 cores |
| `--backend cpu`, 16 threads, offline | 0.8x | | | | 10.5 cores |

OmniVoice `q8_0`, offline, 8 threads: RTF 2.62 on the CPU, 2.01 with the NPU.

A soak next to Qwen3.8: 34 requests, 406 s of speech in 365 s, 0.17 s of dropouts in all, no errors, the LLM at
20.1 tokens/s. Whisper large-v3 reading back seven hard sentences: 4.2% word errors on the NPU, 3.3% on Vulkan, with
the errors on the same words (numbers, Latin).

## Requirements

- Linux with the `amdxdna` kernel driver (in the kernel since 6.14). `ls /dev/accel` shows `accel0`. Do not
  install the DKMS driver over a working in-tree one.
- XRT with its NPU plugin. On Ubuntu 26.04: `apt install libxrt-dev libxrt-npu2 libxrt-utils-npu`. Without
  installing, unpack those packages and `libxrt2`, `libxrt-utils`, `libboost-filesystem1.90.0` and
  `libboost-program-options1.90.0` into a folder and give that folder's `usr` as `ENGINE_XDNA_XRT_ROOT`; the
  program sets `XILINX_XRT` to it by itself. `xrt-smi examine` should list the NPU.
- The core program, `gemm_bfp.xclbin`. Build it once, below. The audio.cpp build itself needs no NPU tools.

## Build

```bash
cmake -S . -B build/npu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENGINE_ENABLE_XDNA=ON -DENGINE_XDNA_XRT_ROOT=/usr \
  -DAUDIOCPP_MODEL_SET=custom -DAUDIOCPP_MODELS="voxcpm2,omnivoice"
cmake --build build/npu --target audiocpp_server audiocpp_cli
```

The core program is built with AMD's open IRON toolchain, in a Python 3.12 environment:

```bash
python3.12 -m venv ironenv && . ironenv/bin/activate
pip install mlir_aie -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4
git clone --depth 1 --branch v1.4.4 https://github.com/Xilinx/mlir-aie
pip install -r mlir-aie/utils/peano-requirements.txt
export MLIR_AIE_INSTALL_DIR=$VIRTUAL_ENV/lib/python3.12/site-packages/mlir_aie
export PEANO_INSTALL_DIR=$VIRTUAL_ENV/lib/python3.12/site-packages/llvm-aie
export PATH=$MLIR_AIE_INSTALL_DIR/bin:$PATH PYTHONPATH=$MLIR_AIE_INSTALL_DIR/python:$PYTHONPATH
tools/xdna/build.sh build/npu/bin/xdna
```

That leaves `xdna/gemm_bfp.xclbin` next to the programs, where they look for it, and checks the instruction
generator in `ggml-xdna-insts.h` against the compiler's output for eight shapes, word for word. Compile it
again only if `tools/xdna/gemm_bfp.py` or `mm_bfp_f32.cc` change; if the placement of the streams then moves,
that check fails and `A_SHIM`, `B_SHIM` and the parameter addresses have to follow.

## Use

```bash
audiocpp_cli --task tts --family voxcpm2 --model voxcpm2-q8_0.gguf --backend npu --threads 2 --text "Hello." --out hello.wav
```

For a server, in the model's entry:

```json
"session_options": {
  "voxcpm2.perf_mode": "cuda_perf",
  "voxcpm2.exec_plan": "{\"audiovae\":{\"impl\":\"ggml\",\"backend\":\"cpu\",\"params\":{\"threads\":4}}}"
},
"default_request_options": { "stream_chunk_policy": "fixed", "stream_frames_per_event": 1 }
```

- **`--threads 2`.** With the NPU doing the multiplies, more generator threads only wait; 2 is faster than 4.
- **`voxcpm2.perf_mode`** packs each layer's gate and up (and query, key, value) into one multiply, so one
  dispatch instead of two or three. Despite its name it is not specific to CUDA.
- **`voxcpm2.exec_plan`** gives the AudioVAE decoder its own CPU threads and its own worker, so the generator
  never waits for it.
- **`stream_frames_per_event: 1`, policy `fixed`** sends audio every 0.32 s of speech. The default doubling
  chunks arrive too far apart for the first seconds and the listener hears a gap.
- **`num_inference_steps`** (default 10) is the most direct way to trade quality for speed: 5 reaches 2.6x
  realtime, with occasional small word errors in our test.

Settings, all optional: `GGML_XDNA_XCLBIN` (the core program), `GGML_XDNA_MIN_M` (smallest batch sent),
`GGML_XDNA_FORCE=1` (send every eligible multiply whatever it costs), `GGML_XDNA_TRACE=1` (time spent, at exit),
`GGML_CPU_SERIAL_ELEMENTS` (see below), `GGML_OP_PROFILE=1` (CPU time per op and matmul shape, at exit).

## How it works

`C[M x N] = A[M x K] * W[N x K]^T`, with `W` a constant weight matrix. All 32 cores (`tools/xdna/gemm_bfp.py`)
run one program that serves every shape with `M` a multiple of 64, `K` of 64 and `N` of 512; other shapes are
padded with zeros. Each core multiplies 16 x 64 x 64 tiles with the block-float MAC, 512 multiplies per
instruction, and keeps the running sums in float32 (`mm_bfp_f32.cc`, from mlir-aie's `mm_bfp.cc`). The operands
are `bfp16ebs8`: eight values sharing an exponent byte, each with a signed 8-bit mantissa.

The arithmetic is the cheap part. A call costs about 70 us in the driver and firmware, whatever its size, plus
moving all of `W` from memory, at up to 47 GB/s, because `W` does not fit in the array's 6 MB. So for
`22 x 1024 x 4096`: 0.18 ms, where 16 CPU threads take 0.20 ms, but without occupying the CPU. A cost estimate
in `ggml_xdna_owns` sends a call only when the NPU should be faster than the team's threads: a single row
(`M = 1`) stays on the CPU.

Weights are packed into the NPU's format on first use, about 0.5 s for VoxCPM2, and kept in their own buffers.
The instruction stream for each shape is generated on the host, also once.

**Accuracy.** A multiply is off by about 0.9% (RMS, on random data) from rounding to 8-bit mantissas. The speech
decoder runs on the CPU in full precision, so this does not reach the audio as noise; it steers the flow
sampling slightly differently, as another seed would. Test: `tools/xdna/test_mul_mat.cpp`, every shape against the
exact product.

**Failure.** If the NPU cannot be opened the program says so once and stays on the CPU. If a run fails, it says so,
finishes that multiply on the CPU and stays there. Two processes can share the NPU; each switch between them costs
about 2.4 ms.

## CPU changes this came with

These also speed up plain `--backend cpu` and give bit-identical output:

- VoxCPM2 attention multiplies through `ggml_mul_mat`'s batch broadcast instead of copying the shared key and
  value heads with a chain of concats (95,000 fewer nodes per sentence).
- `conv_transpose_1d` keeps its transposed kernel for weights instead of redoing it on every call.
- `repeat` of one value per channel fills rows on all threads; causal padding uses `fill` instead of multiplying a
  repeated slice by zero.
- Small nodes (up to `GGML_CPU_SERIAL_ELEMENTS`, default 8192 elements) run on thread 0 with no barrier per node.

## Limits

- `W` crosses memory on every call, so models with many large multiplies per second of audio remain bounded by
  memory speed. VoxCPM2 needs about 13 GB/s for its DiT weights alone at realtime. Next to a language model on the
  GPU a call takes 266 us instead of 197 us and host copies twice as long, so VoxCPM2 drops from 1.75x to 1.0-1.25x:
  realtime, with little margin. A client that buffers 0.1-0.2 s rides out what is left.
- The decoder and everything else still need a few CPU cores.
- Only tested on Strix Halo. Other XDNA2 chips have the same array and should work.
- First-generation XDNA (Phoenix, Hawk Point) is a different array and is not supported by this core program.

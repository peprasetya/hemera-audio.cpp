# AMD XDNA2 NPU (Ryzen AI) Matmul Path

Runs the large matrix multiplies of the CPU backend on the NPU of Strix Point, Strix Halo and Krackan
Point chips, through XRT on Linux. Select it with `--backend npu`. Everything else stays on the CPU, so
the GPU is not used at all, and a language model on the GPU keeps its full speed while speech is made.

It is not a separate ggml backend: there is no graph splitting and no scheduler. The CPU graph loop gives
an eligible `MUL_MAT` to the NPU on thread 0 and carries on (`external/ggml/src/ggml-cpu/xdna/`). Any model
benefits whose weights are 8-bit or float and whose multiplies are large enough; nothing in a model has to
change.

**VoxCPM2 at its full 10 flow steps is faster than realtime with it, also while a language model generates on the GPU
next to it.** See the table.

## Measured

Ryzen AI Max+ 395 (Radeon 8060S, `RyzenAI-npu5`, firmware 1.1.2.65), Ubuntu 26.04, kernel 7.0, XRT 2.21.75,
VoxCPM2 `q8_0`, streaming 7 s of Indonesian speech, 10 flow steps, with the settings below. "Dropouts" is the silence a
listener with no buffer hears.

| | Alone | Next to Qwen3.8 27B on the GPU | The LLM meanwhile | CPU |
|---|---|---|---|---|
| `vulkan`, LLM frozen for each request | 2.4x, 0 s, first audio 0.11 s | 2.4x, 0 s, first audio 0.15-0.20 s | frozen | |
| `vulkan`, LLM left running | 2.4x | 0.5-0.6x, 7-10 s in 10 s | 20 tokens/s | |
| `npu` | 1.77x, 0 s, first audio 0.21-0.24 s | 1.02-1.25x, 0 s, first audio 0.3-0.5 s | 20-22 tokens/s | 4.5 cores |
| `cpu`, 16 threads, offline | 0.8x | | | 10.5 cores |

Speed next to an LLM is at its lowest in the first half minute of the LLM's work and rises afterwards (1.02-1.06x, then about 1.2x after a minute or two, while the LLM holds its speed).

Word errors, Whisper large-v3 reading back seven hard sentences (with numbers and Latin): Vulkan 3.3%, NPU 2.1%. Seven sentences
are few: differences under about 5% mean little.

OmniVoice `q8_0`, offline, 8 threads: RTF 2.62 on the CPU, 2.01 with the NPU.

Stability: 240 requests in a row next to Qwen3.8, 25 minutes of speech, no dropouts, no failure, longest gap between pieces of audio 0.18 s; 100 requests alone, the same; 71 s of speech in one request next to Qwen3.8, 1.02-1.07x throughout.

**Measuring.** Streamed PCM from `audiocpp_server` comes at the model's own rate, 48 kHz for VoxCPM2, not OpenAI's 24
kHz. Read as 24 kHz, every speed looks twice as good, which is the mistake the first version of this document made.
Check a speed against the length of a whole WAV of the same text.

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
  "voxcpm2.lm_threads": 8,
  "voxcpm2.exec_plan": "{\"audiovae\":{\"impl\":\"ggml\",\"backend\":\"cpu\",\"params\":{\"threads\":2}}}"
},
"default_request_options": { "stream_chunk_policy": "fixed", "stream_frames_per_event": 1 }
```

- **`--threads 2`.** These threads prepare and collect the NPU's work and do the small operations between its
  multiplies. 4 give nothing more; 1 loses about 15%.
- **`voxcpm2.lm_threads`.** The language model works on one row at a time (`M = 1`), and that stays on the CPU: the
  CPU's 8-bit weights are smaller than the NPU's, and on so few rows the NPU has no advantage. Its speed is bounded
  by memory, so 12 threads give no more than 8.
- **`voxcpm2.perf_mode`** packs each layer's gate and up (and query, key, value) into one multiply, so one
  doorbell instead of two or three. Despite its name it is not specific to CUDA.
- **`voxcpm2.exec_plan`** gives the AudioVAE decoder its own CPU threads and its own worker, so the generator
  never waits for it. With `stream_decoder_state` it has a quarter of the work, and 2 threads are as good as 4.
- **`stream_frames_per_event: 1`, policy `fixed`** sends audio every 0.16 s of speech. The default doubling chunks
  arrive too far apart for the first seconds and the listener hears a gap.
- **`stream_decoder_state`** (request option, default true) keeps the decoder's convolution history from patch to patch.
  The samples are then those of a whole-sentence decode, to within one step of 16 bits. False decodes a window of
  `stream_left_context` earlier patches for every patch, as upstream does: four times the work at the default of 3,
  which next to an LLM is more than the decoder can do in time, and with a small window you can hear the seams (SNR
  against the whole-sentence decode: 74 dB with 3 patches of context, 32 dB with 1).
- **`num_inference_steps`** (default 10): fewer buy speed, and below 4 the words suffer (see the table above).

Settings, all optional: `GGML_XDNA_XCLBIN` (the core program), `GGML_XDNA_MIN_M` (smallest batch sent),
`GGML_XDNA_FORCE=1` (send every eligible multiply whatever it costs), `GGML_XDNA_TRACE=1` (time spent, at exit),
`GGML_XDNA_TEST_STALL=<n>` (skip the n-th doorbell, to see a failure and the way back), `GGML_CPU_SERIAL_ELEMENTS` (see
below), `GGML_OP_PROFILE=1` (CPU time per op and matmul shape, at exit).

## How it works

`C[M x N] = A[M x K] * W[N x K]^T`, with `W` a constant weight matrix. All 32 cores (`tools/xdna/gemm_bfp.py`)
run one program that serves every shape with `M` a multiple of 32, `K` of 64 and `N` of 512; other shapes are
padded with zeros. Each core multiplies 8 x 64 x 64 tiles with the block-float MAC and keeps the running sums in
float32 (`mm_bfp_f32.cc`, from mlir-aie's `mm_bfp.cc`). The operands are `bfp16ebs8`: eight values sharing an exponent
byte, each with a signed 8-bit mantissa.

The arithmetic is the cheap part. What costs time is the transport, in three parts, and each has its answer.

**Starting a multiply.** A job costs about 75 us in the driver and firmware, whatever it does, and VoxCPM2 makes 2,500
multiplies a second of speech. So a graph's multiplies run as one job, whose instruction stream knows them all in advance
(`ggml-xdna-insts.h`, generated on the host) and waits for a doorbell before each: a word in shared memory, which one
more DMA stream in the core program keeps copying to where the stream can poll it. The host sees that a multiply has
finished when a sentinel at the end of its output is overwritten. That costs about 20 us and no system call. A doorbell's
value carries the job's turn, odd or even, so that nothing left over from the job before can be taken for this job's. Each
job leaves the doorbell's channels idle: a task started on top of one still finishing has been seen, next to a busy GPU, never
to deliver. And nothing may hold up the reads of the doorbell's shim while a job waits for it, so that column fetches ahead
only what its memory tile holds.

**Moving the weights.** All of `W` crosses from memory for each call, because it does not fit in the array, and the NPU
reads at most 57 GB/s in all, however many streams it uses (7 GB/s each). Next to a GPU that wants all of memory, about half
of that. So each multiply's weights start crossing as soon as the one before has finished, into the memory tiles, up to 2.8 MB
in all: the host's work between multiplies overlaps with the transfer.

**The host's copies.** Packing `A` and unpacking `C` go through memory that is flushed from the CPU's caches for the
NPU, and are shared among the thread team.

A cost estimate in `owns` sends a call only when the NPU should be faster than the team's threads: a single row (`M = 1`)
stays on the CPU. Weights are packed into the NPU's format on first use, about 0.5 s for VoxCPM2, and kept in up to three
big buffers, because a job can name only five.

**Accuracy.** A multiply is off by about 0.9% (RMS, on random data) from rounding to 8-bit mantissas. The speech
decoder runs on the CPU in full precision, so this does not reach the audio as noise; it steers the flow
sampling slightly differently, as another seed would. Test: `tools/xdna/test_mul_mat.cpp`, every shape against the
exact product.

**Failure.** If the NPU cannot be opened the program says so once and stays on the CPU. If a multiply gets no answer in 2 s,
it says so, finishes that multiply and the rest of its graph on the CPU, and tries a new context in the background until one
runs a test job. A process killed in the middle of a graph leaves its job waiting for a doorbell until the driver's timeout, 60 s,
and until then the NPU runs nothing: the next process checks that at the start, stays on the CPU in the meantime, and moves over
by itself. Repeated failures wait longer before using the NPU again, up to half an hour.

## CPU changes this came with

These also speed up plain `--backend cpu`. All but the short-K multiply give bit-identical output:

- VoxCPM2 attention multiplies through `ggml_mul_mat`'s batch broadcast instead of copying the shared key and
  value heads with a chain of concats (95,000 fewer nodes per sentence).
- A multiply with a short `K` (attention over a few frames, tested in `tools/xdna/test_short_k.cpp`) goes through `src0` by
  `K`'s index, `N` values at a time, instead of through `vec_dot`'s scalar tail: 8 times faster. Only the order of the
  additions differs.
- `conv_transpose_1d` keeps its transposed kernel for weights instead of redoing it on every call.
- `repeat` of one value per channel fills rows on all threads; causal padding uses `fill` instead of multiplying a
  repeated slice by zero.
- Small nodes (up to `GGML_CPU_SERIAL_ELEMENTS`, default 8192 elements) run on thread 0 with no barrier per node.

## Limits

- `W` crosses memory on every call, so models with many large multiplies per second of audio remain bounded by
  memory speed. Everything that touches memory takes about 2.2 times as long next to a language model on the GPU, CPU and
  NPU alike, which is the drop in the table's second column. The margin next to an LLM is small.
- The decoder and everything else still need a few CPU cores.
- Only tested on Strix Halo. Other XDNA2 chips have the same array and should work.
- First-generation XDNA (Phoenix, Hawk Point) is a different array and is not supported by this core program.
- Two processes can share the NPU, but a job holds it for the length of a graph: the other waits.

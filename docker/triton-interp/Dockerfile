# CPU-only image that runs EchoKV's Triton kernels in Triton's interpreter.
# Lets kernel logic be tested on machines without a CUDA GPU (e.g. macOS):
#
#   docker build -t echokv-triton-interp docker/triton-interp
#   docker run --rm -v "$PWD":/work echokv-triton-interp \
#       python -m pytest -q -p no:cacheprovider tests/echokv_vllm
#
# Under TRITON_INTERPRET=1 every store write and decode in the tests goes
# through the Triton kernels and is checked against the PyTorch oracle. The
# interpreter executes the kernels' indexing and arithmetic exactly, but not
# the GPU compiler: compile errors and performance still need a real GPU.
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "triton==3.5.1" "numpy<2.3" pytest
# Pinned: triton 3.9's interpreter rejects `bool_block & (scalar >= 0)` masks,
# and numpy >= 2.3 breaks its int() on loaded scalars (`range(..., tl.load(...))`).
ENV TRITON_INTERPRET=1
WORKDIR /work

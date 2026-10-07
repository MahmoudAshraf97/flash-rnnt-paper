# Native dependencies

`manifest.json` pins the upstream repositories and compiler flags. `build`
fetches shallow checkouts into the configured external `artifact_root/sources/`
directory, checks each patch with `git apply --check`, then applies it.
Checkouts are validated before building and loading the extensions.

- **Apex:** no source changes.
- **Optimized Transducer:** promote ten vocabulary address products to 64-bit,
  avoid `max()` on an empty target tensor, and use `TORCH_EXTENSION_NAME` for the
  extension module name. Loss equations and floating-point operations are unchanged.
- **ModernGPU:** query clock attributes through the CUDA API instead of removed
  `cudaDeviceProp` fields. This changes device-information printing, not kernels.

`patched_sha256` records the benchmarked contents of modified files. Other
tracked files must match the pinned commit. Unexpected edits, missing patches
or different commits cause an error; the setup never resets existing checkouts.
Interrupted setup can leave an incomplete checkout; use a fresh artifact
directory rather than overwriting it.

To fetch and verify sources without a CUDA installation:

```bash
# From benchmarks/; the normal build command also performs this step.
uv run --locked python -m harness.backends --artifact-root ../../flash-rnnt-benchmark-artifacts
```

The patches modify upstream sources by Fangjun Kuang/Xiaomi (Optimized
Transducer) and Sean Baxter (ModernGPU). Their license notices are in `licenses/`
and their original copyright headers remain in the fetched files.

# Compute presets (`compute.yaml`)

relsynth's own engine runs on the CPU, and the default spec never uses a GPU. GPUs matter only to generator
plug-ins that declare a GPU need, such as the optional neural generators of a later release. A spec chooses the
devices those plug-ins may use with a **preset** and, for experts, **overrides**:

```yaml
synthesis:
  compute:
    preset: gpu-auto          # cpu (default) | gpu-auto | gpu-single | gpu-multi | a preset of `file`
    file: null                # optional: a compute.yaml with your own presets (relative to the spec)
    overrides:                # optional expert settings; each replaces the preset's field
      devices: [1]
      precision: bf16
      batch_size: 4096
      memory_limit: 20GiB
      threads: 8
```

## Presets

The built-in presets are in `relsynth/data/compute.yaml`.

| Preset | Device | GPUs | When no usable GPU is found |
|---|---|---|---|
| `cpu` (default) | CPU | 0 | Not applicable. A generator that *requires* a GPU is refused. |
| `gpu-auto` | GPU | 1 | Falls back to the CPU (`fallback: cpu`), for generators whose GPU need is optional. |
| `gpu-single` | GPU | 1 | Error (`fallback: error`). |
| `gpu-multi` | GPU | all visible | Error (`fallback: error`). |

## Fields

| Field | Values | Meaning |
|---|---|---|
| `device` | `cpu`, `gpu` | The device kind. |
| `gpus` | integer or `all` | How many GPUs to use. |
| `devices` | `null` or a list of ids | Explicit device ids. `null` takes the first `gpus` visible ones. |
| `backend` | `auto`, `cuda`, `rocm` | `auto` takes whichever is found, CUDA first. |
| `precision` | `fp32`, `tf32`, `bf16`, `fp16` | Numeric precision. A plug-in may refuse one. |
| `batch_size` | `auto` or integer | Rows per batch. `auto` lets the plug-in decide. |
| `memory_limit` | `null` or a size (`8GiB`, `512MiB`, `16GB`) | Memory per device. |
| `threads` | `auto` or integer | CPU threads a plug-in may use. |
| `fallback` | `cpu`, `error` | What happens when the GPUs asked for are not found. |

## How a generator gets its device

For each table, relsynth resolves the preset against the generator's declared `gpu` capability
(`relsynth.compute.resolve`):

- **`never`.** The generator always runs on the CPU. A GPU preset is recorded as a note, not an error. All built-in
  generators are in this class, including the default copula.
- **`optional`.** The generator gets the preset's GPUs when they are visible. When they are not, `fallback: cpu`
  runs it on the CPU and records why in the fit warnings, and `fallback: error` stops the run.
- **`required`.** The generator gets the GPUs, or the run stops with the reason. A CPU fallback is never taken, and
  the `cpu` preset refuses it.

So GPU use is always opt-in, and every fallback to the CPU is visible: it is written in the preset and recorded in
the log. The plan (device, device ids, precision, batch size, memory limit, threads) is handed to the plug-in as
`FitContext.compute`, and the fit log records the device of every non-default generator.

GPUs are found without importing a deep-learning framework. relsynth reads the device lists of `nvidia-smi -L` and
`rocm-smi --showid`, restricted by `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`. Setting `RELSYNTH_GPUS`
(for example `cuda:2` or `none`) replaces the probe, which is useful in containers and tests.

`relsynth compute --preset gpu-auto` shows the GPUs found and the device that each kind of generator would get.

## Your own presets

```yaml
# lab-compute.yaml
presets:
  lab-a100:
    device: gpu
    gpus: 2
    precision: bf16
    memory_limit: 70GiB
```

Name the file with `synthesis.compute.file: lab-compute.yaml` and the preset with `preset: lab-a100`. Fields you
leave out come from the built-in preset of the same name, or else from `cpu`.

# 2027ISCAS_SNN

Research code for the 2027 ISCAS SNN experiments, including model definitions,
quantization/error-injection experiments, and evaluation scripts.

## Repository boundary

The project-specific code is kept in this repository. The public SpikingJelly
source and the local compatibility snapshot are maintained separately under:

```text
/work/dependency/2027VLSI_SNN/spikingjelly_repo
/work/dependency/2027VLSI_SNN/spikingjelly_local_patch
```

Checkpoints, datasets, W&B runs, Python environments, and generated work
directories are intentionally excluded by `.gitignore`.

## Environment

The original environment was local to the server and is not part of this
repository. Recreate a suitable Conda environment from the package requirements
needed for the experiment before running scripts.

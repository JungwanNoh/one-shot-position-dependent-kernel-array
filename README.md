# 01_glovalvspdk_test

Task-first branch for comparing a **single fixed global kernel** against **position-dependent local refinement (PDK)**.

## Included tasks
- `synthetic`
- `lowlight`
- `natural`

Classification is intentionally excluded in this branch.

## Core idea
Each task is defined by its own visual goal first, then the global-vs-PDK comparison is applied.

- **synthetic**: structural feature extraction under spatially mixed corruption
- **lowlight**: make object-relevant low-light contours more visible while suppressing noisy background regions
- **natural**: make object-relevant contours more visible for recognition-oriented preprocessing

## Kernel bank update
This version moves away from only sigma-tuned Gaussian/unsharp kernels.

Available kernel families now include:
- `identity`
- `binomial`
- `gaussian`
- `box`
- `highboost` (binomial-based)
- `laplacian_sharpen`

Current default choice:
- **Global**: `binomial(k=3)` for all tasks
- **Preserve**: `identity(k=3)`
- **Recover**: `highboost(alpha=...)`
- **Suppress**: `gaussian` or `binomial`

## PDK rule
The zone rule is based on the failure mode of the global baseline.

- **preserve**: contours already preserved by the global baseline
- **recover**: candidate edges/contours weakened by the global baseline
- **suppress**: non-edge / noisy / clutter-dominated regions

Each task computes a task-specific **recover score**:
- synthetic: `missed-edge + coherence`
- lowlight: `missed-edge + coherence + brightness - noise`
- natural: `missed-edge + coherence - texture penalty`

## Panel layout (3x3)
Each saved panel contains:

Top row:
1. `Raw` (original grayscale image)
2. `Global` (task-specific edge/response map)
3. `PDK` (task-specific edge/response map)

Middle row:
4. `Clean / GT / Raw reference`
5. `Global processed`
6. `PDK processed`

Bottom row:
7. `Global Kernel`
8. `Zone Score (recover)`
9. `Zone → Kernel`

Notes:
- `Global` and `PDK` in the top row are response maps.
- `Raw` in the top row is the original image, not a response image.
- The `Zone → Kernel` panel uses three enlarged mini-heatmaps for `preserve / recover / suppress`.
- Zone maps and recover score maps are also saved separately.

## Directory expectation
Update paths in `src/config.py` if needed.

Default paths:
- Synthetic / Natural: `../dat/BSDS300/images/test`
- Lowlight input: `../dat/LOL/eval15/low`
- Lowlight GT: `../dat/LOL/eval15/high`

## Running
Run individual tasks:

```bash
python main.py --task synthetic
python main.py --task lowlight
python main.py --task natural
```

Run all tasks sequentially:

```bash
python main.py --task all
```

## Output
Results are saved under:

```text
../res/01_glovalvspdk_test/
```

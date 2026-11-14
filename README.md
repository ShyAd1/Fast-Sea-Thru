# download

This directory contains the code and requirements for downloading the training and test datasets used for FathomNetCLEF 2026, plus the underwater image restoration project (`proyect/`, a fast Sea-Thru implementation).

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt` (available via PyPI)

## Installation

```bash
pip install -r requirements.txt
```

## Usage for download images

```bash
python download.py [-h] [-o OUTPUT_DIR] [--min-workers MIN_WORKERS] [--max-workers MAX_WORKERS] [--initial-workers INITIAL_WORKERS] dataset_path
```

The script will autoscale the number of workers based on server failures.

## Usage for carpet proyect

Underwater image restoration (Sea-Thru): MiDaS depth estimation, backscatter removal, illuminant map (CUDA) and wideband attenuation recovery.

### Requirements

- Python 3.10+
- Python packages from `requirements.txt` (PyTorch, OpenCV, NumPy, SciPy, scikit-image, timm)
  - `scikit-image` is needed for `--postprocess-mode original`; without it that mode silently falls back to the OpenCV (`fast`) version
- CUDA-capable GPU recommended for speed (CPU backends are available but slower)
- Optional, for RAW image input: `pip install rawpy`
- To build the CUDA illuminant DLL: CUDA Toolkit (`nvcc`) and Microsoft Visual C++ Build Tools

```bash
pip install -r requirements.txt
```

### Build the CUDA illuminant DLL

Run `build.bat` from the `proyect\` directory (add `--pause` to keep the window open). It is equivalent to:

```bash
nvcc -O3 --shared -arch=native -o .\illuminant_cuda.dll .\illuminant.cu -ccbin "<MSVC>\bin\Hostx64\x64"
```

- The MSVC host-compiler folder is the `CCBIN` variable at the top of `build.bat` (or set the `CCBIN` environment variable). If that folder does not exist the script tries to load the Visual Studio environment (`vcvars64.bat`) instead.
- `nvcc` is taken from `PATH`, or from the newest CUDA toolkit installed under `Program Files`.
- On Linux: `nvcc -O3 --shared -arch=native -Xcompiler -fPIC -o illuminant_cuda.so illuminant.cu`.
- `proyect.py` looks for the library next to itself and in the current directory (`illuminant_cuda.dll/.so`), or at the path in the `SEATHRU_DLL` environment variable. A successful load prints `[dll] Loaded: ...`.

### Run (single image, with diagnostics)

```bash
cd "C:\Users\adhev\Documents\GitHub\Fast-Sea-Thru" && python proyect/proyect.py images_train/01380611-b0c2-4b8d-8d01-01b9b90d9e88.png -o output_mejorado --model DPT_Large --far 10 --illuminant-backend cuda --depth-colormap gray --depth-vis-min 0 --depth-vis-max 10 --save-depth-npy --save-intermediates
```

CLI syntax:

```bash
python proyect/proyect.py input [input ...] [options]
```

Where input can be one or more image files and/or folders.

### Full parameter list

- Input and output
  - input (positional, one or more): image paths and/or folders
  - --output, -o: output directory (default: output)
  - --prefix: optional filename prefix for generated outputs
  - --batch-size: only groups the progress output into `Batch i/n` headers (default: 1). It does **not** limit how many images are processed and does not speed anything up; images are always processed one after another
  - --limit: process at most N images (0 = all, default). Images are taken in sorted (alphabetical) order, after `--skip-existing` is applied
  - --skip-existing: skip images whose `<prefix><name>_restored.png` already exists in the output folder (resume an interrupted run or process a big folder in chunks)
  - --recursive: when an input is a folder, include images in subdirectories
  - --seed: random seed for the multi-restart fits (default: 0, makes runs reproducible)

- Depth source and depth estimation
  - --mode: Predict | Map | Hybrid (default: Predict)
    - Predict: MiDaS-only depth
    - Map: provided depth map only (requires --depth)
    - Hybrid: combines provided depth map + MiDaS (requires --depth)
  - --depth: depth map file or depth directory (required for Map/Hybrid)
  - --hint: optional depth hint file/directory to rescale predicted depth
  - --model: MiDaS_small | DPT_Hybrid | DPT_Large (default: MiDaS_small)
  - --depth-shift: Predict mode, disparity offset in `depth = 10 / (9.5 * disp + shift)` (default: 2.0; the old behavior was 0.5). Larger values compress the far tail of the depth map; see "Depth quality and color blotches"
  - --depth-detail-size: EXPERIMENTAL, Predict mode: also run MiDaS at this long side (e.g. 768) and merge its fine detail into the standard-resolution depth (default 0 = off). If the extra pass fails it falls back to single scale with a warning
  - --far: manual scene far distance in meters for Predict mode. It mostly sets the *units* of the depth map (and the meaning of `--depth-vis-max` and `--att-cap`); it barely changes the restored image (see below)
  - --max-side: downscale the longest side to N pixels (default: 1280). Images are only ever reduced, never enlarged, so values above the image size do nothing
  - --depth-debias-sigma-frac: low-frequency disparity debias (default: 0.0; 0.03-0.08 recommended underwater)
  - --z-min: minimum valid depth when loading map depth (default: 0.1)
  - --z-inf: fallback depth for zero-valued map pixels (default: 6.0)

- Illuminant computation
  - --illuminant-iters: illuminant iterations (default: 200)
  - --illuminant-window-size: neighborhood window size (default: 7; even values behave like +1)
  - --illuminant-block-x: CUDA block X dimension, maps to image columns (default: 32)
  - --illuminant-block-y: CUDA block Y dimension, maps to image rows (default: 16)
    - `block-x * block-y` must be <= 1024, otherwise 32x8 is used
  - --illuminant-backend: auto | cuda | window-cpu | bfs-cpu (default: auto)
    - `auto` silently falls back to the (much slower) CPU backend if the DLL is missing or fails; use `cuda` to fail loudly instead
  - --illuminant-p: weight of the direct signal vs. the neighborhood average (default: 0.4)
  - --illuminant-f: illuminant output scale (default: 2.0)

- Backscatter and attenuation fitting
  - --bs-attempts: random restarts for backscatter fit (default: 40)
  - --att-attempts: random restarts for attenuation fit (default: 12)
  - --fit-steps: optimization steps for GPU fitting (default: 120)
  - --cpu-fits: force scipy CPU fitting instead of GPU fitting
  - --ref-frac: fraction of darkest points per depth bin as references (default: 0.03)
  - --att-method: robust | legacy (default: robust)
    - robust: fits 48 per-depth-bin statistics (mean illuminant per bin) with a weighted curve fit; fast, needs no GPU memory and is not dominated by black/saturated pixels
    - legacy: original per-pixel fit (random subsample of up to 200k points)
  - --att-order: rgb | none (default: rgb)
    - rgb: force attenuation R >= G >= B (physical prior for water). This removes channel inversions from noisy fits, a typical source of magenta/green color casts
    - none: keep the raw per-channel fits (use for turbid/green water if the prior does not hold)

- Recovery controls
  - --beta-depth-cap: cap for attenuation\*depth term, <=0 disables cap (default: 1.2)
  - --att-cap: attenuation coefficient cap, <=0 disables cap (default: 1.2)
  - The log prints `[cap] pixels at the beta*z cap R/G/B`. If it is near 100%, the cap (not the physics) decides the result: every pixel gets the same gain and the depth map is effectively ignored. Raise `--beta-depth-cap`
  - On the test image (ROV close-up, `--depth-shift 2`): 0.75/0.85 gave a conservative, natural look (100% of pixels capped); the default 1.2/1.2 kept more depth-dependent correction (about half capped) with no color cast; 1.5/1.5 started clipping highlights. Caps off was over-saturated

- Output and diagnostics
  - --save-depth: save color depth PNG (enabled by default)
  - --no-save-depth: disable saving color depth PNG
  - --depth-colormap: magma | gray | inferno | viridis | turbo (default: magma). `gray` writes a single-channel 8-bit PNG, brighter = farther. Visualization only: it never affects the restoration
  - --save-depth-npy: also save raw depth arrays (.npy)
  - --save-depth-gray: also save normalized 16-bit grayscale depth PNG
  - --depth-vis-min: fixed min depth for visualization
  - --depth-vis-max: fixed max depth for visualization
  - --save-intermediates: save direct signal and illuminant map images

- Brightness / tone (applied before the result is clipped)
  - --tone-target: target median luminance of the lit pixels (default `0.38`; `0` turns it off and restores the old hard clip). Rescales the *unclipped* recovered radiance to a white point, brightens midtones with a luminance-only gamma (never darkens, gamma >= 0.5), keeps near-black areas black (the lift fades out below ~0.2 luminance, so dark open water does not turn into a blue haze) and rolls highlights off smoothly instead of clipping them. Colors are not shifted (the same gain is applied to R, G and B)
  - --tone-white-pct: percentile of the brightest channel mapped to white (default 99.7; `99` gives a slightly brighter, punchier image and clips a little more)

- Postprocessing
  - --postprocess: enable final postprocess step (CLAHE + white balance; it amplifies contrast, noise and any existing color cast, so compare with and without it)
  - --postprocess-mode: original | fast (default: original)

### Depth quality and color blotches (Predict mode)

MiDaS gives only *relative* inverse depth (disparity), converted to meters with `depth = 10 / (9.5 * disp + shift)`.

- **Problem:** with the old offset (0.5) a small patch of low disparity became up to ~20x farther than its surroundings. Dark regions (shadows, recesses) tend to get low disparity, so they were treated as very distant. With loose caps this produced strong yellow/cyan blotches around those regions in the restored image. Tight caps (e.g. 0.75/0.85) hid the problem by giving every pixel the same gain.
- **Fix:** `--depth-shift 2.0` (default) compresses that far tail (far/near ratio about 6 instead of 20). On the test image the blotch disappeared at shift 2 and was mostly gone at 1.5. Use `--depth-shift 0.5` to reproduce the old behavior.
- **Global scale is irrelevant:** multiplying the whole depth map by a constant (what `--far` does) changes the restored image by about 0.5% on the test image, because the fitted attenuation coefficients scale inversely. Do not spend time tuning `--far` for color; only the *relative shape* of the depth matters.
- **Smoothness:** MiDaS runs internally at ~384 px, so the depth map is smooth and misses fine structure. Post-filtering it with an image-guided filter did not help in tests (it changes the map by about 1% and at high strength only imprints image texture). The experimental `--depth-detail-size 768` merges a higher-resolution pass instead; it has not been validated on real GPU runs, so compare the `_depth.png` and the restored image with and without it. Darkness-versus-distance is also partly physical under vehicle lamps (farther surfaces really are darker), so some correlation is expected.

### Brightness and blown highlights

The recovered radiance used to be hard-clipped to [0, 1]. Strongly amplified bright areas (pale sponges, light rock) then saturated to flat white and the information was lost, while the rest of the image could still look dark. `--tone-target` (on by default, `0.38`) tone-maps the *unclipped* radiance instead.

Measured results (clipped = share of pixels with a channel >= 0.98):

- Rock close-up: 23% clipped -> 0.2%; the texture of the bright rock came back.
- Sponge scene (image where 100% of pixels sat at the `beta*z` cap, so the old output was a uniform gain): 7.5% clipped -> 0.2%; the sponges go from flat white to cream with visible pores, the seabed gets a little brighter and the dark water stays dark. That preview was computed by reconstructing the unclipped radiance from the old output and the original image, not by re-running the full pipeline.

Very dark scenes are lifted only moderately (the gamma is bounded), and open black water far from the camera cannot be recovered by this method. Use `--tone-target 0` to get the previous behavior.

### Useful notes

- --depth is mandatory when --mode is Map or Hybrid.
- --save-depth is enabled by default; use --no-save-depth to disable it.
- You can pass multiple inputs in one command (files and folders mixed). A failure on one image is reported and the batch continues.
- GPU memory: the illuminant DLL runs in its own CUDA runtime, so PyTorch's cached memory is released before each call, and again after each image.
- Sea-Thru assumes ambient (sun) light. Scenes lit by lamps mounted on the vehicle break that assumption, so expect imperfect color; judge results against the original and keep postprocessing conservative.

### Batch processing without extra files

Pointing the script at a folder processes **every** image in it (MiDaS is loaded once; a failure on one image is reported and the batch continues). `--batch-size` does not change that, so to process only some images use `--limit`.

```bash
# the first 20 images of the folder (alphabetical order)
cd "C:\Users\adhev\Documents\GitHub\Fast-Sea-Thru" && python proyect/proyect.py images_test -o output_lote --model DPT_Large --far 10 --illuminant-backend cuda --max-side 3840 --no-save-depth --limit 20 2>&1 | tee lote.log

# the whole folder, resumable: re-run the same command and finished images are skipped
cd "C:\Users\adhev\Documents\GitHub\Fast-Sea-Thru" && python proyect/proyect.py images_test -o output_lote --model DPT_Large --far 10 --illuminant-backend cuda --max-side 3840 --no-save-depth --skip-existing 2>&1 | tee -a lote.log

# next 20 not yet processed (chunks): same command with --skip-existing --limit 20
```

- The first line of the log reports `[batch] N image(s) found, S already done (skipped), M to process`; check it before a long run. After every image it prints the time taken, the elapsed time and an ETA, and at the end a summary (`saved / failed / skipped`, minutes total, seconds per image).
- Only `<name>_restored.png` is written per image. The extra files are opt-in (`--save-depth-npy`, `--save-depth-gray`, `--save-intermediates`) and the depth PNG is on by default, so `--no-save-depth` switches it off.
- Approximate disk use per 1920x1080 image: restored PNG ~4 MB. With the diagnostic flags (`--save-depth-npy` ~16 MB, `--save-intermediates` ~8 MB, depth PNG ~0.3 MB) it is ~28 MB per image. For 1,400 images that is ~5.6 GB versus ~40 GB.
- Keep `--max-side` at or above the image size (3840 keeps 1920x1080 untouched). The default is 1280, which silently downsizes larger images.
- `--illuminant-backend cuda` makes a CUDA problem show up as an error for that image instead of silently falling back to the much slower CPU backend.
- `| tee lote.log` (Git Bash) also saves the whole log (use `tee -a` to append when resuming), so you can check the `[cap]` percentages and any `failed` lines afterwards. In PowerShell use `| Tee-Object lote.log`; in cmd use `> lote.log 2>&1`.
- The output folder must not be the input folder (otherwise the `*_restored.png` files would be picked up as inputs on the next run). Add `--recursive` to include subfolders, or pass several files/folders at once.
- Open the results next to the originals and look for: leftover color casts, blotches near dark regions, blown highlights, or `[cap] ... 100%` in the log.

### Quick presets

- Faster run:
  - --max-side 720 --model MiDaS_small --illuminant-iters 120 --fit-steps 60

- Higher quality run:
  - --max-side 1920 --model DPT_Hybrid --illuminant-iters 300 --fit-steps 140

### Recommended tuning workflow

1. Run with `--save-intermediates --depth-colormap gray`, no `--postprocess`, `--illuminant-backend cuda`.
2. Check the `[cap]` percentages in the log: if they are near 100% the depth map is being ignored; raise `--beta-depth-cap`.
3. Adjust `--beta-depth-cap` / `--att-cap` (start from the defaults 1.2) and, if you see color blotches near dark regions, try `--depth-shift` 1.5 to 3.
4. Tone mapping is already on (`--tone-target 0.38`); try `--tone-white-pct 99` for a brighter image, or `--tone-target 0` to disable it.
5. Only then try `--postprocess --postprocess-mode fast` (or `original`).

Tip: change one parameter group at a time and compare output visually before moving to the next group.

### Troubleshooting

- `[dll] Failed to load` / `[illu] CUDA DLL unavailable`: rebuild with `build.bat`, keep the DLL next to `proyect.py`, and make sure the CUDA runtime DLLs are reachable (install folders under `Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin` and `CUDA_PATH` are registered automatically).
- `[cuda] ... failed at ...`: the message names the failing CUDA call. The DLL frees its buffers and the call is retried once; if it fails again the run falls back to CPU (or stops with `--illuminant-backend cuda`). Out-of-memory errors usually mean other processes are using the GPU.
- `[bs-gpu]` / `[att-gpu] failed (...)`: the fit falls back to the CPU automatically.
- Yellow/cyan/magenta blotches around dark regions: the depth map overestimates distance there. Raise `--depth-shift` (default 2.0), or lower the caps.
- `[midas] detail pass failed`: the experimental `--depth-detail-size` pass could not run (usually VRAM); the run continues with the standard depth.
- `[midas] non-finite fp16 output`: the depth network overflowed in half precision and was recomputed in fp32 (no action needed).

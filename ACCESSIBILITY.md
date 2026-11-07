# Accessibility Statement

We want this project to be usable by as many people as possible.

## Commitment

- Documentation is written in plain text/Markdown so it works with screen readers and
  text-based tools.
- The tool is operated from the command line; all functionality is available through
  documented parameters, with no mouse-only interactions.
- Error messages and logs are plain text, prefixed by tags (e.g. `[dll]`, `[cap]`).

## Supported environments

- **OS:** Windows and Linux (CUDA build scripts are Windows-first; Linux instructions
  are in the README).
- **Python:** 3.10+.
- **Hardware:** CUDA-capable GPU recommended; CPU backends are available.

## Known limitations

- Building the CUDA library requires a C++ toolchain and CUDA Toolkit, which can be a
  barrier on some systems. CPU fallbacks are slower but need no compilation.
- Some depth visualizations use color maps. Use `--depth-colormap gray` (or `viridis`)
  for a single-channel or perceptually uniform, color-blind-friendlier output.
- Output images do not include alternative text; if you publish results, please add
  descriptive captions or alt text.
- Documentation is currently available in English only.

## Feedback

If you find an accessibility barrier, open an issue with the _accessibility_ label or
write to **adheva05@gmail.com**. We aim to respond within 14 days.

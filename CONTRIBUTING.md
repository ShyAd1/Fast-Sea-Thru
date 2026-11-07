# Contributing

Thank you for your interest in improving this project! Contributions of all kinds are
welcome: bug reports, documentation fixes, new features and performance improvements.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug:** open an issue using the *Bug report* template.
- **Suggest an improvement:** open an issue using the *Feature request* template.
- **Fix or improve something:** fork the repository and open a pull request.
- **Report a vulnerability:** do **not** open a public issue; see [SECURITY.md](SECURITY.md).

## Development setup

1. Fork and clone the repository.
2. Create a virtual environment (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. (Optional, for the CUDA illuminant backend) build the library as described in the
   README: `build.bat` on Windows, or on Linux:
   ```bash
   nvcc -O3 --shared -arch=native -Xcompiler -fPIC -o illuminant_cuda.so illuminant.cu
   ```
   Without a GPU you can still work with `--illuminant-backend window-cpu`.

## Workflow

1. Create a branch from `main`: `git checkout -b feature/short-description`.
2. Make focused changes: one logical change per pull request.
3. Test your change on at least one real image (see the *Recommended tuning workflow*
   in the README) and compare against the previous output.
4. Commit with clear messages (e.g. `fix: clamp depth when disparity is zero`).
5. Push and open a pull request using the provided template.

## Guidelines

- **Python:** follow [PEP 8](https://peps.python.org/pep-0008/); keep functions small
  and documented; keep CLI flags backward compatible when possible and document any new
  one in the README parameter list.
- **CUDA/C++:** check every CUDA call for errors and free all buffers on failure paths.
- **Reproducibility:** respect `--seed`; do not introduce hidden randomness.
- **Dependencies:** avoid adding new ones unless clearly justified; update
  `requirements.txt` if you do.
- **Do not commit** datasets, model weights, large images, or compiled binaries
  (`.dll`, `.so`).
- **Documentation:** update the README when behavior or parameters change.

## Reporting bugs well

Include your OS, Python version, GPU and CUDA version, the exact command, and the
console log (the `[dll]`, `[cap]` and error lines are especially useful). If possible,
attach a small image that reproduces the problem.

## Questions

Open an issue with the *question* label or a discussion and we will help as soon as
possible.

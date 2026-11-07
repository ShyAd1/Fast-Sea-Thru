# Security Policy

## Supported versions

Security fixes are applied to the latest code on the `main` branch. Older commits and
forks are not maintained.

| Version         | Supported |
| --------------- | --------- |
| `main` (latest) | ✅        |
| Older commits   | ❌        |

## Reporting a vulnerability

Please **do not report security issues in public issues or pull requests**.

Use one of these private channels:

1. GitHub's **Private vulnerability reporting**:
   _Security → Report a vulnerability_ in this repository.
2. Email: **adheva05@gmail.com** with the subject `[SECURITY] <short summary>`.

Please include a description of the issue, steps to reproduce, affected files or
versions, and the potential impact.

You can expect an acknowledgement within **7 days** and a status update within
**30 days**. Once a fix is available, the issue will be disclosed responsibly and you
will be credited if you wish.

## Scope notes

Areas that deserve particular care:

- **Native library loading:** `proyect.py` loads `illuminant_cuda.dll/.so` from its own
  folder, the working directory, or the path in the `SEATHRU_DLL` environment variable.
  Only load libraries you built yourself or trust.
- **Image and RAW parsing:** processing untrusted images relies on third-party
  libraries (OpenCV, rawpy, scikit-image); keep them updated.
- **Dataset download script:** `download.py` fetches remote files; review the dataset
  source before running it.
- **Model weights:** MiDaS weights are fetched through `torch.hub`; verify the source.

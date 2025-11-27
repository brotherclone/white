# project-level sitecustomize to configure runtime behavior early
# This file will be imported automatically by Python (if the project root
# is on sys.path) before other application code. We use it to suppress
# noisy DeprecationWarnings introduced by the `standard-aifc` and
# `standard-sunau` backports when running on Python 3.13.

import warnings

# Suppress deprecation warnings mentioning `aifc` and `sunau` (backports)
warnings.filterwarnings("ignore", message=r".*aifc.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*sunau.*", category=DeprecationWarning)

# NOTE:
# - This silences the specific runtime warnings about the `aifc`/`sunau`
#   stdlib modules being removed in Python 3.13 and replaced by backports.
# - Prefer migrating away from `aifc`/`sunau` (use `soundfile`/`pydub`) in the
#   longer term; this is a low-risk short-term mitigation to keep logs clean.

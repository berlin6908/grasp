#!/usr/bin/env bash
set -euo pipefail

"${PYTHON:-python}" -m nbconvert --to notebook --execute --inplace local-results.ipynb

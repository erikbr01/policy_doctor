#!/usr/bin/env bash
# Install the ZED SDK + pyzed + NVIDIA video codec libs needed by
# scripts/convert_droid_hdf5_debug.py.
#
# Designed for hosts where:
#   - the NVIDIA driver is installed (we don't touch it) but is headless and
#     lacks libnvcuvid.so.1 / libnvidia-encode.so.1 (typical on GCE deep-learning
#     images);
#   - no system-wide install of the ZED SDK is desired.
#
# Everything is staged under $PREFIX (default: $HOME). The driver-extras .debs
# are downloaded with `apt-get download` and extracted in place — they are NOT
# installed system-wide, so the running driver is not replaced.
#
# Final layout (under $PREFIX):
#   zed_sdk_extracted/                ZED SDK 4.2 (cu12) extracted via --noexec
#   zed_settings/SN<serial>.conf      Camera calibration files
#   nvidia_decode_extra/usr/lib/...   libnvcuvid.so.1 + libnvidia-encode.so.1
#
# Mutations to the target conda env (default: policy_doctor):
#   - conda install -c conda-forge libjpeg-turbo
#   - pip install opencv-python-headless==4.6.0.66   (fixes missing libGL.so.1)
#   - pip install --no-deps pyzed-4.2 (cp39 wheel)
#
# After running, source the printed LD_LIBRARY_PATH line before invoking
# convert_droid_hdf5_debug.py.
#
# Re-runnable: existing artifacts are detected and skipped.

set -euo pipefail

PREFIX="${PREFIX:-$HOME}"
CONDA_ENV="${CONDA_ENV:-policy_doctor}"
ZED_SDK_URL="${ZED_SDK_URL:-https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu22}"
PYZED_WHL_URL="${PYZED_WHL_URL:-https://download.stereolabs.com/zedsdk/4.2/whl/linux_x86_64/pyzed-4.2-cp39-cp39-linux_x86_64.whl}"
# Default camera serials match the kendama DROID rig (wrist + ext1). Override
# with: CAMERA_SERIALS="14313307 36716034 37617599" ./install_droid_zed_deps.sh
CAMERA_SERIALS="${CAMERA_SERIALS:-14313307 36716034}"

ZED_SDK_DIR="${PREFIX}/zed_sdk_extracted"
ZED_SETTINGS_DIR="${PREFIX}/zed_settings"
NVIDIA_EXTRA_DIR="${PREFIX}/nvidia_decode_extra"

source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> Target conda env: ${CONDA_ENV}"
echo "==> Install prefix:   ${PREFIX}"
echo

# ---------------------------------------------------------------------------
# 1. ZED SDK
# ---------------------------------------------------------------------------
echo "Step 1/6  ZED SDK -> ${ZED_SDK_DIR}"
if [[ -d "${ZED_SDK_DIR}/lib" ]]; then
  echo "  already extracted — skip."
else
  installer="$(mktemp -t zed_sdk.XXXXXX.run)"
  trap 'rm -f "${installer}"' EXIT
  echo "  downloading SDK installer ..."
  curl -fSL -o "${installer}" "${ZED_SDK_URL}"
  chmod +x "${installer}"
  echo "  extracting (--noexec, no system install) ..."
  "${installer}" --noexec --target "${ZED_SDK_DIR}" > /dev/null
  rm -f "${installer}"
  trap - EXIT
fi

# ---------------------------------------------------------------------------
# 2. Camera calibration .conf files
# ---------------------------------------------------------------------------
echo "Step 2/6  Camera calibration -> ${ZED_SETTINGS_DIR}"
mkdir -p "${ZED_SETTINGS_DIR}"
for sn in ${CAMERA_SERIALS}; do
  dst="${ZED_SETTINGS_DIR}/SN${sn}.conf"
  if [[ -s "${dst}" ]]; then
    echo "  SN${sn}.conf already present — skip."
  else
    curl -fsSL "https://calib.stereolabs.com/?SN=${sn}" -o "${dst}"
    echo "  fetched SN${sn}.conf ($(wc -c <"${dst}") bytes)"
  fi
done

# ---------------------------------------------------------------------------
# 3. libnvcuvid + libnvidia-encode (extract .deb without installing)
# ---------------------------------------------------------------------------
echo "Step 3/6  NVIDIA video codec libs -> ${NVIDIA_EXTRA_DIR}"
NVIDIA_LIB_DIR="${NVIDIA_EXTRA_DIR}/usr/lib/x86_64-linux-gnu"
if [[ -f "${NVIDIA_LIB_DIR}/libnvcuvid.so.1" && -f "${NVIDIA_LIB_DIR}/libnvidia-encode.so.1" ]]; then
  echo "  libnvcuvid + libnvidia-encode already extracted — skip."
else
  # Match the running driver's major version (e.g. 580.x -> libnvidia-decode-580).
  driver_major="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
    | head -1 | cut -d. -f1)"
  if [[ -z "${driver_major}" ]]; then
    echo "  ERROR: could not determine NVIDIA driver major version via nvidia-smi" >&2
    exit 1
  fi
  echo "  detected NVIDIA driver major: ${driver_major}"
  mkdir -p "${NVIDIA_EXTRA_DIR}"
  tmp_debs="$(mktemp -d -t nvdebs.XXXXXX)"
  trap 'rm -rf "${tmp_debs}"' EXIT
  (
    cd "${tmp_debs}"
    apt-get download \
      "libnvidia-decode-${driver_major}" \
      "libnvidia-encode-${driver_major}"
  )
  for deb in "${tmp_debs}"/libnvidia-decode-*.deb "${tmp_debs}"/libnvidia-encode-*.deb; do
    [[ -f "${deb}" ]] || { echo "  ERROR: missing $deb" >&2; exit 1; }
    dpkg-deb -x "${deb}" "${NVIDIA_EXTRA_DIR}"
  done
  rm -rf "${tmp_debs}"
  trap - EXIT
fi

# ---------------------------------------------------------------------------
# 4. libjpeg-turbo (pyzed dlopen target)
# ---------------------------------------------------------------------------
echo "Step 4/6  libjpeg-turbo in ${CONDA_ENV}"
if conda run -n "${CONDA_ENV}" bash -lc \
   'python -c "import ctypes; ctypes.CDLL(\"libturbojpeg.so.0\")"' &>/dev/null; then
  echo "  libturbojpeg.so.0 already loadable — skip."
else
  conda install -y -n "${CONDA_ENV}" -c conda-forge libjpeg-turbo
fi

# ---------------------------------------------------------------------------
# 5. opencv-python-headless (avoid libGL.so.1 dep)
# ---------------------------------------------------------------------------
echo "Step 5/6  opencv-python-headless in ${CONDA_ENV}"
if conda run -n "${CONDA_ENV}" python -c "import cv2" &>/dev/null; then
  echo "  cv2 already imports — skip."
else
  # opencv-python from conda lacks RECORD and can't be uninstalled by pip; the
  # headless wheel takes precedence at import time once installed.
  conda run -n "${CONDA_ENV}" pip install --upgrade \
    'opencv-python-headless==4.6.0.66'
fi

# ---------------------------------------------------------------------------
# 6. pyzed wheel
# ---------------------------------------------------------------------------
echo "Step 6/6  pyzed in ${CONDA_ENV}"
LDP="${ZED_SDK_DIR}/lib:${NVIDIA_LIB_DIR}"
if LD_LIBRARY_PATH="${LDP}" conda run -n "${CONDA_ENV}" \
     python -c "import pyzed.sl" &>/dev/null; then
  echo "  pyzed already imports — skip."
else
  whl_tmp="$(mktemp -d -t pyzed.XXXXXX)"
  trap 'rm -rf "${whl_tmp}"' EXIT
  # Stereolabs serves the wheel under a non-canonical name; pip rejects that,
  # so download to the canonical filename before installing.
  whl_path="${whl_tmp}/pyzed-4.2-cp39-cp39-linux_x86_64.whl"
  curl -fSL -o "${whl_path}" "${PYZED_WHL_URL}"
  conda run -n "${CONDA_ENV}" pip install --no-deps "${whl_path}"
  rm -rf "${whl_tmp}"
  trap - EXIT
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "==> Verifying pyzed import ..."
LD_LIBRARY_PATH="${LDP}" conda run -n "${CONDA_ENV}" \
  python -c "import pyzed.sl as sl; print('pyzed import OK')"

cat <<EOF

Done. Before running scripts/convert_droid_hdf5_debug.py, set:

  export LD_LIBRARY_PATH=${LDP}
  conda activate ${CONDA_ENV}

Example:

  LD_LIBRARY_PATH=${LDP} \\
    conda run -n ${CONDA_ENV} python scripts/convert_droid_hdf5_debug.py \\
      --input_path /path/to/droid_data/data/success \\
      --output_path /path/to/out.hdf5 \\
      --zed_settings ${ZED_SETTINGS_DIR} \\
      --image_size 180 320 --num_workers 4
EOF

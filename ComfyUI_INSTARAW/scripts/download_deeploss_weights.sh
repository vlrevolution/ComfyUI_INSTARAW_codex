#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/pretrained/deeploss"
CACHE_DIR="${PROJECT_ROOT}/pretrained/cache"
TMP_DIR="$(mktemp -d)"
PARTS=(
  "https://vault.cs.uwaterloo.ca/s/ecQd9W24oDwwKja/download"
  "https://vault.cs.uwaterloo.ca/s/mL6ZxC9itgc9wCR/download"
  "https://vault.cs.uwaterloo.ca/s/iffQ9GPSSqyx2Zt/download"
  "https://vault.cs.uwaterloo.ca/s/9kqjC7KAPQkK5DX/download"
  "https://vault.cs.uwaterloo.ca/s/iJ8aQHcLeojACR5/download"
  "https://vault.cs.uwaterloo.ca/s/LqKMxiMXC5TkwQx/download"
  "https://vault.cs.uwaterloo.ca/s/9T5ZzYa26HqTAYB/download"
)
PART_SUFFIX=(aa ab ac ad ae af ag)
ARCHIVE_ROOT="pretrained_models/loss_provider/weights"
WEIGHT_FILE="${ARCHIVE_ROOT}/rgb_pnet_lin_vgg_trial0.pth"
TARGET_FILE="${DEST_DIR}/rgb_pnet_lin_vgg_trial0.pth"
DIRECT_URL="${DEELOSS_DIRECT_URL:-}"
LOCAL_FILE="${DEELOSS_LOCAL_FILE:-}"

mkdir -p "${DEST_DIR}" "${CACHE_DIR}"

if [[ -f "${TARGET_FILE}" ]]; then
  echo "✅ Deeploss weight already present at ${TARGET_FILE}"
  exit 0
fi

if [[ -n "${LOCAL_FILE}" ]]; then
  if [[ -f "${LOCAL_FILE}" ]]; then
    cp "${LOCAL_FILE}" "${TARGET_FILE}"
    echo "✅ Copied Deeploss weight from ${LOCAL_FILE}"
    exit 0
  else
    echo "⚠️  DEELOSS_LOCAL_FILE=${LOCAL_FILE} does not exist; ignoring."
  fi
fi

if [[ -n "${DIRECT_URL}" ]]; then
  echo "➡️  Downloading Deeploss weight from custom URL..."
  curl -L -o "${TARGET_FILE}" "${DIRECT_URL}"
  echo "✅ Deeploss weight saved to ${TARGET_FILE}"
  exit 0
fi

echo "➡️  Downloading Deeploss weights (this can be several GB)..."

for idx in "${!PARTS[@]}"; do
  url="${PARTS[$idx]}"
  suffix="${PART_SUFFIX[$idx]}"
  target="${CACHE_DIR}/pretrained_models.tar.gz.${suffix}"
  if [[ ! -f "${target}" ]]; then
    echo "  - Fetching part ${suffix}"
    curl -L -o "${target}" "${url}"
  else
    echo "  - Reusing cached part ${suffix} (${target})"
  fi
done

echo "➡️  Extracting Deeploss weight from archive..."
cat "${CACHE_DIR}"/pretrained_models.tar.gz.* | tar xzvf - -C "${TMP_DIR}" "${WEIGHT_FILE}"

SRC_PATH="${TMP_DIR}/${WEIGHT_FILE}"
if [[ ! -f "${SRC_PATH}" ]]; then
  echo "❌ Failed to extract ${WEIGHT_FILE}"
  echo "   Temporary files kept at ${TMP_DIR} for inspection."
  exit 1
fi

install_path="${DEST_DIR}/rgb_pnet_lin_vgg_trial0.pth"
mv "${SRC_PATH}" "${install_path}"

FILE_SIZE=$(stat -c%s "${install_path}")
if [[ "${FILE_SIZE}" -lt 1000000 ]]; then
  echo "❌ Extracted file looks too small (${FILE_SIZE} bytes)."
  echo "   Check the downloaded parts in ${CACHE_DIR} (maybe an HTML error page)."
  echo "   The temp dir ${TMP_DIR} is kept for inspection."
  exit 1
fi

echo "✅ Deeploss weight installed to ${install_path}"

echo "🧹 Cleaning up temporary files..."
rm -rf "${TMP_DIR}"

echo ""
echo "Done. Cached parts are stored in ${CACHE_DIR} (delete them later if you wish)."
echo "Set DEEPLOSS_WEIGHTS_DIR=${DEST_DIR} or leave unset; the code autodetects this path."

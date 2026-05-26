#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-thesis/build}"
MAIN_PDF="${BUILD_DIR}/main.pdf"
COVER_PDF="${BUILD_DIR}/封面.pdf"
COMMITMENT_PDF="${BUILD_DIR}/承诺书.pdf"
OUTPUT_PDF="${BUILD_DIR}/main_final.pdf"

if ! command -v pdfinfo >/dev/null 2>&1; then
  echo "error: pdfinfo is required but not found" >&2
  exit 1
fi

if ! command -v pdfseparate >/dev/null 2>&1; then
  echo "error: pdfseparate is required but not found" >&2
  exit 1
fi

if ! command -v pdfunite >/dev/null 2>&1; then
  echo "error: pdfunite is required but not found" >&2
  exit 1
fi

if [[ ! -f "${MAIN_PDF}" ]]; then
  echo "error: main PDF not found: ${MAIN_PDF}" >&2
  exit 1
fi

if [[ ! -f "${COVER_PDF}" ]]; then
  echo "error: cover PDF not found: ${COVER_PDF}" >&2
  exit 1
fi

if [[ ! -f "${COMMITMENT_PDF}" ]]; then
  echo "error: commitment PDF not found: ${COMMITMENT_PDF}" >&2
  exit 1
fi

PAGE_COUNT="$(pdfinfo "${MAIN_PDF}" | awk '/^Pages:/ { print $2 }')"
if [[ -z "${PAGE_COUNT}" || "${PAGE_COUNT}" -lt 4 ]]; then
  echo "error: ${MAIN_PDF} must have at least 4 pages" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

LAST_BODY_PAGE=$((PAGE_COUNT - 1))
pdfseparate -f 3 -l "${LAST_BODY_PAGE}" "${MAIN_PDF}" "${TMP_DIR}/page-%04d.pdf"

REST_PAGES=("${TMP_DIR}"/page-*.pdf)
pdfunite "${COVER_PDF}" "${REST_PAGES[@]}" "${COMMITMENT_PDF}" "${OUTPUT_PDF}"

echo "wrote ${OUTPUT_PDF}"

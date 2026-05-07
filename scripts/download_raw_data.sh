#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${REPO_ROOT}/raw_data"
FORCE=0
KEEP_TEMP=0

PTBXL_URL=""
PTBXL_ARCHIVE=""
PTBXL_SOURCE=""
PTBXL_URL_DEFAULT="https://physionet.org/files/ptb-xl/1.0.1/"
CPSC_URL=""
CPSC_ARCHIVE=""
CPSC_SOURCE=""
CPSC_URL_DEFAULT="http://hhbucket.oss-cn-hongkong.aliyuncs.com/TrainingSet3.zip"
G12EC_URL=""
G12EC_ARCHIVE=""
G12EC_SOURCE=""
G12EC_URL_DEFAULT="https://storage.googleapis.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_E.tar.gz"

TMP_DIR=""

usage() {
    cat <<'EOF'
Usage: scripts/download_raw_data.sh [OPTIONS]

Prepare raw ECG datasets under <target-dir> for this repository.

Options:
  --target-dir PATH        Root directory for raw data (default: <repo>/raw_data)
  --force                  Overwrite existing dataset folders
  --keep-temp              Keep temporary download files

  --ptbxl-url URL          URL to PTB-XL archive
  --ptbxl-archive PATH     Local PTB-XL archive file
  --ptbxl-source-dir PATH  Existing PTB-XL directory to copy/link

  --cpsc-url URL           URL to CPSC2018 archive
  --cpsc-archive PATH      Local CPSC2018 archive file
  --cpsc-source-dir PATH   Existing CPSC2018 directory to copy/link

  --g12ec-url URL          URL to G12EC archive
  --g12ec-archive PATH     Local G12EC archive file
  --g12ec-source-dir PATH   Existing G12EC directory to copy/link

  --help                   Show this help and exit

Notes:
- This script only prepares the required local folder layout:
  - raw_data/PTBXL/1.0.1/ptbxl
  - raw_data/CPSC2018
  - raw_data/G12EC/WFDB_v230901
- If URL sources are not available, please download manually and run with
  --<dataset>-source-dir.
EOF
}

cleanup() {
    if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}"
    fi
}
trap cleanup EXIT

command_exists() {
    command -v "${1}" >/dev/null 2>&1
}

download_file() {
    local url="${1}"
    local out_path="${2}"
    if command_exists wget; then
        wget -c --content-disposition -O "${out_path}" "${url}"
    elif command_exists curl; then
        curl -L --fail -o "${out_path}" "${url}"
    else
        echo "ERROR: Neither wget nor curl is installed." >&2
        return 1
    fi
}

is_archive_url() {
    local url="${1}"
    case "${url}" in
        *.tar.gz|*.tgz|*.tar.bz2|*.tar.xz|*.zip)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

download_from_url() {
    local dataset_name="${1}"
    local url="${2}"
    local target_dir="${3}"

    local base_dir
    local leaf_dir
    local candidate
    local found

    TMP_DIR="$(mktemp -d)"

    if is_archive_url "${url}"; then
        local file_name="dataset_$(printf '%s' "${dataset_name}" | tr 'A-Z' 'a-z').archive"
        local tmp_archive="${TMP_DIR}/${file_name}"
        echo "[info] ${dataset_name}: downloading archive from ${url}"
        download_file "${url}" "${tmp_archive}"
        extract_archive "${tmp_archive}" "${target_dir}"
        return 0
    fi

    if ! command_exists wget; then
        echo "ERROR: wget is required to download directory source for ${dataset_name}: ${url}" >&2
        return 1
    fi

    echo "[info] ${dataset_name}: downloading directory tree from ${url}"
    leaf_dir="$(printf '%s' "${url}" | sed -E 's#/$##')"
    base_dir="$(basename "${leaf_dir}")"
    wget -r -N -c -np -nH -R "index.html*" -P "${TMP_DIR}" "${leaf_dir}"

    candidate="${TMP_DIR}/${base_dir}"
    if [[ -d "${candidate}" ]]; then
        copy_source_dir "${candidate}" "${target_dir}"
        return 0
    fi

    found="$(find "${TMP_DIR}" -type d -name "${base_dir}" | head -n 1)"
    if [[ -n "${found}" && -d "${found}" ]]; then
        copy_source_dir "${found}" "${target_dir}"
        return 0
    fi

    echo "ERROR: unable to locate downloaded dataset files for ${dataset_name} under ${TMP_DIR}" >&2
    return 1
}

extract_archive() {
    local archive="${1}"
    local out_dir="${2}"

    mkdir -p "${out_dir}"
    case "${archive}" in
        *.tar.gz|*.tgz)
            tar -xzf "${archive}" -C "${out_dir}"
            ;;
        *.tar.bz2)
            tar -xjf "${archive}" -C "${out_dir}"
            ;;
        *.tar.xz)
            tar -xJf "${archive}" -C "${out_dir}"
            ;;
        *.zip)
            if ! command_exists unzip; then
                echo "ERROR: unzip is required for ${archive}" >&2
                return 1
            fi
            unzip -q "${archive}" -d "${out_dir}"
            ;;
        *)
            echo "ERROR: Unsupported archive format: ${archive}" >&2
            return 1
            ;;
    esac
}

copy_source_dir() {
    local src="${1}"
    local dst="${2}"

    if [[ ! -d "${src}" ]]; then
        echo "ERROR: source directory not found: ${src}" >&2
        return 1
    fi
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
}

is_nonempty_dir() {
    local d="${1}"
    [[ -d "${d}" && "$(ls -A "${d}")" ]]
}

ensure_rooted_dir() {
    local dataset_dir="${1}"
    local expected_path="${2}"
    if [[ -f "${expected_path}" || "${dataset_dir}" == "${expected_path}" ]]; then
        return 0
    fi
    return 1
}

check_ptbxl() {
    local root="${1}"
    local ok=0
    [[ -f "${root}/ptbxl_database.csv" ]] && ok=$((ok+1))
    [[ -f "${root}/scp_statements.csv" ]] && ok=$((ok+1))
    [[ -d "${root}/records500" ]] && ok=$((ok+1))
    if (( ok < 3 )); then
        return 1
    fi
    find "${root}/records500" -type f -name "*_hr" | head -n 1 | grep -q .
}

check_cpsc() {
    local root="${1}"
    [[ -f "${root}/TrainingSet3/REFERENCE.csv" ]] || return 1
    find "${root}" -type f -name "*.mat" | head -n 1 | grep -q .
}

check_g12ec() {
    local root="${1}"
    find "${root}" -type f -name "*.hea" | head -n 1 | grep -q .
    if find "${root}" -type f -name "*.dat" | head -n 1 | grep -q .; then
        return 0
    fi
    find "${root}" -type f -name "*.mat" | head -n 1 | grep -q .
}

deploy_dataset() {
    local dataset_name="${1}"
    local target_dir="${2}"
    local url="${3}"
    local archive="${4}"
    local source_dir="${5}"
    local check_fn="${6}"

    if [[ "${FORCE}" -eq 0 && -d "${target_dir}" ]] && "${check_fn}" "${target_dir}"; then
        echo "[skip] ${dataset_name}: already prepared at ${target_dir}"
        return 0
    fi

    rm -rf "${target_dir}"
    mkdir -p "${target_dir}"

    if [[ -n "${source_dir}" ]]; then
        echo "[info] ${dataset_name}: copying from source dir ${source_dir}"
        copy_source_dir "${source_dir}" "${target_dir}"
    elif [[ -n "${archive}" ]]; then
        if [[ ! -f "${archive}" ]]; then
            echo "ERROR: archive not found for ${dataset_name}: ${archive}" >&2
            return 1
        fi
        local extract_dir
        extract_dir="$(mktemp -d)"
        echo "[info] ${dataset_name}: extracting local archive ${archive}"
        extract_archive "${archive}" "${extract_dir}"
        if [[ -d "${extract_dir}" ]]; then
            copy_source_dir "${extract_dir}" "${target_dir}"
        fi
        rm -rf "${extract_dir}"
    elif [[ -n "${url}" ]]; then
        TMP_DIR=""
        echo "[info] ${dataset_name}: downloading from ${url}"
        if ! download_from_url "${dataset_name}" "${url}" "${target_dir}"; then
            return 1
        fi
        if (( KEEP_TEMP == 0 )); then
            if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
                rm -rf "${TMP_DIR}"
            fi
            TMP_DIR=""
        fi
    else
        echo "ERROR: no source specified for ${dataset_name}. Provide --${dataset_name,,}-url, --${dataset_name,,}-archive, or --${dataset_name,,}-source-dir." >&2
        return 1
    fi

    if ! "${check_fn}" "${target_dir}"; then
        echo "ERROR: ${dataset_name} layout check failed at ${target_dir}" >&2
        echo "  expected minimum files/directories are listed in README.md under 'Raw data placement for reproducibility'." >&2
        return 1
    fi

    echo "[done] ${dataset_name}: prepared at ${target_dir}"
}

while [[ $# -gt 0 ]]; do
    case "${1}" in
        --target-dir)
            TARGET_DIR="${2}"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --keep-temp)
            KEEP_TEMP=1
            shift
            ;;
        --ptbxl-url)
            PTBXL_URL="${2}"
            shift 2
            ;;
        --ptbxl-archive)
            PTBXL_ARCHIVE="${2}"
            shift 2
            ;;
        --ptbxl-source-dir)
            PTBXL_SOURCE="${2}"
            shift 2
            ;;
        --cpsc-url)
            CPSC_URL="${2}"
            shift 2
            ;;
        --cpsc-archive)
            CPSC_ARCHIVE="${2}"
            shift 2
            ;;
        --cpsc-source-dir)
            CPSC_SOURCE="${2}"
            shift 2
            ;;
        --g12ec-url)
            G12EC_URL="${2}"
            shift 2
            ;;
        --g12ec-archive)
            G12EC_ARCHIVE="${2}"
            shift 2
            ;;
        --g12ec-source-dir)
            G12EC_SOURCE="${2}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: ${1}" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${PTBXL_URL}" ]]; then
    PTBXL_URL="${PTBXL_URL_DEFAULT}"
fi
if [[ -z "${CPSC_URL}" ]]; then
    CPSC_URL="${CPSC_URL_DEFAULT}"
fi
if [[ -z "${G12EC_URL}" ]]; then
    G12EC_URL="${G12EC_URL_DEFAULT}"
fi

mkdir -p "${TARGET_DIR}"

PTBXL_TARGET_DIR="${TARGET_DIR}/PTBXL/1.0.1/ptbxl"
CPSC_TARGET_DIR="${TARGET_DIR}/CPSC2018"
G12EC_TARGET_DIR="${TARGET_DIR}/G12EC/WFDB_v230901"

echo "[start] target raw root: ${TARGET_DIR}"

deploy_dataset "PTBXL" "${PTBXL_TARGET_DIR}" "${PTBXL_URL}" "${PTBXL_ARCHIVE}" "${PTBXL_SOURCE}" check_ptbxl
deploy_dataset "CPSC" "${CPSC_TARGET_DIR}" "${CPSC_URL}" "${CPSC_ARCHIVE}" "${CPSC_SOURCE}" check_cpsc
deploy_dataset "G12EC" "${G12EC_TARGET_DIR}" "${G12EC_URL}" "${G12EC_ARCHIVE}" "${G12EC_SOURCE}" check_g12ec

echo "[done] Raw data folders:"
echo " - ${PTBXL_TARGET_DIR}"
echo " - ${CPSC_TARGET_DIR}"
echo " - ${G12EC_TARGET_DIR}"
echo "Please align config.yaml paths with these locations (or use raw_data-relative defaults if already matching)."

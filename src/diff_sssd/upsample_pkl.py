import os
import glob
import pickle
from typing import List, Sequence, Union

import numpy as np
from scipy.signal import resample_poly

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PKL_ROOT_DEFAULT = os.path.join(
    REPO_ROOT,
    "outputs",
    "experiment",
    "v260327",
    "diff-gen00s",
    "gen0005",
    "20260331-100431-syn_ecg-DGXH100",
)


def load_pkl(path: str) -> np.ndarray:
    """pklファイルからnumpy配列を読み込む。"""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return np.asarray(obj)


def save_pkl(obj, path: str) -> None:
    """オブジェクトをpkl形式で保存する。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def upsample_ecg_array(
    arr: np.ndarray,
    orig_fs: int = 100,
    target_fs: int = 500,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    ECG配列 (N, 8, T) を orig_fs -> target_fs にアップサンプリングする。

    Parameters
    ----------
    arr : np.ndarray
        shape = (N, 8, T)
    orig_fs : int
        元のサンプリング周波数
    target_fs : int
        変換後のサンプリング周波数
    dtype : np.dtype
        出力dtype

    Returns
    -------
    np.ndarray
        shape = (N, 1, T * target_fs / orig_fs)
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array (N, 8, T), but got shape={arr.shape}")
    if arr.shape[1] != 8:
        raise ValueError(f"Expected 8 channels, but got shape={arr.shape}")

    n, c, t = arr.shape
    if t != 1000:
        print(f"[Warning] Expected T=1000 (10 sec @ 100Hz), but got T={t}")

    # 時間軸(axis=2)をアップサンプリング
    # 100Hz -> 500Hz なので up=5, down=1
    out = resample_poly(arr, up=target_fs, down=orig_fs, axis=2)

    # Extract lead-II
    out = out[:, 1:2, :]  # shape=(N, 1, T_new)

    return out.astype(dtype, copy=False)


# def make_output_path(
#     input_path: str,
#     output_dir: str = None,
#     suffix: str = "_500hz",
# ) -> str:
#     """
#     入力ファイル名から出力パスを生成する。
#     例: sample.pkl -> sample_500hz.pkl
#     """
#     base_dir = os.path.dirname(input_path) if output_dir is None else output_dir
#     filename = os.path.basename(input_path)
#     stem, ext = os.path.splitext(filename)
#     return os.path.join(base_dir, f"{stem}{suffix}{ext}")


def process_one_file(
    input_path: str,
    output_path: str = None,
    orig_fs: int = 100,
    target_fs: int = 500,
    output_dtype: np.dtype = np.float32,
) -> str:
    """
    1つのpklファイルを読み込み、アップサンプリングして保存する。

    Returns
    -------
    str
        保存先パス
    """
    if output_path is None:
        output_path = make_output_path(input_path)

    arr = load_pkl(input_path)
    print(f"Loaded : {input_path}")
    print(f"  shape={arr.shape}, dtype={arr.dtype}")

    upsampled = upsample_ecg_array(
        arr,
        orig_fs=orig_fs,
        target_fs=target_fs,
        dtype=output_dtype,
    )

    print(f"Upsampled:")
    print(f"  shape={upsampled.shape}, dtype={upsampled.dtype}")

    save_pkl(upsampled, output_path)
    print(f"Saved  : {output_path}\n")

    return output_path


def collect_pkl_paths(pattern_or_list: Union[str, Sequence[str]]) -> List[str]:
    """globパターンまたはファイルリストから入力pkl一覧を得る。"""
    if isinstance(pattern_or_list, str):
        paths = sorted(glob.glob(pattern_or_list))
    else:
        paths = list(pattern_or_list)

    if not paths:
        raise FileNotFoundError("No pickle files found.")

    return paths


def process_multiple_files(
    input_paths: Sequence[str],
    output_dir: str = None,
    suffix: str = "_500hz",
    orig_fs: int = 100,
    target_fs: int = 500,
    output_dtype: np.dtype = np.float32,
) -> List[str]:
    """
    複数のpklファイルを個別にアップサンプリングして保存する。
    mergeはしない。
    """
    saved_paths = []

    for input_path in input_paths:
        output_path = input_path.replace(
            "/samples/", "/samples_500hz/")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        saved_path = process_one_file(
            input_path=input_path,
            output_path=output_path,
            orig_fs=orig_fs,
            target_fs=target_fs,
            output_dtype=output_dtype,
        )
        saved_paths.append(saved_path)

    return saved_paths


def main(data_split, pkl_root):    

    pkl_regex = os.path.join(
        pkl_root, data_split, "samples/*.pkl")
    input_paths = collect_pkl_paths(pkl_regex)

    process_multiple_files(
        input_paths=input_paths,
        suffix="_500hz",
        orig_fs=100,
        target_fs=500,
        output_dtype=np.float32,
    )

if __name__ == "__main__":

    # pkl_root = os.path.join(
    #     REPO_ROOT,
    #     "outputs",
    #     "experiment",
    #     "v260327",
    #     "diff-gen00s",
    #     "gen0002",
    #     "20260327-170822-syn_ecg-DGXH100",
    # )
    # pkl_root = os.path.join(
    #     REPO_ROOT,
    #     "outputs",
    #     "experiment",
    #     "v260327",
    #     "diff-gen00s",
    #     "gen0003",
    #     "20260327-211331-syn_ecg-DGXH100",
    # )
    # pkl_root = os.path.join(
    #     REPO_ROOT,
    #     "outputs",
    #     "experiment",
    #     "v260327",
    #     "diff-gen00s",
    #     "gen0004",
    #     "20260328-005747-syn_ecg-DGXH100",
    # )
    # main("val", pkl_root)
    # main("train", pkl_root)
    pkl_root = PKL_ROOT_DEFAULT
    main("val", pkl_root)
    main("train", pkl_root)    

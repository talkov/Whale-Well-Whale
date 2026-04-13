# to run this do this with your wanted parameters:
# python scripts/preprocess.py \
#   --raw_data_dir data/raw/train2 \
#   --output_dir data/processed/logmel_fmax1000_224 \
#   --n_fft 512 \
#   --hop_length 128 \
#   --n_mels 128 \
#   --fmin 0 \
#   --fmax 1000 \
#   --target_size 224 \
#   --min_db -80 \
#   --save_relative_paths




import argparse
from pathlib import Path

import cv2
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess .aif whale-call audio files into log-mel .npy features and metadata CSV."
    )

    # Paths
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to input directory containing .aif files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--audio_extension",
        type=str,
        default=".aif",
        help="Audio file extension to search for, default: .aif",
    )

    # Spectrogram parameters
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--fmin", type=float, default=0.0)
    parser.add_argument("--fmax", type=float, default=1000.0)
    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--min_db", type=float, default=-80.0)

    # Metadata behavior
    parser.add_argument(
        "--save_relative_paths",
        action="store_true",
        help="If set, save paths in metadata.csv relative to output_dir instead of absolute paths",
    )

    return parser.parse_args()


def load_audio(file_path: Path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr


def normalize_waveform(y: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)


def compute_log_mel(
    y: np.ndarray,
    sr: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


def normalize_spectrogram(mel_db: np.ndarray, min_db: float) -> np.ndarray:
    mel_db = np.clip(mel_db, min_db, 0.0)
    mel_norm = (mel_db - min_db) / (0.0 - min_db)
    return mel_norm.astype(np.float32)


def resize_spectrogram(mel_norm: np.ndarray, target_size: int) -> np.ndarray:
    resized = cv2.resize(
        mel_norm,
        (target_size, target_size),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized.astype(np.float32)


def process_file(file_path: Path, args) -> np.ndarray:
    y, sr = load_audio(file_path)
    y = normalize_waveform(y)

    mel_db = compute_log_mel(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
    )

    mel_norm = normalize_spectrogram(mel_db, min_db=args.min_db)
    mel_resized = resize_spectrogram(mel_norm, target_size=args.target_size)

    return mel_resized


def get_label(file_name: str):
    if file_name.endswith("_1.aif"):
        return 1
    if file_name.endswith("_0.aif"):
        return 0
    return None


def maybe_make_relative(path: Path, base_dir: Path, use_relative: bool) -> str:
    if use_relative:
        return str(path.relative_to(base_dir))
    return str(path.resolve())


def main():
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    metadata_path = output_dir / "metadata.csv"

    print("RAW_DATA_DIR exists:", raw_data_dir.exists())
    print("RAW_DATA_DIR:", raw_data_dir)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {raw_data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"*{args.audio_extension.lstrip('.')}"
    if not args.audio_extension.startswith("."):
        pattern = f"*.{args.audio_extension}"

    files = sorted(raw_data_dir.rglob(pattern))

    print("\nTotal audio files found:", len(files))

    whale = []
    noise = []
    test = []

    for f in files:
        if f.name.endswith("_1.aif"):
            whale.append(f)
        elif f.name.endswith("_0.aif"):
            noise.append(f)
        else:
            test.append(f)

    print("Whale files:", len(whale))
    print("Noise files:", len(noise))
    print("Test files:", len(test))

    if len(files) == 0:
        raise FileNotFoundError("No audio files found. Check directory structure and file extension.")

    print("\nExample files:")
    for f in files[:5]:
        print(f)

    metadata = []

    print("\nStarting preprocessing...")

    for file_path in tqdm(files):
        label = get_label(file_path.name)
        feature = process_file(file_path, args)

        feature_name = file_path.stem + ".npy"
        feature_path = feature_dir / feature_name

        np.save(feature_path, feature)

        metadata.append(
            {
                "sample_id": file_path.stem,
                "audio_path": maybe_make_relative(
                    file_path, output_dir, args.save_relative_paths
                ),
                "feature_path": maybe_make_relative(
                    feature_path, output_dir, args.save_relative_paths
                ),
                "label": label if label is not None else -1,
            }
        )

    df = pd.DataFrame(metadata)
    df.to_csv(metadata_path, index=False)

    print("\nPreprocessing complete")
    print("Features saved to:", feature_dir)
    print("Metadata saved to:", metadata_path)


if __name__ == "__main__":
    main()

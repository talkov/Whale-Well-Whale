import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Apply bandpass filtering to audio files and save filtered audio.")
    parser.add_argument("--raw_data_dir", type=str, required=True, help="Path to directory containing input .aif files")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory where filtered audio and metadata will be saved")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Target sample rate for loading audio")
    parser.add_argument("--max_audio_seconds", type=float, default=2.0, help="Maximum audio length in seconds")
    parser.add_argument("--apply_bandpass", action="store_true", help="Apply Butterworth bandpass filter")
    parser.add_argument("--lowcut", type=float, default=50.0, help="Low cutoff frequency for bandpass filter")
    parser.add_argument("--highcut", type=float, default=350.0, help="High cutoff frequency for bandpass filter")
    parser.add_argument("--filter_order", type=int, default=5, help="Order of Butterworth bandpass filter")
    parser.add_argument("--debug_num_files", type=int, default=0, help="If > 0, process only the first N files")
    return parser.parse_args()


def extract_label_from_filename(file_path: Path):
    stem = file_path.stem
    last_part = stem.split("_")[-1]
    if last_part not in {"0", "1"}:
        return None
    return int(last_part)


def bandpass_filter(wav: np.ndarray, sr: int, lowcut: float, highcut: float, order: int = 5) -> np.ndarray:
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist

    if not (0.0 < low < high < 1.0):
        raise ValueError(f"Invalid bandpass range: lowcut={lowcut}, highcut={highcut}, sr={sr}")

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, wav)
    return filtered.astype(np.float32)


def load_audio(path: str, sample_rate: int, max_audio_seconds: float) -> np.ndarray:
    wav, sr = librosa.load(path, sr=sample_rate, mono=True)
    wav = wav.astype(np.float32)

    max_audio_len = int(sample_rate * max_audio_seconds)
    if len(wav) > max_audio_len:
        wav = wav[:max_audio_len]
    elif len(wav) < max_audio_len:
        wav = np.pad(wav, (0, max_audio_len - len(wav)), mode="constant")

    return wav


def preprocess_audio(
    path: str,
    sample_rate: int,
    max_audio_seconds: float,
    apply_bandpass: bool,
    lowcut: float,
    highcut: float,
    filter_order: int,
) -> np.ndarray:
    wav = load_audio(
        path=path,
        sample_rate=sample_rate,
        max_audio_seconds=max_audio_seconds,
    )

    if apply_bandpass:
        wav = bandpass_filter(
            wav=wav,
            sr=sample_rate,
            lowcut=lowcut,
            highcut=highcut,
            order=filter_order,
        )

    if not np.isfinite(wav).all():
        raise ValueError(f"Non-finite waveform after preprocessing: {path}")

    return wav


def main():
    args = parse_args()

    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    filtered_audio_dir = output_dir / "filtered_audio"
    metadata_path = output_dir / "metadata.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_audio_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_data_dir.rglob("*.aif"))
    if args.debug_num_files > 0:
        files = files[:args.debug_num_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No .aif files found under: {raw_data_dir}")

    rows = []

    for file_path in tqdm(files, desc="Filtering audio"):
        label = extract_label_from_filename(file_path)
        if label is None:
            continue

        wav = preprocess_audio(
            path=str(file_path),
            sample_rate=args.sample_rate,
            max_audio_seconds=args.max_audio_seconds,
            apply_bandpass=args.apply_bandpass,
            lowcut=args.lowcut,
            highcut=args.highcut,
            filter_order=args.filter_order,
        )

        filtered_audio_path = filtered_audio_dir / f"{file_path.stem}.aif"
        sf.write(filtered_audio_path, wav, args.sample_rate, format="AIFF")

        rows.append(
            {
                "sample_id": file_path.stem,
                "label": int(label),
                "audio_path": str(file_path),
                "filtered_audio_path": str(filtered_audio_path),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No labeled files processed.")

    if df["sample_id"].duplicated().any():
        dupes = df[df["sample_id"].duplicated(keep=False)]["sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate sample_id values found. Examples: {dupes}")

    df.to_csv(metadata_path, index=False)

    print("\nDone.")
    print(f"Metadata CSV: {metadata_path}")
    print(f"Filtered audio dir: {filtered_audio_dir}")


if __name__ == "__main__":
    main()

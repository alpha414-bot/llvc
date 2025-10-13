# validate_llvc_dataset.py
# Usage:
#   python validate_llvc_dataset.py /path/to/dataset --sr 16000
#
# The dataset directory should contain train/, val/, dev/ subfolders.
# Each subfolder should contain files named PREFIX_original.wav and PREFIX_converted.wav.

import os
import argparse
from collections import defaultdict
import math

# prefer soundfile, fallback to scipy
try:
    import soundfile as sf

    def read_wav_info(path):
        data, sr = sf.read(path, always_2d=True)
        channels = data.shape[1]
        duration = data.shape[0] / sr
        return sr, channels, duration

except Exception:
    from scipy.io import wavfile

    def read_wav_info(path):
        sr, data = wavfile.read(path)
        if data.ndim == 1:
            channels = 1
        else:
            channels = data.shape[1] if len(data.shape) > 1 else 0
        duration = (data.shape[0] if len(data.shape) > 0 else 0) / sr
        return sr, channels, duration


def find_pairs(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".wav")])
    pairs = {}
    originals = {}
    for f in files:
        if f.endswith("_original.wav"):
            prefix = f[: -len("_original.wav")]
            originals[prefix] = os.path.join(folder, f)
        elif f.endswith("_converted.wav"):
            prefix = f[: -len("_converted.wav")]
            pairs.setdefault(prefix, {})["converted"] = os.path.join(folder, f)
    # merge
    result = {}
    for prefix, orig_path in originals.items():
        conv_path = pairs.get(prefix, {}).get("converted", None)
        result[prefix] = (orig_path, conv_path)
    # also include converted-only prefixes (warn later)
    for prefix, kv in pairs.items():
        if prefix not in result:
            result[prefix] = (None, kv.get("converted"))
    return result


def infer_speaker_from_prefix(prefix):
    # Reasonable default: speaker id is first token before underscore
    if "_" in prefix:
        return prefix.split("_", 1)[0]
    elif "-" in prefix:
        return prefix.split("-", 1)[0]
    else:
        # fallback: whole prefix
        return prefix


def check_folder(path, expected_sr, min_seconds=0.3, duration_tol=0.3):
    missing = []
    bad_sr = []
    bad_channels = []
    short_files = []
    duration_mismatch = []
    pairs = find_pairs(path)
    speakers = set()
    for prefix, (orig, conv) in pairs.items():
        if orig is None or conv is None:
            missing.append((prefix, orig is None, conv is None))
            continue
        try:
            sr_o, ch_o, dur_o = read_wav_info(orig)
            sr_c, ch_c, dur_c = read_wav_info(conv)
        except Exception as e:
            print(f"ERROR reading {orig} or {conv}: {e}")
            continue
        if sr_o != expected_sr:
            bad_sr.append((orig, sr_o))
        if sr_c != expected_sr:
            bad_sr.append((conv, sr_c))
        if ch_o != 1:
            bad_channels.append((orig, ch_o))
        if ch_c != 1:
            bad_channels.append((conv, ch_c))
        if dur_o < min_seconds or dur_c < min_seconds:
            short_files.append((prefix, dur_o, dur_c))
        if abs(dur_o - dur_c) > max(duration_tol, duration_tol * max(dur_o, dur_c)):
            duration_mismatch.append((prefix, dur_o, dur_c))
        speakers.add(infer_speaker_from_prefix(prefix))
    return {
        "pairs_count": len(pairs),
        "missing_pairs": missing,
        "bad_sr": bad_sr,
        "bad_channels": bad_channels,
        "short_files": short_files,
        "duration_mismatch": duration_mismatch,
        "speakers": speakers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_dir", help="Path to dataset dir containing train/ val/ dev/"
    )
    parser.add_argument(
        "--sr", type=int, default=16000, help="Expected sample rate (default 16000)"
    )
    parser.add_argument(
        "--min_seconds",
        type=float,
        default=0.3,
        help="Minimum clip duration in seconds",
    )
    parser.add_argument(
        "--dur_tol",
        type=float,
        default=0.3,
        help="Duration mismatch tolerance (absolute fraction)",
    )
    args = parser.parse_args()

    for sub in ["train", "val", "dev"]:
        path = os.path.join(args.dataset_dir, sub)
        if not os.path.isdir(path):
            print(f"Missing folder: {path}")
            return
    print("Checking folders...")

    results = {}
    for sub in ["train", "val", "dev"]:
        print(f"\n=== Checking {sub} ===")
        res = check_folder(
            os.path.join(args.dataset_dir, sub),
            expected_sr=args.sr,
            min_seconds=args.min_seconds,
            duration_tol=args.dur_tol,
        )
        results[sub] = res
        print(f"{sub}: {res['pairs_count']} prefixes found")
        if res["missing_pairs"]:
            print(
                f"  Missing pair files (prefix, orig_missing, conv_missing): {res['missing_pairs'][:10]}"
            )
        if res["bad_sr"]:
            print(f"  Files with unexpected sample rate: {res['bad_sr'][:10]}")
        if res["bad_channels"]:
            print(f"  Files with non-mono channels: {res['bad_channels'][:10]}")
        if res["short_files"]:
            print(f"  Short files (<{args.min_seconds}s): {res['short_files'][:10]}")
        if res["duration_mismatch"]:
            print(
                f"  Duration mismatches (orig, converted): {res['duration_mismatch'][:10]}"
            )

    # Speaker overlap checks
    train_spk = results["train"]["speakers"]
    val_spk = results["val"]["speakers"]
    dev_spk = results["dev"]["speakers"]

    print("\n=== Speaker sets summary ===")
    print(f"train speakers: {len(train_spk)}")
    print(f"val speakers:   {len(val_spk)}")
    print(f"dev speakers:   {len(dev_spk)}")

    # LLVC expectation: val contains clips from same speakers as test.
    # We don't have test here, but at least check val != dev
    inter_val_dev = val_spk & dev_spk
    if inter_val_dev:
        print(
            f"WARNING: val and dev share speakers ({len(inter_val_dev)}). LLVC recommends dev speakers should be different from test and val."
        )
    else:
        print("OK: val and dev appear disjoint in speaker sets.")

    # Check train/test leakage heuristics:
    if train_spk & val_spk:
        print(
            "INFO: Some speakers appear in both train and val (this is common if speakers have multiple clips). Make sure test evaluation remains held out."
        )
    if train_spk & dev_spk:
        print("INFO: Some speakers appear in both train and dev.")

    print("\nValidation complete. Fix warnings before training for best results.")


if __name__ == "__main__":
    main()

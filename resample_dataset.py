import soundfile as sf
import os
import librosa

if __name__ == "__main__":
    # Load an audio file with a target sample rate of 8000 Hz
    directory = os.path.join(os.getcwd(), "test_wavs")
    all_files = [
        f for f in os.listdir(directory) if f.lower().endswith((".wav", ".mp3"))
    ]
    print(f"Found {len(all_files)} audio files in {directory}")
    datasets_dir = os.path.normpath(
        os.path.join(f"{directory}", "test_wavs_resample")
    )
    os.makedirs(datasets_dir, exist_ok=True)

    for index, file in enumerate(all_files):
        src_path = os.path.join(directory, file)
        # load and resample to 16000 Hz
        # y, sr = librosa.load(src_path, sr=16000)
        y, sr = librosa.load(src_path, sr=48000)
        print(f"File: {file}, Loaded SR: {sr}, New SR: 16000, Samples: {len(y)}")

        name, ext = os.path.splitext(file)
        # parts = name.split("_", 1)
        # if len(parts) == 2 and parts[0].isdigit():
        #     number = int(parts[0])
        #     rest = parts[1]
        # else:
        #     number = index + 1
        #     rest = name

        # out_name = f"Speaker{number}_{rest.lower()}{number}_converted.wav"
        out_path = os.path.join(datasets_dir, f"{name}.wav")
        sf.write(out_path, y, samplerate=48000, subtype="PCM_16")
        print(f"Wrote: {out_path}")

        # # If original was mp3, delete it after conversion
        # if ext.lower() == ".mp3":
        #     try:
        #         os.remove(src_path)
        #         print(f"Deleted original mp3: {src_path}")
        #     except OSError as e:
        #         print(f"Failed to delete {src_path}: {e}")

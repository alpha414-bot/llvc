import soundfile as sf
import os
import librosa

if __name__ == "__main__":
    # Load an audio file with a target sample rate of 8000 Hz
    directory = os.path.join(os.getcwd(), "../custom", "source")
    all_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    # print(all_files)
    for file in all_files:
        index = all_files.index(file)
        y, sr = librosa.load(
            os.path.join(directory, file), sr=16000
        )  # Downsample 44.1kHz to 8kHz
        print(f"File: {file}, Original SR: {sr}, New SR: 16000, Samples: {len(y)}")
        parts = file.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            number = int(parts[0])  # e.g. "12" from "12_might.wave"
            rest = parts[1].replace(".wav", "")  # e.g. "might"
        else:
            number = index + 1
            rest = file

        out_path = os.path.join(
            directory,
            "../datasets",
            f"Speaker{number}_{rest.lower()}{number}_original.wav",
        )
        sf.write(out_path, y, samplerate=16000)
        # src_path = os.path.join(directory, file)
        # if os.path.exists(src_path) and os.path.isfile(src_path):
        #     try:
        #         os.remove(src_path)
        #         print(f"Removed old file: {src_path}")
        #     except OSError as e:
        #         print(f"Failed to remove {src_path}: {e}")

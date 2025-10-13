import sys, os

sys.path.append(os.path.dirname(__file__))
from model import Net
import torch
import torchaudio
import time
import numpy as np
import argparse
import json
import os
from utils import glob_audio_files
from tqdm import tqdm
import soundfile as sf
import io

# Ensure higher-quality resampling (use sox_io if available)
try:
    torchaudio.set_audio_backend("sox_io")
except Exception:
    pass

def load_model(checkpoint_path, config_path, device=None):
    with open(config_path) as f:
        config = json.load(f)
    model = Net(**config["model_params"])
    state = torch.load(checkpoint_path, map_location="cpu")
    # support checkpoints that either store dict with "model" key or raw state_dict
    model_state = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    fp16 = config.get("fp16_run", False) if "fp16_run" in config else False 
    if fp16 and device.type == "cuda":
        model.half()
    return model, config["data"]["sr"]


def load_audio(audio_path, sample_rate):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(0, keepdim=False)
    audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    return audio


def load_audio_full(source, sample_rate):
    """
    Load audio from a file path, bytes/bytearray, or file-like object.
    Returns a mono 1D float32 torch.Tensor sampled at sample_rate.
    """
    # read into numpy + sample rate for bytes/file-like, otherwise use torchaudio for paths
    if isinstance(source, (bytes, bytearray)):
        # data, sr = sf.read(io.BytesIO(source), dtype="float32")
        data, sr = sf.read(io.BytesIO(source))
        # soundfile returns (frames,) or (frames, channels)
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)  # [1, frames]
        else:
            waveform = torch.from_numpy(data.T)  # [channels, frames]
    elif hasattr(source, "read"):
        data, sr = sf.read(source, dtype="float32")
        if data.ndim == 1:
            waveform = torch.from_numpy(data).unsqueeze(0)
        else:
            waveform = torch.from_numpy(data.T)
    else:
        waveform, sr = torchaudio.load(source)  # [channels, frames]
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)

    # ensure [channels, frames]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # resample if needed (sox_io backend used above for better quality if available)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    waveform = waveform.to(torch.float32)
    return waveform.squeeze(0)  # 1D tensor [frames]


def save_audio(audio, audio_path, sample_rate):
    torchaudio.save(audio_path, audio, sample_rate)


def infer(model, audio):
    return model(audio.unsqueeze(0).unsqueeze(0)).squeeze(0)


def infer_stream(model, audio, chunk_factor, sr):
    L = model.L
    chunk_len = model.dec_chunk_size * L * chunk_factor
    # pad audio to be a multiple of L * dec_chunk_size
    original_len = len(audio)
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = torch.nn.functional.pad(audio, (0, pad_len))

    # scoot audio down by L
    audio = torch.cat((audio[L:], torch.zeros(L)))
    audio_chunks = torch.split(audio, chunk_len)
    # add lookahead context from prev chunk
    new_audio_chunks = []
    for i, a in enumerate(audio_chunks):
        if i == 0:
            front_ctx = torch.zeros(L * 2)
        else:
            front_ctx = audio_chunks[i - 1][-L * 2 :]
        new_audio_chunks.append(torch.cat([front_ctx, a]))
    audio_chunks = new_audio_chunks

    outputs = []
    times = []
    with torch.inference_mode():
        enc_buf, dec_buf, out_buf = model.init_buffers(1, torch.device("cpu"))
        if hasattr(model, "convnet_pre"):
            convnet_pre_ctx = model.convnet_pre.init_ctx_buf(1, torch.device("cpu"))
        else:
            convnet_pre_ctx = None
        for chunk in audio_chunks:
            # Process each audio chunk
            start = time.time()
            output, enc_buf, dec_buf, out_buf, convnet_pre_ctx = model(
                chunk.unsqueeze(0).unsqueeze(0),
                enc_buf,
                dec_buf,
                out_buf,
                convnet_pre_ctx,
                pad=(not model.lookahead),
            )
            outputs.append(output)
            times.append(time.time() - start)
        # concatenate outputs
    outputs = torch.cat(outputs, dim=2)
    # Calculate RTF
    avg_time = np.mean(times)
    rtf = (chunk_len / sr) / avg_time
    # calculate e2e latency
    e2e_latency = ((2 * L + chunk_len) / sr + avg_time) * 1000
    # remove padding
    outputs = outputs[:, :, :original_len].squeeze(0)
    return outputs, rtf, e2e_latency


def do_infer(model, audio, chunk_factor, sr, stream):
    with torch.no_grad():
        if stream:
            outputs, rtf, e2e_latency = infer_stream(model, audio, chunk_factor, sr)
            return outputs, rtf, e2e_latency
        else:
            outputs = infer(model, audio)
            rtf = None
            e2e_latency = None
    return outputs, rtf, e2e_latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        "-p",
        type=str,
        default="llvc_models/models/checkpoints/llvc/G_500000.pth",
        help="Path to LLVC checkpoint file",
    )
    parser.add_argument(
        "--config_path",
        "-c",
        type=str,
        default="experiments/llvc/config.json",
        help="Path to LLVC config file",
    )
    parser.add_argument(
        "--fname",
        "-f",
        type=str,
        default="test_wavs",
        help="Path to audio file or directory of audio files to convert",
    )
    parser.add_argument(
        "--out_dir",
        "-o",
        type=str,
        default="converted_out",
        help="Directory to save converted audio",
    )
    parser.add_argument(
        "--chunk_factor",
        "-n",
        type=int,
        default=1,
        help="Chunk factor for streaming inference",
    )
    parser.add_argument(
        "--stream", "-s", action="store_true", help="Use streaming inference"
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sr = load_model(args.checkpoint_path, args.config_path, device=device)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    # check if fname is a directory
    if os.path.isdir(args.fname):
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        # recursively glob wav files
        rtf_list = []
        fnames = glob_audio_files(args.fname)
        e2e_times_list = []
        for fname in tqdm(fnames):
            audio = load_audio_full(fname, sr)
            out, rtf_, e2e_latency_ = do_infer(
                model, audio, args.chunk_factor, sr, args.stream
            )
            rtf_list.append(rtf_)
            e2e_times_list.append(e2e_latency_)
            out_fname = os.path.join(args.out_dir, os.path.basename(fname))
            save_audio(out, out_fname, sr)
        rtf = np.mean(rtf_list) if rtf_list[0] is not None else None
        e2e_latency = np.mean(e2e_times_list) if e2e_times_list[0] is not None else None
    else:
        audio = load_audio_full(args.fname, sr)
        out, rtf, e2e_latency = do_infer(
            model, audio, args.chunk_factor, sr, args.stream
        )
        out_fname = os.path.join(args.out_dir, os.path.basename(args.fname))
        save_audio(out, out_fname, sr)
    print(
        f"Saved outputs to {args.out_dir}, loaded audio using load_audio_full() with stream {args.stream}"
    )
    # if rtf is not None and e2e_latency is not None:
    print(f"RTF: {rtf:.3f}") if rtf is not None else None
    (
        print(f"End-to-end latency: {e2e_latency:.3f}ms")
        if e2e_latency is not None
        else None
    )


if __name__ == "__main__":
    main()

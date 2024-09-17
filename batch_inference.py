"""Inference multiple files in batch."""

import argparse
import os
import time
import warnings

import librosa
import loguru
import torch
import torchaudio
import yaml

from hf_utils import load_custom_model_from_hf
from modules.audio import mel_spectrogram
from modules.campplus.DTDNN import CAMPPlus
from modules.commons import build_model
from modules.commons import load_checkpoint
from modules.commons import recursive_munch
from modules.cosyvoice_tokenizer.frontend import CosyVoiceFrontEnd
from modules.hifigan.f0_predictor import ConvRNNF0Predictor
from modules.hifigan.generator import HiFTGenerator

warnings.simplefilter("ignore")

REP_ID_SEEDVC = "Plachta/Seed-VC"
REP_ID_FACODEC = "Plachta/FAcodec"
REP_ID_CAMPPLUS = "funasr/campplus"
CFG_ID_SEEDVC = "config_dit_mel_seed_facodec_small_wavenet.yml"
CFG_ID_HIFIGAN = "hifigan.yml"
CFG_ID_CAMPPLUS = None
CFG_ID_COSYVOICE = None
CFG_ID_FACODEC = "config.yml"
CKPT_ID_SEEDVC = "DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth"
CKPT_ID_HIFIGAN = "hift.pt"
CKPT_ID_CAMPPLUS = "campplus_cn_common.bin"
CKPT_ID_COSYVOICE = "speech_tokenizer_v1.onnx"
CKPT_ID_FACODEC = "pytorch_model.bin"


def load_yaml(cfg_path: str) -> dict:
    """Load configuration from a yaml file."""
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# Load model and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dit_ckpt_path, dit_cfg_path = load_custom_model_from_hf(
    REP_ID_SEEDVC,
    CKPT_ID_SEEDVC,
    CFG_ID_SEEDVC,
)

config = load_yaml(dit_cfg_path)
model_params = recursive_munch(config["model_params"])
model = build_model(model_params, stage="DiT")
hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
sr = config["preprocess_params"]["sr"]

# Load checkpoints
model, _, _, _ = load_checkpoint(
    model,
    None,
    dit_ckpt_path,
    load_only_params=True,
    ignore_modules=[],
    is_distributed=False,
)
for key in model:
    model[key].eval()
    model[key].to(device)
model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

campplus_ckpt_path = load_custom_model_from_hf(
    REP_ID_CAMPPLUS,
    CKPT_ID_CAMPPLUS,
    config_filename=CFG_ID_CAMPPLUS,
)
campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
campplus_model.load_state_dict(
    torch.load(campplus_ckpt_path, map_location="cpu"))
campplus_model.eval()
campplus_model.to(device)

hift_ckpt_path, hift_cfg_path = load_custom_model_from_hf(
    REP_ID_SEEDVC, CKPT_ID_HIFIGAN, CFG_ID_HIFIGAN)
hift_config = load_yaml(hift_cfg_path)
hift_gen = HiFTGenerator(
    **hift_config["hift"],
    f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]),
)
hift_gen.load_state_dict(torch.load(hift_ckpt_path, map_location="cpu"))
hift_gen.eval()
hift_gen.to(device)

speech_tokenizer_type = config["model_params"]["speech_tokenizer"].get(
    "type", "cosyvoice")
if speech_tokenizer_type == "cosyvoice":
    speech_tokenizer_path = load_custom_model_from_hf(REP_ID_SEEDVC,
                                                      CKPT_ID_COSYVOICE,
                                                      CFG_ID_COSYVOICE)
    cosyvoice_frontend = CosyVoiceFrontEnd(
        speech_tokenizer_model=speech_tokenizer_path,
        device="cuda",
        device_id=0)
elif speech_tokenizer_type == "facodec":
    ckpt_path, config_path = load_custom_model_from_hf(
        REP_ID_FACODEC,
        CKPT_ID_FACODEC,
        CFG_ID_FACODEC,
    )

    codec_config = load_yaml(config_path)
    codec_model_params = recursive_munch(codec_config["model_params"])
    codec_encoder = build_model(codec_model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in codec_encoder:
        codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
    _ = [codec_encoder[key].eval() for key in codec_encoder]
    _ = [codec_encoder[key].to(device) for key in codec_encoder]
# Generate mel spectrograms
mel_fn_args = {
    "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
    "win_size": config["preprocess_params"]["spect_params"]["win_length"],
    "hop_size": config["preprocess_params"]["spect_params"]["hop_length"],
    "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
    "sampling_rate": sr,
    "fmin": 0,
    "fmax": 8000,
    "center": False,
}


def to_mel(x: torch.Tensor) -> torch.Tensor:
    """Convert waveform to mel spectrogram."""
    return mel_spectrogram(x, **mel_fn_args)


@torch.no_grad()
def process_one(  # noqa: PLR0913
    source: str,
    dst_dir: str,
    target_name: str,
    length_adjust: float,
    diffusion_steps: int,
    inference_cfg_rate: float,
    n_quantizers: int,
) -> None:
    """Process one audio file."""
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]
    # decoded_wav = encodec_model.decoder(encodec_latent)
    # torchaudio.save("test.wav", decoded_wav.cpu().squeeze(0), 24000)
    # crop only the first 30 seconds
    source_audio = source_audio[:sr * 30]
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)

    ref_audio = ref_audio[:(sr * 30 - source_audio.size(-1))]
    ref_audio = torch.tensor(ref_audio).unsqueeze(0).float().to(device)

    source_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    if speech_tokenizer_type == "cosyvoice":
        s_alt = cosyvoice_frontend.extract_speech_token(source_waves_16k)[0]
        s_ori = cosyvoice_frontend.extract_speech_token(ref_waves_16k)[0]
    elif speech_tokenizer_type == "facodec":
        converted_waves_24k = torchaudio.functional.resample(
            source_audio, sr, 24000)
        waves_input = converted_waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (quantized, codes) = codec_encoder.quantizer(z, waves_input)
        s_alt = torch.cat([codes[1], codes[0]], dim=1)

        # s_ori should be extracted in the same way
        waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
        waves_input = waves_24k.unsqueeze(1)
        z = codec_encoder.encoder(waves_input)
        (quantized, codes) = codec_encoder.quantizer(z, waves_input)
        s_ori = torch.cat([codes[1], codes[0]], dim=1)

    mel = to_mel(source_audio.to(device).float())
    mel2 = to_mel(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)
                                       ]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    # Length regulation
    cond = model.length_regulator(s_alt,
                                  ylens=target_lengths,
                                  n_quantizers=int(n_quantizers))[0]
    prompt_condition = model.length_regulator(
        s_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers))[0]
    cat_condition = torch.cat([prompt_condition, cond], dim=1)

    time_vc_start = time.time()
    vc_target = model.cfm.inference(
        cat_condition,
        torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
        mel2,
        style2,
        None,
        diffusion_steps,
        inference_cfg_rate=inference_cfg_rate,
    )
    vc_target = vc_target[:, :, mel2.size(-1):]

    # Convert to waveform
    vc_wave = hift_gen.inference(vc_target)

    time_vc_end = time.time()
    loguru.logger.info(
        f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

    source_name = source.split("/")[-1].split(".")[0]
    target_name = target_name.split("/")[-1].split(".")[0]
    torchaudio.save(
        os.path.join(
            dst_dir,
            f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav",
        ),
        vc_wave.cpu(),
        sr,
    )


def main(args: argparse.Namespace) -> None:
    """Main function."""
    src_dir = args.src_dir
    target_name = args.target
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    n_quantizers = args.n_quantizers
    dst_dir = args.dst_dir
    os.makedirs(dst_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        source = os.path.join(src_dir, f)
        process_one(
            source,
            dst_dir,
            target_name,
            length_adjust,
            diffusion_steps,
            inference_cfg_rate,
            n_quantizers,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, default="./examples/source")
    parser.add_argument("--dst-dir", type=str, default="./reconstructed")
    parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--n-quantizers", type=int, default=3)
    args = parser.parse_args()
    main(args)

# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.vc.vevo.vevo_utils import *


def test_timbreInfer():
        # ===== Device =====
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print("now device is : ", device)
    print("check point test_timbreInfer 1")


    # ========== function inference pipeline=============
    # def vevo_timbre(content_wav_path, reference_wav_path, output_path):
    #     print("check point vevo_timbre 1")
    #     gen_audio = inference_pipeline.inference_fm(
    #         src_wav_path=content_wav_path,
    #         timbre_ref_wav_path=reference_wav_path,
    #         flow_matching_steps=32,
    #     )
    #     print("check point vevo_timbre 2")
    #     save_audio(gen_audio, output_path=output_path)
    #     print("check point vevo_timbre 3")
    #==================================




    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )
    tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")
    print("check point test_timbreInfer 2")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )

    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")
    print("check point test_timbreInfer 3")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")
    print("check point test_timbreInfer 4")

    # ===== Inference =====
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=tokenizer_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )
    print("check point test_timbreInfer 5")

    content_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    reference_wav_path = "./models/vc/vevo/wav/english_female.wav"
    output_path = "./models/vc/vevo/wav/output_vevotimbre_arabicM_to_englishW.wav"
    print("check point test_timbreInfer 6")

    # vevo_timbre(content_wav_path, reference_wav_path, output_path)
    # ====== inference start ======
    print("check point vevo_timbre 1")
    gen_audio = inference_pipeline.inference_fm(
        src_wav_path=content_wav_path,
        timbre_ref_wav_path=reference_wav_path,
        flow_matching_steps=32,
    )
    print("check point vevo_timbre 2")
    save_audio(gen_audio, output_path=output_path)
    print("check point vevo_timbre 3")
    # ====== inference finish ======
    print("check point test_timbreInfer finished!!!")



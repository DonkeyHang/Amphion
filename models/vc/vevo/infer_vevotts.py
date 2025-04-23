# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from huggingface_hub import snapshot_download

from models.vc.vevo.vevo_utils import *


def test_ttsInfer():
    # ===== Device =====
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    print("now device is : ", device)
    print("check point test_timbreInfer 1")


    # ===== Content-Style Tokenizer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["tokenizer/vq8192/*"],
    )

    content_style_tokenizer_ckpt_path = os.path.join(local_dir, "tokenizer/vq8192")

    # ===== Autoregressive Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["contentstyle_modeling/PhoneToVq8192/*"],
    )

    ar_cfg_path = "./models/vc/vevo/config/PhoneToVq8192.json"
    ar_ckpt_path = os.path.join(local_dir, "contentstyle_modeling/PhoneToVq8192")

    # ===== Flow Matching Transformer =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vq8192ToMels/*"],
    )

    fmt_cfg_path = "./models/vc/vevo/config/Vq8192ToMels.json"
    fmt_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vq8192ToMels")

    # ===== Vocoder =====
    local_dir = snapshot_download(
        repo_id="amphion/Vevo",
        repo_type="model",
        cache_dir="./ckpts/Vevo",
        allow_patterns=["acoustic_modeling/Vocoder/*"],
    )

    vocoder_cfg_path = "./models/vc/vevo/config/Vocoder.json"
    vocoder_ckpt_path = os.path.join(local_dir, "acoustic_modeling/Vocoder")

    # ===== Inference =====
    inference_pipeline = VevoInferencePipeline(
        content_style_tokenizer_ckpt_path=content_style_tokenizer_ckpt_path,
        ar_cfg_path=ar_cfg_path,
        ar_ckpt_path=ar_ckpt_path,
        fmt_cfg_path=fmt_cfg_path,
        fmt_ckpt_path=fmt_ckpt_path,
        vocoder_cfg_path=vocoder_cfg_path,
        vocoder_ckpt_path=vocoder_ckpt_path,
        device=device,
    )

    # 恢复原始长文本
    src_text = "I don't really care what you call me. I've been a silent spectator, watching species evolve, empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture you; ignore me and you shall face the consequences."

    ref_wav_path = "./models/vc/vevo/wav/arabic_male.wav"
    ref_text = "Flip stood undecided, his ears strained to catch the slightest sound."


    # 添加调试输出：测试g2p处理
    print("\n===== 调试：测试文本到音素转换 =====")
    try:
        print("检查vocab.json是否存在:")
        import os
        vocab_path = "./models/tts/maskgct/g2p/g2p/vocab.json"
        print(f"文件存在: {os.path.exists(vocab_path)}")
        
        from models.vc.vevo.vevo_utils import g2p_
        src_phonemes, src_tokens = g2p_(src_text, "en")
        print(f"源文本: {src_text[:50]}...")
        print(f"生成的音素: {src_phonemes[:100]}...")
        print(f"Token IDs (前20个): {src_tokens[:20]}")
        
        ref_phonemes, ref_tokens = g2p_(ref_text, "en")
        print(f"\n参考文本: {ref_text}")
        print(f"参考音素: {ref_phonemes}")
        print(f"参考Token IDs: {ref_tokens}")
    except Exception as e:
        print(f"g2p处理出错: {e}")
        import traceback
        traceback.print_exc()

    def vevo_tts(
        src_text,
        ref_wav_path,
        timbre_ref_wav_path=None,
        output_path=None,
        ref_text=None,
        src_language="en",
        ref_language="en",
    ):
        if timbre_ref_wav_path is None:
            timbre_ref_wav_path = ref_wav_path
        
        # 输出调试信息
        print(f"\n===== TTS参数 =====")
        print(f"源语言: {src_language}")
        print(f"参考语言: {ref_language}")
        
        try:
            # 获取AR输入前的信息
            print("\n===== AR输入前信息 =====")
            # 这里调用g2p_但不用于实际处理，仅为调试
            ar_test_ids = g2p_(src_text, src_language)[1]
            print(f"预期AR输入ID长度: {len(ar_test_ids)}")
            print(f"预期AR输入ID(前20): {ar_test_ids[:20]}")
            
            if ref_text:
                ref_test_ids = g2p_(ref_text, ref_language)[1]
                print(f"预期参考ID长度: {len(ref_test_ids)}")
                print(f"预期参考ID(前20): {ref_test_ids[:20]}")
                
            # 实际处理
            gen_audio = inference_pipeline.inference_ar_and_fm(
                src_wav_path=None,
                src_text=src_text,
                style_ref_wav_path=ref_wav_path,
                timbre_ref_wav_path=timbre_ref_wav_path,
                style_ref_wav_text=ref_text,
                src_text_language=src_language,
                style_ref_wav_text_language=ref_language,
            )
            
            assert output_path is not None
            save_audio(gen_audio, output_path=output_path)
            print(f"音频已保存到: {output_path}")
            
        except Exception as e:
            print(f"TTS处理出错: {e}")
            import traceback
            traceback.print_exc()

    # def vevo_tts(
    #     src_text,
    #     ref_wav_path,
    #     timbre_ref_wav_path=None,
    #     output_path=None,
    #     ref_text=None,
    #     src_language="en",
    #     ref_language="en",
    # ):
    #     if timbre_ref_wav_path is None:
    #         timbre_ref_wav_path = ref_wav_path

    #     gen_audio = inference_pipeline.inference_ar_and_fm(
    #         src_wav_path=None,
    #         src_text=src_text,
    #         style_ref_wav_path=ref_wav_path,
    #         timbre_ref_wav_path=timbre_ref_wav_path,
    #         style_ref_wav_text=ref_text,
    #         src_text_language=src_language,
    #         style_ref_wav_text_language=ref_language,
    #     )

    #     assert output_path is not None
    #     save_audio(gen_audio, output_path=output_path)



    # 1. Zero-Shot TTS (the style reference and timbre reference are same)
    vevo_tts(
        src_text,
        ref_wav_path,
        output_path="./models/vc/vevo/wav/output_vevotts1.wav",
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )

    # 2. Style and Timbre Controllable Zero-Shot TTS (the style reference and timbre reference are different)
    vevo_tts(
        src_text,
        ref_wav_path,
        timbre_ref_wav_path="./models/vc/vevo/wav/english_female.wav",
        output_path="./models/vc/vevo/wav/output_vevotts2.wav",
        ref_text=ref_text,
        src_language="en",
        ref_language="en",
    )




# if __name__ == "__main__":

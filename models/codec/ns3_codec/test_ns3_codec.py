from models.codec.ns3_codec.facodec import FACodecEncoder,FACodecDecoder
from models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
from huggingface_hub import hf_hub_download
import torch
import librosa
import soundfile as sf

def ut_ns3_codec1():
    fa_encoder = FACodecEncoder(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )

    fa_decoder = FACodecDecoder(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
    decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

    fa_encoder.load_state_dict(torch.load(encoder_ckpt))
    fa_decoder.load_state_dict(torch.load(decoder_ckpt))

    fa_encoder.eval()
    fa_decoder.eval()

    ##inference
    test_wav_path = "/Users/donkeyddddd/Documents/Rx_projects/python_projects/Amphion/models/vc/vevo/wav/mandarin_female.wav"
    test_wav = librosa.load(test_wav_path, sr=16000)[0]
    test_wav = torch.from_numpy(test_wav).float()
    test_wav = test_wav.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        # encode
        enc_out = fa_encoder(test_wav)
        print(enc_out.shape)

        # quantize
        vq_post_emb, vq_id, _, quantized, spk_embs = fa_decoder(enc_out, eval_vq=False, vq=True)
        
        # latent after quantization
        print(vq_post_emb.shape)
        
        # codes
        print("vq id shape:", vq_id.shape)
        
        # get prosody code
        prosody_code = vq_id[:1]
        print("prosody code shape:", prosody_code.shape)
        
        # get content code
        cotent_code = vq_id[1:3]
        print("content code shape:", cotent_code.shape)
        
        # get residual code (acoustic detail codes)
        residual_code = vq_id[3:]
        print("residual code shape:", residual_code.shape)
        
        # speaker embedding
        print("speaker embedding shape:", spk_embs.shape)

        # decode (recommand)
        recon_wav = fa_decoder.inference(vq_post_emb, spk_embs)
        print(recon_wav.shape)
        sf.write("/Users/donkeyddddd/Documents/Rx_projects/python_projects/Amphion/models/vc/vevo/wav/mandarin_female_ns3_codec_recon.wav", recon_wav[0][0].cpu().numpy(), 16000)

def ut_ns3_codec2_v2():
    test_wav1_path = "/Users/donkeyddddd/Documents/Rx_projects/python_projects/Amphion/models/vc/vevo/wav/arabic_male.wav"
    test_wav1 = librosa.load(test_wav1_path, sr=16000)[0]
    test_wav1 = torch.from_numpy(test_wav1).float()
    test_wav1 = test_wav1.unsqueeze(0).unsqueeze(0)

    test_wav2_path = "/Users/donkeyddddd/Documents/Rx_projects/python_projects/Amphion/models/vc/vevo/wav/mandarin_female.wav"
    test_wav2 = librosa.load(test_wav2_path, sr=16000)[0]
    test_wav2 = torch.from_numpy(test_wav2).float()
    test_wav2 = test_wav2.unsqueeze(0).unsqueeze(0)
    
    # Same parameters as FACodecEncoder/FACodecDecoder
    fa_encoder_v2 = FACodecEncoderV2(
        ngf=32,
        up_ratios=[2, 4, 5, 5],
        out_channels=256,
    )
    fa_decoder_v2 = FACodecDecoderV2(
        in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True,
    )

    encoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
    decoder_v2_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")

    fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
    fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))

    with torch.no_grad():
        enc_out_a = fa_encoder_v2(test_wav1)
        prosody_a = fa_encoder_v2.get_prosody_feature(test_wav1)
        enc_out_a = enc_out_a[:,:,:481]
        enc_out_b = fa_encoder_v2(test_wav2)
        prosody_b = fa_encoder_v2.get_prosody_feature(test_wav2)

        vq_post_emb_a, vq_id_a, _, quantized, spk_embs_a = fa_decoder_v2(
            enc_out_a, prosody_a, eval_vq=False, vq=True
        )
        vq_post_emb_b, vq_id_b, _, quantized, spk_embs_b = fa_decoder_v2(
            enc_out_b, prosody_b, eval_vq=False, vq=True
        )

        vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
        recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)    
    
    sf.write("/Users/donkeyddddd/Documents/Rx_projects/python_projects/Amphion/models/vc/vevo/wav/arabicM_2_mandarinW_ns3_codecV2_recon.wav", recon_wav_a_to_b[0][0].cpu().numpy(), 16000)
    print("Reconstruction complete.")


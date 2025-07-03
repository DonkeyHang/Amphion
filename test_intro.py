# 先导入补丁程序，修复LangSegment库的导入问题
# import langsegment_patch
import os

#vevo
from models.vc.vevo.infer_vevotimbre import test_timbreInfer
# from models.codec.kmeans.repcodec_model import test_RepCodec
from models.vc.vevo.infer_vevostyle import test_styleInfer
# from models.vc.vevo.infer_vevostyle import test_styleInfer
from models.vc.vevo.infer_vevotts import test_ttsInfer
from models.vc.vevo.infer_vevovoice import test_voiceInfer
from models.codec.ns3_codec.test_ns3_codec import ut_ns3_codec1,ut_ns3_codec2_v2
# maskgct
# from models.tts.maskgct.maskgct_inference import test_maskgct

#metis
# from models.tts.metis.metis_infer_vc import test_metis_infer_vc

if __name__=="__main__":
    # ==================== vevo ====================
    # test_timbreInfer()    # vevotimbre offline done!!!
    # test_styleInfer()     # run ok
    # test_voiceInfer()     # run ok
    # test_ttsInfer()       # bad error!!!!!
    # test_styleInfer()     
    # test_voiceInfer()     # run ok
    # test_ttsInfer()       # bad error!!!!!

    # ==================== maskgct ==================
    # test_maskgct()        # debug not successful

    # ==================== metis ====================
    # test_metis_infer_vc()  # error
    
    # ==================== ns3_codec =================
    # ut_ns3_codec1()
    ut_ns3_codec2_v2()
    xxx = 1
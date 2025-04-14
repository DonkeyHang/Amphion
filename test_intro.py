# 先导入补丁程序，修复LangSegment库的导入问题
import langsegment_patch
import os
from models.vc.vevo.infer_vevotimbre import test_timbreInfer
from models.codec.kmeans.repcodec_model import test_RepCodec
from models.vc.vevo.infer_vevostyle import test_styleInfer
from models.vc.vevo.infer_vevotts import test_ttsInfer


if __name__=="__main__":
    # test_timbreInfer()    # vevotimbre offline done!!!
    # test_RepCodec()       # run ok

    # test_styleInfer()     # run ok
    test_ttsInfer()       

    
    
    xxx = 1
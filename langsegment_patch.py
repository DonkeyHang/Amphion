# 修补LangSegment库的导入问题
import sys
import types

# 首先检查LangSegment库是否已经被导入
if 'LangSegment' in sys.modules:
    # 如果已经导入，我们需要将其移除，以便下面的修改生效
    del sys.modules['LangSegment']

# 从源模块导入真正可用的函数
try:
    # 直接导入模块文件，绕过__init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "LangSegment.LangSegment",
        "/opt/anaconda3/envs/vevo/lib/python3.10/site-packages/LangSegment/LangSegment.py"
    )
    langmodule = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(langmodule)
    
    # 创建新的LangSegment模块
    langsegment_module = types.ModuleType("LangSegment")
    sys.modules["LangSegment"] = langsegment_module
    
    # 为新模块添加需要的函数
    setfilters = getattr(langmodule, "setfilters")
    getfilters = getattr(langmodule, "getfilters")
    getTexts = getattr(langmodule, "getTexts")
    classify = getattr(langmodule, "classify")
    getCounts = getattr(langmodule, "getCounts")
    printList = getattr(langmodule, "printList")
    
    # 添加函数到模块
    langsegment_module.setfilters = setfilters
    langsegment_module.getfilters = getfilters
    langsegment_module.getTexts = getTexts
    langsegment_module.classify = classify
    langsegment_module.getCounts = getCounts
    langsegment_module.printList = printList
    
    # 添加缺失的setLangfilters作为setfilters的别名
    langsegment_module.setLangfilters = setfilters
    langsegment_module.getLangfilters = getfilters
    
    # 创建LangSegment子模块
    langsegment_submodule = types.ModuleType("LangSegment.LangSegment")
    sys.modules["LangSegment.LangSegment"] = langsegment_submodule
    
    # 添加函数到子模块
    langsegment_submodule.setfilters = setfilters
    langsegment_submodule.getfilters = getfilters
    langsegment_submodule.getTexts = getTexts
    langsegment_submodule.classify = classify
    langsegment_submodule.getCounts = getCounts
    langsegment_submodule.printList = printList
    
    # 添加缺失的setLangfilters作为setfilters的别名
    langsegment_submodule.setLangfilters = setfilters
    langsegment_submodule.getLangfilters = getfilters
    
    # 添加子模块到主模块
    langsegment_module.LangSegment = langsegment_submodule
    
    print("LangSegment库导入已修补完成")
except Exception as e:
    print(f"LangSegment库修补失败：{e}")

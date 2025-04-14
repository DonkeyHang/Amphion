# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 直接实现必要的功能，不通过导入有问题的模块

_lang_filters = ["en", "zh", "ja", "ko", "fr", "de"]  # 默认过滤器

def setLangfilters(filters):
    """
    设置要保留的语言过滤器
    """
    global _lang_filters
    _lang_filters = filters
    # 实际调用LangSegment中的函数
    import LangSegment.LangSegment as ls
    return ls.setfilters(filters)

def getTexts(text):
    """
    对文本进行语言分段
    """
    import LangSegment.LangSegment as ls
    return ls.getTexts(text)

# 其他可能需要的函数
def getfilters():
    import LangSegment.LangSegment as ls
    return ls.getfilters()

def classify(text):
    import LangSegment.LangSegment as ls
    return ls.classify(text)

def getCounts():
    import LangSegment.LangSegment as ls
    return ls.getCounts()

def printList(langlist):
    import LangSegment.LangSegment as ls
    return ls.printList(langlist)

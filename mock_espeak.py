# mock模拟版本的EspeakBackend，替代真实的espeak依赖
import json
import os

# 加载词元映射文件
try:
    token_path = "./models/tts/maskgct/g2p/utils/mls_en.json"
    with open(token_path, "r") as f:
        token_data = json.loads(f.read())
        # 获取词元列表（字典键）
        available_tokens = list(token_data.keys())
except Exception as e:
    print(f"警告: 无法加载词元文件: {e}")
    available_tokens = ["a", "i", "u", "e", "o", "p", "t", "k", "b"]

class MockEspeakBackend:
    def __init__(self, language='en', 
                preserve_punctuation=True, 
                with_stress=False, 
                language_switch='keep-flags',
                punctuation_marks=None,
                tie=False,
                words_mismatch='ignore'):
        self.language = language
        self.preserve_punctuation = preserve_punctuation
        self.with_stress = with_stress
        self.language_switch = language_switch
        self.punctuation_marks = punctuation_marks
        self.tie = tie
        self.words_mismatch = words_mismatch
        print(f"初始化模拟EspeakBackend，语言: {language}")
        
        # 为不同语言设置基本的模拟词元
        self.lang_tokens = {
            'cmn': ["a", "i", "u", "o", "zh", "ch"],
            'en-us': ["a", "i", "u", "e", "o", "p", "t", "k"],
            'ja': ["a", "i", "u", "e", "o", "ka", "ki"],
            'ko': ["a", "i", "u", "e", "o", "ga"],
            'fr-fr': ["a", "e", "i", "o", "u", "fr"],
            'de': ["a", "e", "i", "o", "u", "de"]
        }
        # 默认使用available_tokens列表中的词元
        self.tokens = available_tokens
        
    def phonemize(self, text, separator=None, strip=False, njobs=1):
        print(f"模拟Espeak转换: {text}")
        
        # 提供的是列表则对每个元素单独处理
        if isinstance(text, list):
            return [self._mock_phonemize_text(t) for t in text]
        return self._mock_phonemize_text(text)
    
    def _mock_phonemize_text(self, text):
        """将文本转换为空格分隔的音素序列"""
        # 确保返回的词元在token字典中可用
        if not text:
            return ""
            
        # 简单地将每个字符替换为有效的词元
        # 从前几个可用词元中选择，确保生成的词元在token字典中存在
        result = []
        for i, char in enumerate(text):  # 移除字符数量限制
            # 根据字符选择音素词元
            token_index = ord(char) % len(self.tokens)
            result.append(self.tokens[token_index])
            
        # 返回空格分隔的字符串
        return " ".join(result)

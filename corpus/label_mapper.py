import re


class AnalysisMapper:

    def __init__(self, nchars=None, pattern=None):
        self.pattern = r"([^+^]+|[+^][A-Za-z0-9#.]+)" if pattern is None else pattern
        self.nchars = nchars if nchars else 4

    def map(self, analysis, splitted=False):
        if splitted == False:
            analysis = re.findall(self.pattern, analysis)
        lexical_root = analysis[0][-self.nchars:]
        last_affix = analysis[-1]
        return f'{lexical_root}{last_affix}'


a = AnalysisMapper(nchars=2)
label = "бары^Pron+PL+POSS.3SG+ABL"

b = a.map(label)

print(b)


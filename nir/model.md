I have a rule-based morphological analyzer for the Yakut language. Since Yakut is agglutinative, the morphological analyzer takes the surface form of a word and decomposes it into a lexical root followed by a sequence of morphemes. For example:

surface form: үлэлээбэт
morphological analysis: үлэ+N+VRBLZ#1+NEG+PART_PRES

The mapping between the surface form and the corresponding morphological analysis is not one-to-one, given that some surface forms can have more than one possible interpretation.

My idea is to train a model to score ambiguous analysis using the context of the sentence. I have anotated data that looks like this:

{"sentence": "иккитэ [MASK].", "candidates": ["сырыт+V+PART_PAST+PL+POSS.3SG"], "label": 0}
{"sentence": "[MASK] ууга түһэн турабын.", "candidates": ["биир+Num+ADVZ#8", "биир+Num+PRTV", "бии+N+VRBLZ#4+CSTV+CONV#2", "бии+N+VRBLZ#4+PST.3SG"], "label": 0}
{"sentence": "биирдэ [MASK] түһэн турабын.", "candidates": ["уу+Adj+DAT", "уу+N+DAT"], "label": 1}
{"sentence": "биирдэ ууга [MASK] турабын.", "candidates": ["түс+N+VRBLZ#3+REFL/PASS", "түс+V+CONV#1", "түһээ+V+REFL/PASS"], "label": 1}


let's discuss possible approaches.




BATCH

masked token prediction data with multiple candidate answers per masked position, and a label indicating which candidate is correct.
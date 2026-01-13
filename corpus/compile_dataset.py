from glob import glob
import os
import json


files = glob(os.path.join('**', 'processed', '*'), recursive=True)


def format(parses):
    tokens = list()
    roots = list()
    for parse in parses:
        affixes = parse.get('affixes', None)
        if affixes:
            root = parse['root']
            affs = parse['affixes']
            token = f"{root}{''.join(affs)}"
            roots.append(root)
        else:
            token = parse['type']
        tokens.append(token)
    return ' '.join(tokens), roots


lines = list()
unique_words = set()


for file in files:
    print(file)
    with open(file, 'r') as f:
        sentences = json.load(f)
        for sentence in sentences:
            print(sentence)
            parses= sentence['parses']
            formated_parses, extracted_roots = format(parses)
            lines.append(f"{sentence['sah']}\t{formated_parses}")
            for r in extracted_roots:
                unique_words.add(r)


with open('dataset.txt', 'w') as f:
    for line in lines:
        f.write(f'{line}\n')

print(len(unique_words))
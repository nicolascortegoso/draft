from glob import glob

files = [
    'adjectives.lex',
    'adverbs.lex',
    'auxiliaries.lex',
    'cardinals.lex',
    'conjunctions.lex',
    'figuratives.lex',
    'interjections.lex',
    'modals.lex',
    'nouns.lex',
    'onomatopoeias.lex',
    'particles.lex',
    'postpositions.lex',
    'pronouns.lex',
    'verbals.lex'
]


roots = set()
for file in files:
    with open(file, 'r') as f:
        [roots.add(root.strip().split('<')[0]) for root in f.readlines()]


with open('lexical_roots.txt', 'w') as f:
    for root in roots:
        f.write(f'{root}\n')
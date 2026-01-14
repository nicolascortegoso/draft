import re

pattern = re.compile(r'[а-яҥҕөһү]+')

names = list()


file = 'female_names'

with open(f'{file}.txt', 'r') as f:
    for line in f.readlines():
        text = line.lower()
        name = pattern.findall(text)
        names.extend(name)


def get_ending(word):
  if word[-1] in ['у', 'е', 'ё', 'ы', 'а', 'о', 'э', 'я', 'и', 'ю']:
    return '<propn-v>'
  return '<propn-c>'


with open(f'{file}.lex', 'w') as f:
    for n in sorted(names):
        line = f'{n}{get_ending(n)}\n'
        f.write(line)

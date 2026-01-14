with open('surnames.txt', 'r') as f:
    surnames = f.readlines()

with open('surnames_pr.txt', 'w') as f:
    for s in surnames:
        man = s.strip().lower()
        woman = man + 'а'
        for s in [man, woman]:
            last = 'c' if s[-1] not in ['у', 'е', 'ы', 'а', 'о', 'я', 'ю'] else 'v'
            f.write(f'{s.strip().lower()}<propn-{last}>\n')

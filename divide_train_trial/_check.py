import json, re

FIELD = ""

found = False
with open('train_classified.jsonl') as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        text = rec[FIELD]
        if '<Outline>' in text:
            found = True
            print(f'Record {i} still has <Outline> tags')
            m = re.search(r'<Outline>', text)
            print(repr(text[max(0, m.start()-100):m.start()+300]))
            # show if it's inside or outside a Parallel block
            no_parallel = re.sub(r'<Parallel>.*?</Parallel>', '', text, flags=re.DOTALL)
            if '<Outline>' in no_parallel:
                print('  --> This <Outline> is OUTSIDE a <Parallel> block')
            else:
                print('  --> This <Outline> is inside a <Parallel> block (should have been replaced)')
            break

if not found:
    print('No records with <Outline> found — all replaced correctly!')

# Also count total occurrences the raw way
total_outline = sum(
    len(re.findall(r'<Outline>', json.loads(line)[FIELD]))
    for line in open('train_classified.jsonl')
)
print(f'\nTotal raw <Outline> occurrences in output: {total_outline}')

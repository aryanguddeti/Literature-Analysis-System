import os

files = [
    'agents/analyst.py',
    'agents/retriever.py',
    'agents/showcase.py',
    'eval/benchmark_runner.py',
    'core/hitl.py'
]

for f in files:
    try:
        content = open(f, encoding='utf-8').read()
        new = content.replace(
            'datetime.datetime.now(datetime.UTC).isoformat()',
            'datetime.datetime.utcnow().isoformat()'
        )
        open(f, 'w', encoding='utf-8').write(new)
        print(f'Fixed: {f}')
    except Exception as e:
        print(f'Skipped: {f} — {e}')

print('Done!')

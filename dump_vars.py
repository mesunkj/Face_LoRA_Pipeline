import json

with open('colab_runner.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print('--- Runner Variables ---')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        for line in source.split('\n'):
            line = line.strip()
            if line.startswith('FACE_ID =') or line.startswith('PROJECT_DIR =') or \
               line.startswith('EVAL_OUTPUT_DIR =') or line.startswith('FIXED_SEED =') or \
               line.startswith('NEGATIVE_PROMPT =') or 'PROMPTS =' in line or '\"scene_' in line:
                print(line)

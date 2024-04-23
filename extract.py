import json

# Open the Jupyter Notebook file
with open('flow.ipynb', 'r') as notebook_file:
    notebook_content = json.load(notebook_file)

# Extract code blocks from notebook cells
code_blocks = []
for cell in notebook_content['cells']:
    if cell['cell_type'] == 'code':
        code_blocks.append(''.join(cell['source']))

# Write the code blocks to the new Python file
with open('train_flow.py', 'w') as python_file:
    for code_block in code_blocks:
        python_file.write(code_block.strip() + '\n')
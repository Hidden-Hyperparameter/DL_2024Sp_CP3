import json

# Open the Jupyter Notebook file
with open('gan.ipynb', 'r') as notebook_file:
    notebook_content = json.load(notebook_file)

# Extract code blocks from notebook cells
out_blocks = []
# cell = len(notebook_content['cells'][29]['outputs'])
lst = notebook_content['cells'][29]['outputs']
print(len(lst))
for i in range(1,len(lst)-3,2):
    try:
        cell = lst[i]['text']
    except:
        pass
    else:
        out_blocks.append(''.join(cell))

    # print(cell)
# for i,cell in enumerate(notebook_content['cells'][15:]):
#     if cell['cell_type'] == 'code':
#         try:
#             # print(' '.join(cell['outputs'][1]['text']))
#             out_blocks.append(''.join(cell['outputs'][1]['text']))
#             out_blocks.append(f'Epoch {i}')
#         except:
#             pass
#         else:
#             print(i)
        # print('\n'*3)


def plot(data):
    lines = data.split('\n')
    dl1 = []
    dl2 = []
    gl1 = []
    gl2 = []
    for line in lines:
        ls = line.split(' ')
        if len(line)==0:
            continue
        if line[0].startswith('d'):
            dl1.append(float(ls[-2]));dl2.append(float(ls[-1]))
        elif line[0].startswith('g'):
            gl1.append(float(ls[-2]));gl2.append(float(ls[-1]))
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0,len(dl1),len(dl1)//10+1)
    labels = ['D real loss','D fake loss','G discriminate loss','G feature matching loss']
    for i,l in enumerate([dl1,dl2,gl1,gl2]):
        # print(np.array(l))
        plt.plot(x,np.array(l).clip(-5,5)[::10],label=f'{labels[i]}')
    plt.legend()
    plt.show()

s = '\n'.join(out_blocks)
# print(len(s.split('\n')))
plot(s)

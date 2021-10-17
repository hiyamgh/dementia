import os

path = 'results_errors_filtered/'

for filename in os.listdir(path):
    if filename.endswith('.tex'):
        with open(os.path.join(path, filename), 'r') as f:
            lines = f.readlines()
            lines = lines[2:-1]
        f.close()
        open(os.path.join(path, filename), 'w').close()
        with open(os.path.join(path, filename), 'w') as f:
            f.writelines(lines)
        f.close()

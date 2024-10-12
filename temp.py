import glob
import os
import re

full_path = '/root/projects/reinforcement_learning/saves/dqn'
# Use glob to list all .pt files
pt_files = glob.glob(os.path.join(full_path, '**', '*.pt'), recursive=True)
# Print the list of .pt files
pattern = r'/dqn/([a-z]+#\d{3})_\d'
for file in pt_files:
    print(f"file:{file}")
    # Regular expression pattern to match the desired part
    match = re.search(pattern, file)
    if match:
        print(f"match.group(1):{match.group(1)}")
    
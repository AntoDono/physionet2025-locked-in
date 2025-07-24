import sys
import os
import pandas as pd
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_code import find_records, load_label, load_text

data_folder = "./data"
records = find_records(data_folder)
positives_index = []
negatives_index = []

num_of_records = 100
num_of_positives = 0
num_of_negatives = 0

for record in records:
    
    label = load_label(os.path.join(data_folder, record))
    
    if label == 1 and num_of_positives < num_of_records:
        positives_index.append(record)
        num_of_positives += 1
    elif label == 0 and num_of_negatives < num_of_records:
        negatives_index.append(record)
        num_of_negatives += 1
        

### SAVE INPUT (.HEA AND .DAT)
if not os.path.exists("./holdout_data"):
    os.makedirs("./holdout_data")
    
for record in positives_index:
    shutil.copy(os.path.join(data_folder, f"{record}.hea"), os.path.join("./holdout_data", f"{record}.hea"))
    shutil.copy(os.path.join(data_folder, f"{record}.dat"), os.path.join("./holdout_data", f"{record}.dat"))

for record in negatives_index:
    shutil.copy(os.path.join(data_folder, f"{record}.hea"), os.path.join("./holdout_data", f"{record}.hea"))
    shutil.copy(os.path.join(data_folder, f"{record}.dat"), os.path.join("./holdout_data", f"{record}.dat"))
    
print(f"Saved {len(positives_index)} positive records and {len(negatives_index)} negative records to ./holdout_data")

### SAVE OUTPUTS (LABELS)
if not os.path.exists("./holdout_outputs"):
    os.makedirs("./holdout_outputs")

for record in positives_index:
    text = load_text(os.path.join(data_folder, f"{record}.hea"))
    with open(os.path.join("./holdout_outputs", f"{record}.txt"), "w") as f:
        f.write(text)

for record in negatives_index:
    text = load_text(os.path.join(data_folder, f"{record}.hea"))
    with open(os.path.join("./holdout_outputs", f"{record}.txt"), "w") as f:
        f.write(text)

print(f"Saved {len(positives_index)} positive records and {len(negatives_index)} negative records to ./holdout_outputs")
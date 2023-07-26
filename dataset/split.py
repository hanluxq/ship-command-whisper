import json
import random
file1 = "1.json"
file2 = "2.json"
file3 = "3.json"

data = []

with open(file1, "r",encoding="utf-8") as f:
    data.extend(json.load(f))

with open(file2, "r" ,encoding="utf-8") as f:
    data.extend(json.load(f))

with open(file3, "r" ,encoding="utf-8") as f:
    data.extend(json.load(f))

random.shuffle(data)
split_index = int(len(data) * 1)
train_data = data[:split_index]
test_data = data[split_index:]

train_file = "train.json"
test_file = "test.json"

with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)


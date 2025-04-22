import glob
import json

all_data = {}
for filename in glob.glob("okpd2_part_*.json"):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
        all_data.update(data)

with open("okpd2_full.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)
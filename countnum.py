import os
import json

test_dir = "data/vizwiz/test"
json_path = "data/vizwiz/test_grounding.json"
expected_total = 2373  # expected total number of test images

# existing image files (jpg only)
test_images = set(f for f in os.listdir(test_dir) if f.lower().endswith('.jpg'))

# image files listed in JSON
with open(json_path, 'r') as f:
    json_data = json.load(f)
json_images = set(json_data.keys())

# analyze
json_but_missing_in_folder = sorted(json_images - test_images)
folder_but_missing_in_json = sorted(test_images - json_images)

# output
print(f"✅ Images in test folder: {len(test_images)}")
print(f"✅ Entries in JSON: {len(json_images)}")
print(f"🔻 In JSON but missing from test folder: {len(json_but_missing_in_folder)}")
for f in json_but_missing_in_folder[:5]:
    print("    ❌", f)

print(f"🔻 In test folder but missing from JSON: {len(folder_but_missing_in_json)}")
for f in folder_but_missing_in_json[:5]:
    print("    ❌", f)

# check for missing
if len(test_images) != expected_total:
    print(f"\n⚠️ Image count does not match expected! (expected: {expected_total}, actual: {len(test_images)})")
else:
    print("\n✅ Image count matches expected.")
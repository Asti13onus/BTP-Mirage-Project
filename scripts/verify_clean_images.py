import os
from PIL import Image
import glob

root = '/home1/astitva_s/BTP_Mirage_Project/clean_images'
img_paths = glob.glob(os.path.join(root, '**', '*.*'), recursive=True)
valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
count, bad = 0, []

for path in img_paths:
    if not path.lower().endswith(valid_exts):
        continue
    try:
        with Image.open(path) as im:
            im.verify()
        count += 1
    except Exception as e:
        bad.append(path)

print(f"✅ Verified {count} valid images.")
if bad:
    print(f"⚠️ {len(bad)} bad files:")
    for b in bad:
        print('  ', b)
else:
    print("No corrupted images found!")


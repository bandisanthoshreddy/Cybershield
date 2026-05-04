import pandas as pd
import os
import shutil

# These 6 files have label=0=NON-TOXIC (wrong — needs to become 0=TOXIC)
FILES_TO_FIX = [
    "./dataset/english_jigsaw_train.csv",
    "./dataset/english_jigsaw_val.csv",
    "./dataset/english_test.csv",
    "./dataset/english_train.csv",
]

for fpath in FILES_TO_FIX:
    if not os.path.exists(fpath):
        print(f"⚠️  Not found: {fpath}")
        continue

    # Backup original
    shutil.copy(fpath, fpath + ".bak")

    df = pd.read_csv(fpath)
    lbl = next((c for c in ["label", "toxic", "class"]
               if c in df.columns), None)
    if not lbl:
        print(f"⚠️  No label column in {fpath}")
        continue

    # Flip 0↔1
    df[lbl] = df[lbl].apply(lambda x: 1 - int(x) if str(x)
                            in ["0", "1", "0.0", "1.0"] else x)
    df.to_csv(fpath, index=False)
    print(f"✅ Fixed: {fpath}")

print("\nDone. Originals saved as .bak files.")

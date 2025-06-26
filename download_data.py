import os
import urllib.request
import tarfile

# 1. Define URL and local paths
URL         = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
ARCHIVE     = os.path.join("data", "facades.tar.gz")
EXTRACT_DIR = os.path.join("data", "facades")

# 2. Ensure data folder exists
os.makedirs("data", exist_ok=True)

# 3. Download the .tar.gz (if missing)
if not os.path.isfile(ARCHIVE):
    print("Downloading facades.tar.gz …")
    urllib.request.urlretrieve(URL, ARCHIVE)
    print("Download complete.")
else:
    print("Archive already present, skipping download.")

# 4. Extract the archive (if not already extracted)
if not os.path.isdir(EXTRACT_DIR):
    print("Extracting to", EXTRACT_DIR)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall("data")
    print("Extraction complete.")
else:
    print("Dataset already extracted, skipping.")

# 5. List a few files to verify
for split in ("train", "test"):
    folder = os.path.join(EXTRACT_DIR, split)
    print(f"\nContents of `{split}/` ({folder}):")
    if os.path.isdir(folder):
        files = sorted(os.listdir(folder))
        for f in files[:5]:
            print("   ", f)
        print(f"   … {len(files)} files total")
    else:
        print("   ERROR: folder not found!")

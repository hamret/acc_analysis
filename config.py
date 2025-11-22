import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Default ACC Spa homography matrix (Eau Rouge → Raidillon)
HOMOGRAPHY_MATRIX = [
    [3.002, -1.145, 522.11],
    [0.241, 5.885, -842.77],
    [0.0012, 0.0067, 1.0]
]

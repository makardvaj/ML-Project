import pandas as pd
improt numpy as np
import requests
import zipfile
import tarfile
import io
from tqdm import tqdm

url = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
# (Grug Repurposing Knowledge Graph)
# Since the file is quite big, we have already downloaded it manually
# The file has been placed in the CWD.

# download progress bar
response = requests.get(url, stream = True)
tf = tarfile.open(fileobj = io.BytesIO(response.content), mode = "r:gz")
tf.extractall(path = "data/dkrg")

# total_size = int(response.headers.get('content-length', 0))
# block_size = 1024

with open('drkg.tsv.gz', wb)
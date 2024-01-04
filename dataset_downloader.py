import urllib.request
import os
from tqdm import tqdm

# Add the class' name in here to download it.
# Make sure to put the name exactly as it is stored in the google cloud, or otherwise it won't download.

class_names = ['airplane', 'alarm clock', 'ambulance', 'angel',
               'anvil', 'apple', 'arm', 'axe',
               'backpack', 'banana', 'bandage', 'barn',
               'bat', 'boomerang', 'bowtie', 'The Eiffel Tower']

baseurl = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
folder_name = "datasets"

os.makedirs(folder_name, exist_ok=True)

for name in class_names:
    try:
        url = baseurl + name.replace(" ", "%20") + ".npy"
        file_path = os.path.join(folder_name, name + ".npy")

        with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
            file_size = int(response.headers.get('content-length'))
            progress = tqdm(unit="B", unit_scale=True, total=file_size, desc=f"Downloading {name}")

            while True:
                data = response.read(1024)
                if not data:
                    break
                out_file.write(data)
                progress.update(len(data))

            progress.close()

    except Exception as e:
        print(f"Could not download {name}. Make sure the name is correct.")

input("Download is done. Press any key to close this window.")

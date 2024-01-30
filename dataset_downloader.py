import urllib.request
import os
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

# Add a class' name in here to download it.
# Make sure to put the name exactly as it is stored in the google cloud, or otherwise it won't download.
class_names = ['airplane', 'alarm clock', 'ambulance', 'angel',
               'anvil', 'apple', 'arm', 'axe',
               'backpack', 'banana', 'bandage', 'barn',
               'bat', 'boomerang', 'bowtie', 'The Eiffel Tower']

baseurl = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
folder_name = "datasets"

os.makedirs(folder_name, exist_ok=True)

def download_data(_class):
  global baseurl
  global folder_name
  name = _class
  try:
      # get the url of the data and the filepath to save it in
      url = baseurl + name.replace(" ", "%20") + ".npy"
      file_path = os.path.join(folder_name, name + ".npy")
      # download the data with a progress bar
      with urllib.request.urlopen(url) as response, open(file_path, 'wb') as out_file:
          file_size = int(response.headers.get('content-length'))
          # initialize the progress bar
          progress = tqdm(unit="B", unit_scale=True, total=file_size, desc=f"Downloading {name}")

          while True:
              # download and read a chunk of the data, break if there isn't any
              data = response.read(1024)
              if not data:
                  break
              # write the data to the output file and update the progress bar
              out_file.write(data)
              progress.update(len(data))
          # close the progress bar
          progress.close()

  except Exception as e:
      print(f"Error while downloading {name}: {e}. Make sure the name is correct.")

# Download the datasets in parallel
with ThreadPoolExecutor() as executor:
    result = executor.map(download_data, class_names)
print("All done!")

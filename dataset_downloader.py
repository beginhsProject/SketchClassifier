import requests
# Add the class' name in here to download it.
# Make sure to put the name exactly as it is stored in the google cloud, or otherwise it won't download.

class_names = ['airplane', 'alarm clock', 'ambulance', 'angel',
               'anvil', 'apple', 'arm', 'axe',
               'backpack', 'banana', 'bandage', 'barn',
               'bat', 'boomerang', 'bowtie', 'The Eiffel Tower']

baseurl = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
!mkdir datasets
for name in class_names:
  try:
   print(f"downloading {name} to datasets/{name}.npy...\n")
   open("datasets/"+name+".npy","wb").write(requests.get(url=baseurl+name.replace(" ","%20")+".npy").content)
  except Exception as e:
    print(f"Could not download {name}. Make sure the name is correct.")
print("All done!")
input("Press anything to close...")

python -m pip install virtualenv
virtualenv sketch_env
call sketch_env/scripts/activate.bat
python -m pip install -r requirements.txt
echo call sketch_env/scripts/activate.bat > start_training.bat
echo python SketchClassifier.py "train" >> start_training.bat
echo call sketch_env/scripts/activate.bat > start_testing.bat
echo set /p model=Enter model filename >> start_testing.bat
echo python SketchClassifier.py "test" "%%model%%" >> start_testing.bat
echo call sketch_env/scripts/activate.bat > download_data.bat
echo python dataset_downloader.py >> download_data.bat
echo pause >> download_data.bat
del setup.bat

python -m venv "%cd%"
cd scripts
activate.bat
cd ..
pip install -r requirements.txt
echo cd scripts > start_training.bat
echo activate.bat >> start_training.bat
echo cd .. >> start_training.bat
echo start SketchClassifier.py "train" >> start_training.bat
echo cd scripts > start_testing.bat
echo activate.bat >> start_testing.bat
echo cd .. >> start_testing.bat
echo set /p model=Enter model filename >> start_testing.bat
echo start SketchClassifier.py "test" "%%model%%" >> start_testing.bat
del setup.bat

# Installing required packages
pip3 install -r requirements.txt

# Compiling pixel_iteration
python3 setup.py build_ext --inplace

This is necessary in order to import pixel_iteration in codec_1.py. If you don't want to use the optimized module, comment the import

# Running codec_1
ex: python3 codec_1.py ../videos/ducks_take_off_444_720p50.y4m ducks_444_encoded.bin -encode
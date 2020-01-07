# Installing required packages
pip3 install -r requirements.txt

# Compiling cython modules
python3 setup.py build_ext --inplace
python3 setup_golomb.py build_ext --inplace

This is necessary in order to generate the required .so files.

# Running codec_1 (lossless intra frame only)
## Encoding
ex: python3 codec_1.py ../videos/ducks_take_off_444_720p50.y4m ducks_444_encoded.bin -encode

## Decoding
ex: python3 codec_1.py ducks_444_encoded.bin ducks_444.y4m -decode

# Running codec_2 (lossless hybrid)
## Encoding
ex: python3 codec_2.py ../videos/ducks_take_off_444_720p50.y4m ducks_444_encoded.bin -encode

## Decoding
ex: python3 codec_2.py ducks_444_encoded.bin ducks_444.y4m -decode
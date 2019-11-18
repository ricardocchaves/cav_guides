# Installing required packages
pip3 install -r requirements.txt

# Encoding and Decoding Lossless
## Using only differences (without prediction)
python3 codec.py

## Using differences, prediction and M prediction
python3 encode.py
python3 decode.py

# Encoding and Decoding Lossy
python3 encode_lossy.py
python3 decode_lossy.py
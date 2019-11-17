from codec import decode
import sys

def main():
    if len(sys.argv) < 2:
        print("USAGE: python3 {} encoded_sound_file.wav".format(sys.argv[0]))
        return
    fname = sys.argv[1]
    decode(fname,"./decoded_file.wav")

if __name__ == "__main__":
    main()
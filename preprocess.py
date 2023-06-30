import os
import sys
import csv
from pydub.utils import mediainfo

def write_audio_files(input_folder):
    with open('audios.txt', 'w') as output_file:
        for file_name in os.listdir(input_folder):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder, file_name)
                audio_info = mediainfo(file_path)
                output_file.write(f"{file_path}|{audio_info['duration']}\n")

def write_text_files(input_folder):
    metadata_file = os.path.join(input_folder, 'metadata.csv')
    wav_folder = os.path.join(input_folder, 'wavs')

    with open(metadata_file, 'r') as csvfile, open('texts.txt', 'w') as output_file:
        csvreader = csv.reader(csvfile, delimiter='|')
        for row in csvreader:
            file_path = os.path.join(wav_folder, row[0] + '.wav')
            output_file.write(f"{file_path}|{row[1]}\n")

def main():
    if len(sys.argv) < 3:
        print('Please provide a mode ("--audio" or "--text") and an input folder')
        return

    mode = sys.argv[1]
    input_folder = sys.argv[2]

    if mode == '--audio':
        write_audio_files(input_folder)
    elif mode == '--text':
        write_text_files(input_folder)
    else:
        print('Invalid mode. Please provide either "--audio" or "--text"')

if __name__ == '__main__':
    main()

import soundfile as sf
import os
import sys
import csv
import hparams
from phonemizer.backend import EspeakBackend
from tqdm import tqdm

def write_audio_files(input_folder):
    with open('train.txt', 'w') as output_file:
        for file_name in tqdm(os.listdir(input_folder)):
            if file_name.endswith('.wav'):
                file_path = os.path.join(input_folder, file_name)
                audio_info = sf.info(file_path).duration
                output_file.write(f"{file_path}|{audio_info['duration']}\n")
    #creating eval.txt, that contains the last 5 lines from train.txt, and deleting them out of train.txt using bash
    os.system("""tail -n 5 train.txt > eval.txt && sed -i -e :a -e '$d;N;2,5ba' -e 'P;D' train.txt""")

def write_text_files(input_folder):
    phonemizer= EspeakBackend(
        hparams.defaults['lang'],
        punctuation_marks=';:,.!?',
        preserve_punctuation=True,
        with_stress=True,
        language_switch='remove-flags'
    )
    metadata_file = os.path.join(input_folder, 'metadata.csv')
    wav_folder = os.path.join(input_folder, 'wavs')

    with open(metadata_file, 'r') as csvfile, open('train.txt', 'w') as output_file:
        csvreader = csv.reader(csvfile, delimiter='|')
        for row in tqdm(csvreader):
            file_path = os.path.join(wav_folder, row[0] + '.wav')
            text=row[1]
            text=phonemizer.phonemize(text)
            output_file.write(f"{file_path}|{text}\n")
    #creating eval.txt, that contains the last 5 lines from train.txt, and deleting them out of train.txt using bash
    os.system("""tail -n 5 train.txt > eval.txt && sed -i -e :a -e '$d;N;2,5ba' -e 'P;D' train.txt""")
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

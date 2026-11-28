from actions import *
from audio_extract import *

def the_divorce():
    folders = fetch_all_folders()
    os.chdir("raw_data")

    for folder in folders:
        print(os.getcwd())
        os.chdir(folder) #go into that folder
        video_files = os.listdir(os.getcwd()) #get a list of all files

        os.chdir("..")
        os.chdir("..")

        for video_name in video_files:
            audio_name = video_name.replace(".webm", ".wav")
            print(audio_name)

            # Load the video file
            audio_path = "./raw_audio/" + folder + "/" + video_name
            video_path = "./raw_data/" + folder + "/" + video_name

            extract_audio(input_path=video_path, output_path=audio_path)


the_divorce()
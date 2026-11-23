from actions import *

print("hello world!")

def save_data(clip_data):
    #print(os.getcwd())
    os.chdir("visual_data") #change to correct folder

    filename = clip_data["id"]

    chunk_to_save = {
        "data" : clip_data["features"],
        "label" : clip_data["name"]
    }

    try:
        data_file = open(filename, 'wb')
        pickle.dump(chunk_to_save, data_file)
        data_file.close()
    except:
        print("Unable to save to file...")

    os.chdir("..") #change back to correct folder

def process_data():
    print("Processing Data...")

    #first, collect all the data
    folders = fetch_all_folders()
    all_data = fetch_all_videos(folders)

    for clip in all_data:
        print("Processing data for ", clip["name"])
        clip["frames"] = cut_mouths(clip) #cutting out the mouths from each clip
        #then we want to get the feature vectors for each frame

        features = []
        for frame in clip["frames"]:
            dct_feature = whole_mouth_dct(frame)
            features.append(dct_feature)

        clip["features"] = features 
        #so now, the clip["features"] has an array of arrays. 
        # Each array contains a feature vector for the mouth
        # For each frame in the clip
        save_data(clip)
    



def run_here():
    #process data by uncommenting this bit
    process_data()

    #train model by uncommenting this bit

run_here()
from actions import *

print("hello world!")

#processing
print("Processing...")
folders = fetch_all_folders()
print(folders)
test = fetch_all_videos(folders)

#print(test[0]["name"])
#print(test[0]["id"])
#test_frame = test[0]["frames"][10]
#test_frame = flip_rgb(test_frame)

#show_frame(test_frame)#show the first frame
#plot_pixel_histogram(test_frame)#plot the first frame
#mean_rgb(test_frame, 505, 582, 931, 1060)

def shrink_to_mouth():
    for clip in test:
        #loop through the test clips
        mouths = []
        for frame in clip["frames"]:
            #loop through frames of each clip
            just_mouth = mouth_locator(frame)
            
            show_frame(just_mouth)

shrink_to_mouth()

#notes to rho:
#the mouth region always appears to be in the 400 zone for the x coordinate
#so discard mouths that are outside of this region?





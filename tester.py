from actions import *

print("hello world!")

#processing
folders = fetch_all_folders()
print(folders)
test = fetch_all_videos(folders)


print(test[0]["name"])
print(test[0]["id"])
test_frame = test[0]["frames"][10]
test_frame = flip_rgb(test_frame)

show_frame(test_frame)#show the first frame
plot_pixel_histogram(test_frame)#plot the first frame
mean_rgb(test_frame, 505, 582, 931, 1060)

#my lips apparently have a mean rgb value of (163, 105, 92)







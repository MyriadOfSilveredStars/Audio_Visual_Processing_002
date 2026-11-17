from actions import *

print("hello world!")

#processing
test = fetch_all_videos()

print(test[0]["name"])
test_frame = test[0]["frames"][10]
test_frame = flip_rgb(test_frame)

show_frame(test_frame)#show the first frame
plot_pixel_histogram(test_frame)#plot the first frame
mean_rgb(test_frame)

#my lips apparently have a mean rgb value of (163, 105, 92)






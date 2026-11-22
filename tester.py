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

def dct_1d(x):
    N = len(x)
    n = np.arange(N)
    X = np.zeros(N)
    top = 1

    for k in range(N):

        multiplier = ((np.pi * k) * ((2 * n)+1)) / (2 * N)

        s = np.sqrt(top/N)
        X[k] = s * np.sum(x * np.cos(multiplier))
        top = 2
    
    return X

def shrink_to_mouth():
    for clip in test:
        #loop through the test clips
        mouths = []
        for frame in clip["frames"]:
            #loop through frames of each clip
            just_mouth = mouth_locator(frame)
            just_mouth = flip_rgb(just_mouth)
            
            show_frame(just_mouth)

            do_dct(just_mouth)

def do_dct(mouth):
    #h, w, c = mouth.shape() #get height, width, and channels

    x = [200, 200, 50, 50]
    X = dct_1d(x)

    print(f"X= {X}")
    print("OpenCV:", cv.dct(np.float32(x).reshape(1, -1)).ravel())

    # Apply 2D DCT to a floating-point type array
    img_dct = cv.dct(np.float32(mouth))

    # scale data for better visualisation
    img_dct_log = np.log(np.abs(img_dct))
    plt.imshow(img_dct_log, cmap="jet")
    plt.show()


shrink_to_mouth()

#notes to rho:
#the mouth region always appears to be in the 400 zone for the x coordinate
#so discard mouths that are outside of this region?





## IMPORTING EVERYTHING ##
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import glob
from pathlib import Path
import os
import pickle
from scipy.signal import get_window
import scipy.fftpack as fft

from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D,InputLayer, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model, load_model

from sklearn import metrics

import cv2 as cv
import skimage as skimg

## LOADING TEST DATA ##

def trim_id(id):
    id = id.split(".")
    id = id[0]
    return id

def trim_name(name):
    name = name.split("(")
    name = name[0]
    return name

def fetch_all_folders():
    # RHO NEEDS TO CHANGE THIS TO NEW FILE STRUCTURE
    os.chdir("AVP_002")
    cwd = os.getcwd() + "/raw_data" #locate the "names" folder

    name_folders = os.listdir(cwd) #get a list of all the folders
    #print(name_folders)

    return name_folders

def fetch_all_videos():
    os.chdir("test_data") #change this to the proper folder after testing
    #for now, this simply runs through all the videos in one folder

    test_videos = []
    video_files = os.listdir(os.getcwd()) #get a list of all files

    for video_name in video_files:
        try:
            video = cv.VideoCapture(video_name)
            fps = video.get(cv.CAP_PROP_FPS)
            frames = []

            while video.isOpened():
                ret, frm = video.read()

                if not ret:
                    break

                frames.append(frm)
                

            video.release()
            cv.destroyAllWindows()

            video_dict = {
                "name" : video_name,
                "fps" : fps,
                "frames" : frames
            }
            test_videos.append(video_dict)
            print(video_dict["name"])

        except:
            print("Could not open video file...")

    os.chdir("..")
    return test_videos

    

def extract_test_audio(name_folders):
    test_audio = []

    #go one by one through each folder
    os.chdir("raw_data")
    for folder in name_folders:
        os.chdir(folder)
        audio_files = os.listdir(os.getcwd())

        for audio in audio_files:
            try:
                r, fs = sf.read(audio, dtype='float')
                audio_file = {
                    "id" : trim_id(audio),
                    "r" : r,
                    "fs" : fs,
                    "name" : trim_name(folder)
                }
                test_audio.append(audio_file)
            except IOError:
                print("Failed to open file: " + IOError)
            except SystemError:
                print("Failed to open file: " + SystemError)
            except:
                print("Failed to open file...")
            
        os.chdir("..")

    os.chdir("..")
    return test_audio

## HANDLING AUDIO FILES ##

def load_audio(filename):
    r, fs = sf.read(filename, dtype='float32')
    return r, fs

def save_audio(r, fs, filename):
    print("Saving audio as .wav file...")
    sf.write(filename, r, fs)

def record_audio():
    print("Recording Audio...")
    fs = 44100 #sampling rate
    seconds = 1 #number of seconds
    r = sd.rec(seconds * fs, samplerate=fs, channels=1) #record some audio for five seconds
    sd.wait() #wait until audio is recorded

    return r, fs

def play_audio(r, fs):
    print("Playing audio...")
    sd.play(r, fs)
    sd.wait() #wait until audio has finished playing

## PLOTTING GRAPHS ##

def plot_multiple(x_axes):
    #input an array of x values to plot
    num_axs = len(x_axes)
    fig, axs = plt.subplots(num_axs)

    for i in range(0, num_axs - 1):
        axs[i].plot(x_axes[i])

    plt.show()

def plot_audio_pyplot(x, plot_title, x_label, y_label):
    plt.plot(x)
    plt.title = plot_title
    plt.xlabel = x_label
    plt.ylabel = y_label

    plt.show()

def plot_spectrogram(r, fs):
    #this is for a narrowband spectrogram
    plt.specgram(r, window=np.hamming(512), noverlap=400, NFFT=512, Fs=fs)
    plt.title = "Spectrogram"
    plt.show()

def plot_mfcc(mfcc):
    #plot the MFCC - Code from https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/#melfilterbank
    plt.figure(figsize=(14, 5))
    plt.imshow(mfcc.T, cmap='hot', aspect='auto')
    plt.title = "MFCC"
    plt.xlabel = "Frame Index"
    plt.ylabel = "Cepstral Coefficient Index"
    plt.show()

def plot_history(history):
    #check that the training has gone as intended
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_confusion_matrix(model, X_test, y_test):
    #pass in the model and the test data
    predicted_probs=model.predict(X_test,verbose=0)
    predicted=np.argmax(predicted_probs,axis=1)

    confusion_matrix = metrics.confusion_matrix(np.argmax(y_test,axis = 1), predicted)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.show()

## AUDIO PROCESSING ##

def frame_audio(audio, fft_size, hop_size, fs):
    #this essentially splits the audio into frames
    #but the frames overlap, rather than being simple slices

    audio = np.pad(audio, int(fft_size / 2), mode='reflect')
    frame_len = np.round(fs * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - fft_size) / frame_len) + 1
    frames = np.zeros((frame_num, fft_size))

    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+fft_size]

    return frames
    
def hamming_audio(frame, ham_size):
    #hamming code helped by https://numpy.org/doc/stable/reference/generated/numpy.hamming.html
    window = np.hamming(ham_size)

    audio_win = frame * window
    return np.transpose(audio_win)

def fast_fourier_transform(speechFrame):
    #calculate the DFT
    dft = np.fft.fft(speechFrame)

    return dft

def get_magnitude_spec(dftFrame, half_length):
    half = int(half_length)
    magSpec = np.abs(dftFrame)
    magSpec = magSpec[:half] #half the magSpec, as it is reflected

    return magSpec

def get_phase_spectrum(dftFrame):
    phaseSpec = np.angle(dftFrame)

    return phaseSpec

def mag_and_phase(speechFrame): #altogether now
    length_data = len(speechFrame)
    half_length = length_data / 2

    #first hamming
    hammed_Frame = hamming_audio(speechFrame)

    #then DFT
    fft_Frame = fast_fourier_transform(hammed_Frame)

    #then take magnitude and phase spectrums
    magFrame = get_magnitude_spec(fft_Frame, half_length)
    phaseFrame = get_phase_spectrum(fft_Frame)

    return magFrame, phaseFrame


## CONVERSIONS ##

def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def freq_to_mel(freq): #convert frequency to mel-frequency
    mel = 1125 * np.log(1 + (freq/700))
    return mel

def mel_to_freq(mel): #convert from mel-frequency to frequency
    mel_div = mel / 1125
    freq = 700 * (np.exp(mel_div) - 1)

    return freq

def find_mel_points(fs, num_channels):
    min = 0
    max = fs / 2

    minMel = freq_to_mel(min)
    maxMel = freq_to_mel(max)
    melPoints = np.linspace(minMel, maxMel, num_channels + 2)

    return melPoints

def calculate_bin_points(fs, num_channels, length):
    min = 0
    max = fs / 2

    minMel = freq_to_mel(min)
    maxMel = freq_to_mel(max)
    melPoints = np.linspace(minMel, maxMel, num_channels + 2)

    freqPoints = mel_to_freq(melPoints)

    binPoints = np.floor((length + 1) * freqPoints / fs).astype(int)

    return binPoints

## FILTERBANKS ##

def new_mel_filterbank(magSpec, num_channels, binPoints):
    #where k is index in magnitude spectrum and m is filterbank index
    filterbank = np.zeros((num_channels, len(magSpec)))

    for channel in range(0, num_channels):
        #calculate values
        left = binPoints[channel]
        centre = binPoints[channel + 1]
        right = binPoints[channel + 2]

        #work out the filter
        for k in range(len(magSpec)):
            if k < left:
                filterbank[channel, k] = 0
            elif left <= k and k <= centre:
                filterbank[channel, k] = (k-left) / (centre - left)
            elif centre <= k <= right:
                filterbank[channel, k] = (right - k) / (right - centre)
            elif k > right:
                filterbank[channel, k] = 0
            else:
                print("Error")

    # application of the filterbank code from https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
    # normalise the filters
    enorm = 2.0 / (binPoints[2:num_channels + 2] - binPoints[:num_channels])
    filterbank *= enorm[:, np.newaxis]

    #plt.figure(figsize=(15,4))
    #for n in range(filterbank.shape[0]):
        #plt.plot(filterbank[n])
    #plt.show()

    # then filter the audio and apply log
    audio_filtered = np.dot(filterbank, np.transpose(magSpec))
    audio_log = 10.0 * np.log10(audio_filtered)

    return audio_log

def linear_rectangular_filterbank(magSpec, num_channels):
    #code for a linear filterbank
    fbank = np.zeros(num_channels)
    length = len(magSpec)

    min = 0
    max =  int(length / num_channels) #starting values

    length = len(magSpec)

    for i in range(0, num_channels):
        r = np.zeros(length)
        r[min:max] = 1 #set section to equal 1

        fbank[i] = np.matmul(r, magSpec)
        min = max
        max = max + 32

    return fbank

## FINAL PROCESSING ##

def dct_time(num_filters, filter_len):
    #code for the dct from https://www.kaggle.com/code/ilyamich/mfcc-implementation-and-tutorial
    basis = np.empty((num_filters,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, num_filters):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis

def save_to_file(full_data):
    #print(os.getcwd())
    os.chdir("noisy_mfccs") #change to correct folder

    filename = full_data["id"]

    chunk_to_save = {
        "data" : full_data["MFCC"],
        "label" : full_data["name"]
    }

    try:
        filename = "noisy_" + filename
        data_file = open(filename, 'wb')
        pickle.dump(chunk_to_save, data_file)
        data_file.close()
    except:
        print("Unable to save to file...")

    os.chdir("..") #change to correct folder

## RUNNING NEURAL NETWORK ##

def unpickle_data(filename):
    #extract the pickled dictionary files
    with open(filename, 'rb') as data_file:
        try:
            data_extract = pickle.load(data_file)
            return data_extract
        except:
            print("Unable to unpickle the file...")


def get_accuracy(model, X_test, y_test):
    #get the accuracy for the model

    predicted_probs = model.predict(X_test,verbose=0)
    predicted = np.argmax(predicted_probs,axis=1)
    actual = np.argmax(y_test,axis=1)
    accuracy = metrics.accuracy_score(actual, predicted)
    print(f'Accuracy: {accuracy * 100}%')

    print("Predictions: " + str(predicted))

def create_model():
    #create the model
    numClasses = 20 #number of classes there are
    model = Sequential()

    #input layer
    model.add(InputLayer(input_shape=(110, 40, 1)))

    #hidden decision layers
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    #converging decisions into one output
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(numClasses))
    model.add(Activation('softmax')) #for multi-class classifiers

    return model

def find_most_frames(data):
    max_frames = 0
    for i in data:
        for j in i:
            num_frames = len(j)
            if num_frames > max_frames:
                max_frames = num_frames

    print(max_frames)

def handle_mfcc_files():
    data = []
    labels = []

    #print(os.getcwd())
    #alter to match your file structure
    os.chdir("AVP_002")
    os.chdir("mixed_mfccs")

    all_files = os.listdir(os.getcwd()) #get a list of all the folders
    #print(all_files)

    max_frames = 110 #there are a maximum of 110 sets of 40
    filler = np.zeros((40)) #filler array

    for mfcc_file in all_files:
        file_data = unpickle_data(mfcc_file)
        
        mfcc_data = file_data["data"]
        data_label = file_data["label"]

        #fill out so everything is homogenous
        while int(len(mfcc_data) < max_frames):
            mfcc_data.append(filler)

        data.append(mfcc_data)
        labels.append(data_label)

    print(max_frames)
    print(np.shape(data))

    os.chdir("..")
    #print(len(data))
    #convert to numpy arrays
    labels = np.array(labels)
    data = np.array(data)

    #normalise the data
    data = data / np.max(data)
    #print(np.shape(data))

    return data, labels

def the_machine(data, labels):
    #properly encode the labels, so they can be read by the machine
    label_encoder = LabelEncoder()
    classes = ['AARON', 'ALEX', 'ALI', 'CHARLIE', 'HARRIET', 'JACEK', 'JACK', 'JADE', 'JOSHUA', 'MASON', 'NATHAN', 'NICHOLAS', 'SAM', 'SAMUEL', 'SEAN', 'TOBY', 'TOMMY', 'XINGCHEN', 'ZIANG', 'ZIHAN']
    label_encoder = label_encoder.fit(classes)
    new_labels = to_categorical(label_encoder.transform(labels))

    #split the data into 80% training, 10% validation, and 10% testing
    X_train, X_tmp, y_train, y_tmp = train_test_split(data, new_labels, test_size=0.2, random_state=0)
    #use the temp value to split the 20% into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0)
    
    print("Test data shape: " + str(np.shape(X_test)))

    #now create the model
    model = create_model()
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
    model.summary()
    
    #set up the hyperparameters for the model
    num_epochs = 15
    num_batch_size = 32

    #and run the model...
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose=1)

    #save the weights
    model.save("name_classifier.h5")
    model.save_weights("mixed_model_weights.weights.h5")

    #get accuracy
    get_accuracy(model, X_test, y_test)
    plot_confusion_matrix(model, X_test, y_test)

    return history, model


## MAIN CALLERS ##


def process_data(audio_dict): #call this to process the data entirely
    #first set up some variables
    fs = audio_dict["fs"]
    r = audio_dict["r"] / np.max(np.abs(audio_dict["r"])) #normalise the audio

    num_channels = 10 #channels for the filterbank
    fft_size = 2048
    mfccs = []

    #frame the audio
    framed_r = frame_audio(r, 2048, 15, fs)

    for frame in framed_r:
        
        #first, hamming time
        hammed_frame = hamming_audio(frame, fft_size)
        #plot_audio_pyplot(hammed_frame, "Hammed Frame", "x", "y")
        #print("Hamming frame: " + str(np.shape(hammed_frame)))

        #then apply the fast fourier transform
        fft_audio = fast_fourier_transform(hammed_frame)
        #print(fft_audio)
        #plot_audio_pyplot(fft_audio, "Fast Fourier Audio", "x", "y")
        #print("FFT: " + str(np.shape(fft_audio)))

        #now find the magnitude spectrum
        magspec = get_magnitude_spec(fft_audio, len(fft_audio) / 2)
        #plot_audio_pyplot(magspec, "Magnitude Spectrum", "x", "y")
        #print("Mag Spec: " + str(np.shape(magspec)))

        #then we can apply the filterbank
        length = len(magspec)
        binPoints = calculate_bin_points(fs, num_channels, length)
        filtered_audio = new_mel_filterbank(magspec, num_channels, binPoints)
        #print("Filtered Audio: " + str(np.shape(filtered_audio)))

        dct_filters = dct_time(40, num_channels)
        cepstral_coefficents = np.dot(dct_filters, filtered_audio)
        #print("Cepstral Coefficients: " + str(np.shape(cepstral_coefficents)))

        #print(np.shape(cepstral_coefficents))
        mfccs.append(cepstral_coefficents)

    #print(np.shape(mfccs))

    processed_dict = {
        "name" : audio_dict["name"],
        "id" : audio_dict["id"],
        "fs" : audio_dict["fs"],
        "r" : audio_dict["r"],
        "MFCC" : mfccs
    }

    return processed_dict

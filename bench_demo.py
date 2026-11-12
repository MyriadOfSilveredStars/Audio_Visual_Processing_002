from actions import *

from pynput.keyboard import Key, Listener

#okay from the top
# we want to run the machine first, so it is ready and loaded

def run_machine():
    data, labels = handle_mfcc_files()
    
    history, model = the_machine(data, labels)
    
    #plot_history(history)
    return model

def decode_labels(predicted_value):
    #this turns the numbers back into names, for testing
    label_encoder = LabelEncoder()
    classes = ['AARON', 'ALEX', 'ALI', 'CHARLIE', 'HARRIET', 'JACEK', 'JACK', 'JADE', 'JOSHUA', 'MASON', 'NATHAN', 'NICHOLAS', 'SAM', 'SAMUEL', 'SEAN', 'TOBY', 'TOMMY', 'XINGCHEN', 'ZIANG', 'ZIHAN']
    label_encoder = label_encoder.fit(classes)

    result = label_encoder.inverse_transform(predicted_value)

    return result

def handle_recording():
    #record the audio
    r, fs = record_audio()
    
    #check that it worked and has recorded
    #play_audio(r, fs)
    save_audio(r, fs, "test_audio.wav")
    r, fs = load_audio("test_audio.wav")
    #plot_audio_pyplot(r, "Recorded Audio", "Time", "Audio")

    #process the test audio
    test_audio = {
        "fs" : fs,
        "r" : r,
        "name" : "TEST",
        "id" : "TEST"
    }
    
    processed_dict = process_data(test_audio)
    return processed_dict

def analyse_speech(test):

    
    data = test["MFCC"] #the data needs padding
    
    max_frames = 110 #there are a maximum of 110 sets of 40
    filler = np.zeros((40)) #filler array

    while int(len(data) < max_frames):
        data.append(filler)

    #turn to numpy array
    test_data = [] #needs to be a 3d array i think
    test_data.append(data)
    test_data = np.array(test_data)
    
    #normalise the data
    test_data = test_data / np.max(test_data)

    #print(np.shape(test_data)) #should be [1, 110, 40]

    single_item_model = create_model()
    single_item_model.load_weights('rho_model_weights.weights.h5')
    single_item_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))

    predicted_probs = single_item_model.predict(test_data,verbose=0)
    predicted = np.argmax(predicted_probs,axis=1)
    pred_name = decode_labels(predicted)
    print("You said: " + str(pred_name))


def demo(key):
    if key == Key.enter:
        test = handle_recording()
        analyse_speech(test)

    elif key == Key.backspace:
        print("End of demo!")
        return False

print("Press ENTER to start recording")
print("Press BACKSPACE to end demo")
with Listener(on_press = demo) as listener:   
    listener.join()
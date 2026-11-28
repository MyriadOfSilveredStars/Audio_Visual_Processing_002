# Audio_Visual_Processing_002

Repository for the second AVP coursework.
Contributors: Rho Bates and Esme Arthur-Aidoo

Last Updated: 28th November, by Rho
Changes: the model runs with all twenty names. Current accuracy is 25%. I'm now working on getting the mouth frames to be split into chunks of 8x8, because that should increase accuracy. Like how the waveforms were split into frames when we did audio

To Do:
- get machine running with blocked data
- extract audio from visual
- overlay noise to audio (?)
- use late stage integration and configure weighting
- all together now!!

- esme working on poster & helping out with code when possible

CURRENT SOFTWARE:
- Data recorded in .webm format
- Data split manually into separate names
- Each name video loaded as a series of frames
- Audio to be split from video and saved as a separate file
FOR VIDEO
- CV's mouth cascade classifier used to locate mouth region in each frame
- Find DCTs of each mouth region and label appropriately
- Using DCTs, analyse mouth region movements and train model
FOR AUDIO
- Perform same processing as Coursework 001
- Overlay noise to entire dataset to have one noisy and one clean dataset
- Obtain MFCCs for each aduio file
- Train model on noisy and clean data
COMBINE
- Late-stage integration
- Experiment with weighting each model, until the highest accuracy is reached


Uploaded from previous coursework:
- Audio recogniser with 95% accuracy
- Works with both noisy and clean data
- Runs with 15 epochs, with batch sizes of 32
- Has two Conv2D layers and runs with a learning rate of 0.001





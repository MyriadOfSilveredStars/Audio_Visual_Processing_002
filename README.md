# Audio_Visual_Processing_002

Repository for the second AVP coursework.
Contributors: Rho Bates and Esme Arthur-Aidoo

Last Updated: 23rd November, by Rho
Changes: got the buggering dct function to work

To Do:
- rho to set up the skeleton of the learning model & extract audio from video
- esme to possibly develop zig-zag dct extraction, if understood. otherwise, working on poster

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





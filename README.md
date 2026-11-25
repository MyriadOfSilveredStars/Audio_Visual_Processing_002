# Audio_Visual_Processing_002

Repository for the second AVP coursework.
Contributors: Rho Bates and Esme Arthur-Aidoo

Last Updated: 25th November, by Rho
Changes: the model runs on the current feature vectors. Only works with four names so far, because I still need to process the other names, but we're getting 60% accuracy

To Do:
- extract audio from visual
- overlay noise to audio (?)
- process all the names so the model runs with all 20
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





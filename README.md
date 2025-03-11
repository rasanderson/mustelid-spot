# Fox-spot
Code repository for the Fox detection and classification project using AI. This project between Newcastle University and Perdix is running from September 2024 to May 2025. 

## Team

The Newcastle University team is made up of Roy Sanderson, Pete Robertson, Nicholas Allen and Ava Waine.

## Project plan
### Objectives
-   Create a dataset of camera trap images split into common species occuring in the British countryside
-   Create a species classification model for camera trap images

### Create a comprehensive dataset
#### Image sources
A comprehensive dataset has been compiled from multiple sources and partners. The following table lists contributors and their contact details.

|Contributor |Contact  | Contribution|
--- | --- | ---|
|APHA|[Sarah Beatham](mailto:Sarah.Beatham@apha.gov.uk)|Annotated camera trap images of Fox, Boar and Muntjack in Loch Ness and Forest of Dean areas|
|Newcastle University|[Aileen Mill](mailto:aileen.mill@newcastle.ac.uk)|Camera trap images at Gosforth Nature Reserve over 4 years|
|Newcastle University|[Marion Pfeifer](mailto:Marion.Pfeifer@newcastle.ac.uk)|Camera trap images at Cockle Park over 2 years|
|Newcastle University|[Mark Whittingham](mailto:mark.whittingham@newcastle.ac.uk)|Camera trap platform access (Agouti) for mob grazing trials at Cockle Park, *not used at this time*|
|MammalWeb and Durham University|[Phil Stevens](mailto:philip.stephens@durham.ac.uk)|Agreed in principle to help, but at cost of extracting images, open to collaboration|
|iNaturalist|[Online]()|Camera trap and other images
|Wildlife Insights|[Online]()|Camera trap and other images
|Badger Trust|[Victoria Coulton](mailto:victoria@badgertrust.org.uk)|Badger Camera trap videos
|CEH, Chornobyl Exclusion Zone, Ukraine|[Online](https://catalogue.ceh.ac.uk/documents/a657ffc3-8f62-458f-bcb7-30e116807174)|Camera trap images of Deer, Hares, Foxes, Squirrels, Wild Boar|


CNNs perform best when trained on a balanced dataset. Therefore 5,000 images of the following classes were gathered for inclusion in model training:
-	Fox
-	Badger
-	Deer
-	Squirrels
-	Birds
-	Dog
-	Human
-	Rabbit/Hare

An 'other' class was discussed, but not included due to the range of possible images, which would likely confuse the model. Instead, in deployment the model defaults to 'unknown' when confidence in the predictions are low. 

#### Folder structure
The images are structured to allow for rapid deployment in your own training models. There are two main folders, Original Images, and Pre-processed images. The pre-processed images are ready transformed to 224, 224 resolution, and single channel (Grayscale) for use in CNN or other models. 

The exact structure is as follows:
```
├── Raw
│   ├── fox
│   │   ├── source
│   ├── not fox
│   │   ├── species 1
│   │   ├── species 2
├── Pre-processed
│   ├── fox
│   │   ├── source
│   ├── not fox
│   │   ├── species 1
│   │   │   ├── source
│   │   ├── species 2
```
### Data pre-processing and augmentation
Images from our dataset and partners were first passed through EcoAssist, now [AddaxAI](https://addaxdatascience.com/addaxai/) in order to leverage Megadetector's cropping function. Indeed, cropped images of animals were used in order to standardise the size of the individual and remove effects of background and other confounding factors, allowing the model to pick out features of interest directly on the animal. 

The data was pre-processed first to resize the images to 224, 224 pixels which is an ideal size for CNN models. The images were converted to grayscale, known as single channel images, to homogenise the images for the model, as most fox sightings are at night. 

The pre-processed images were then augmented to double the dataset size by horizontally flipping them, which reduces the risk of overfitting. This is ok in our situation due to the small filter sizes used in convolution meaning the images will be distinct enough from each other for the model to realise. Furthermore random image filters such as random rotations, random zoom and random translations were applied to some images to further randomise the dataset and avoid overfitting. These were applied to existing images, rather than creating new ones. 

### Training of a CNN species classification model
A custom CNN was created in Python using the pre-processed and augmented images described above. The model uses Tensorflow and Keras for ease of use and deployment options. Iterative testing of model architectures led to the creation of a model with three blocks of 32, 64 and 128 filters respectively, each with dropout, followed by fully connected layers before the predictions. 

### CNN training results
Overall the best performing model resulted in accuracy of 0.86. **ADD MORE RESULTS**

## Deployment
### Our data pipeline
For deployment we propose a data pipeline where photos are sent to an object detection model (Megadetector) to identify animals in photos. Bounding boxes are created around animals, cars and people by the [Megadetector model](https://github.com/microsoft/CameraTraps/tree/main), which crops detected animals in the resulting image. 

The cropped animals are then passed through our CNN model for classification into the any of the classes mentionned above. Any classification with a confidence of below 0.7 is classed as 'unknown'. Results of images in a folder are summarised as a csv ready for further analysis. 


## Outputs
The main outputs from this project are a comprehensive **dataset** with a representative sample of species commonly found in the British countryside. There are arond 5,000 images per class from a range of habitats. A **species classification model** with an emphasis on reliable fox detection is the other main output. 


## Try it for yourself.
The files required are contained within this repository for a range of application, including training your own model and performing inference on your own data, including videos.

First, clone the repository:

```
git clone https://github.com/nickallen0/Fox-spot.git
```

Inside an Anaconda Prompt window, navigate to the repository in your files. 
```
cd "repo/folder/location"
```

Then create and activate a virtual environment with the required packages to run tensorflow on your NVIDIA GPU. It should default to CPU if you don't have one.
```
conda env create -f tensorflow-windows.yml
conda activate gpuenv-tensorflow
```
You can run **inference** on a folder of images by running the following:
```
python pipeline.py "C:/your/image/folder/path"
```
If you want to train your own model, you can adjust the parameters in the ***Claude-CNN.py*** file, which can then be run like this:
```
python CNN-inference.py
```

### Videos
If you have a video, you can use the **video_to_frames.py** script to extract frames at an interval rate of your choice. You can call it like this:
```
python video_to_frames.py input_directory output_directory --interval 5.0
```
then put the images through the pipeline script above.
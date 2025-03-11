# Fox-spot
Code repository for the Fox detection and classification project using AI. This project between Newcastle University and Perdix is running from September 2024 to May 2025. 

## Team

The Newcastle University team is made up of Roy Sanderson, Pete Robertson, Nicholas Allen and Ava Waine.

## Project plan
### Objectives
-   Create a dataset of camera trap images from the UK split into common species occuring in the British countryside
-   Create a species classification model for camera trap images

### Create a comprehensive dataset
A comprehensive dataset has been compiled from multiple sources and partners. The following table lists contributors and their contact details.

|Contributor |Contact  | Contribution|
--- | --- | ---|
|Apha|[Sarah Beatham](mailto:Sarah.Beatham@apha.gov.uk)|Annotated camera trap images of Fox, Boar and Muntjack in Loch Ness and Forest of Dean areas|
|Newcastle University|[Aileen Mill](mailto:aileen.mill@newcastle.ac.uk)||Camera trap images at Gosforth Nature Reserve over 4 years|
|Newcastle University|[Marion Pfeifer](mailto:Marion.Pfeifer@newcastle.ac.uk)|Camera trap images at Cockle Park over 2 years|
|Newcastle University|[Mark Whittingham](mailto:mark.whittingham@newcastle.ac.uk)|Camera trap platform access (Agouti) for mob grazing trials at Cockle Park, *not used at this time*|
|MammalWeb and Durham University|[Phil Stevens](mailto:philip.stephens@durham.ac.uk)|Agreed in principle to help for the cost of extracting images, open to collaboration|
|iNaturalist|[Online]()|Camera trap and other images
|Wildlife Insights|[Online]()|Camera trap and other images
|Badger Trust|[Victoria Coulton](mailto:victoria@badgertrust.org.uk)|Badger Camera trap videos
|CEH, Chornobyl Exclusion Zone, Ukraine|[Online](https://catalogue.ceh.ac.uk/documents/a657ffc3-8f62-458f-bcb7-30e116807174)|Camera trap images of Deer, Hares, Foxes, Squirrels, Wild Boar|


Bare images and images with non target species are required for a balanced dataset. The following ones will be included:
-	Fox
-	Badger
-	Deer
-	Squirrels
-	Birds
-	Dog
-	Cats
-	Human
-	Rabbit/Hare
-	Other

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
│   ├── fox
│   │   ├── source
│   ├── not fox
│   │   ├── species 1
│   │   │   ├── source
│   │   ├── species 2
```
#### Image sources
The images

### Use an object detection model (Megadetector) to identify animals in photos
Bounding boxes will be created by passing the images through the [Megadetector model](https://github.com/microsoft/CameraTraps/tree/main) and cropping the resulting image. 

### Create a CNN to classify detected animals into groups including foxes
A custom CNN will be created in Python to classify animal species into the classes listed above. 

## Outputs
The main outputs from this project are a comprehensive **dataset** with a representative sample of species commonly found in the British countryside. A **species classification model** with an emphasis on reliable fox detection is the other main output. 


## Getting started
Clone the repository

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
You can run **inference** on a folder of images by running the following
```
python pipeline.py "C:/your/image/folder/path"
```
If you want to train your own model, you can adjust the parameters in the ***Claude-CNN.py*** file, which can then be run like this:
```
python CNN-inference.py
```

## Videos
If you have a video, you can use the **video_to_frames.py** script to extract frames at an interval rate of your choice. You can call it like this:
```
python video_to_frames.py input_directory output_directory --interval 5.0
```
then put the images through the pipeline script above.
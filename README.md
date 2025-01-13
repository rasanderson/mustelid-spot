# Fox-spot
Repository for the Fox detection project using AI, between Newcastle University and Perdix. This project is running September 2024 to May 2025. 

## Team

The Newcastle University team is made up of Roy Sanderson, Pete Robertson, Nicholas Allen and Ava Waine.

## Project plan

#### Create a comprehensive dataset
A comprehensive dataset is being compiled from multiple sources including by webscraping iNaturalist, using Roboflow Universe images, gathering our own camera trap imagery across Cockle Park farm, Gosforth Nature Reserve and the Highbury allotments in Newcastle. An appeal for other images will be sent out to other partners including GNR volunteers, APHA and DEFRA.

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


#### Use an object detection model (Megadetector) to identify animals in photos
Bounding boxes will be created by passing the images through the [Megadetector model](https://github.com/microsoft/CameraTraps/tree/main) and cropping the resulting image. 

#### Create a CNN to classify detected animals into groups including foxes
A custom CNN will be created in Python to classify animal species into the classes listed above. 

## Outputs
The main outputs from this project are a comprehensive **dataset** with a representative sample of species commonly found in the British countryside. A **species classification model** with an emphasis on reliable fox detection is the other main output. 

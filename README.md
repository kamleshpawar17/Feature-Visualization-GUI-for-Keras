# Feature-Visualization-GUI-for-Keras
This is an analysis tool for visualization of feature maps learned by a deep learning network using a simple and easy to use Graphical User Interface (GUI) 

If you have keras working then just start the GUI and you are ready to go

Tested on **python 2.7.15**
#### Dependencies
Keras, scipy, pillow, Tkinter


## How to start GUI in python
From the directory where all the files downloaded from this repository is located, run startGUI.py
````
python startGUI.py
````
#### This is will instantiate a GUI

![FeatVisScreenshot](https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/Feat-Vis-GUI.png).


## Describtion of options available
**Browse:** Select a keras model file in *.h5 or *.hdf5 format through browse button

**Parse Model:** Click this to parses the selected model, this will create an image of the model in the current working directory as 'model_featvis.png'

**Choose a Layer:** Select a layer for which you want to visualize the feature maps

**Input Image:** Select an input image for which you want to visualize the features. Currently supported image formats: JPEG, PNG, TIFF, and BMP. See **Advanced Options** section for loading arbitrary image file format and preprocessing option. 

**Compute Features:** Click this to compute the features maps corresponding to the selected layer and the selected image

**Display:** Click this to display the computed feature maps 

**No of Rows/Columns to Display:** Select how many rows/columns you want to display per figure

**Close All Display:** Closes all the opened display windows

**Display Colormap:** Choose from a large number of color schemes to display images

**Save Features:** Click this to save the computed feature map as numpy array



## An Example of Features Visualized using Feat-Vis-GUI
### Input Image
<p align="center">
  <img src="https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/input.png">
</p>

### Visualization of Feature maps
<p align="center">
  <img src="https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/1.png">
</p>
<p align="center">
  <img src="https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/2.png">
</p>
<p align="center">
  <img src="https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/3.png">
</p>
<p align="center">
  <img src="https://github.com/kamleshpawar17/Feature-Visualization-GUI-for-Keras/blob/master/screenshots/4.png">
</p>

## Advanced options
The currently supported input image formats are JPEG, PNG, TIFF, and BMP. However, some of the application may need to read other image formats such as dicom/nifiti images in medical imaging.

Apart from reading an image one may want to perform a preprocessing of the input image such as normalizing mean and standard deviation.

Reading an arbitrary image format and including preprocessing requires just a little bit of coding. Below is the standard way on how to achieve this:
````
class FeatVis_child(FeatVis):
    def __init__(self, name="CNN Feature Visualization"):
        FeatVis.__init__(self, name)

    def imageReader(self):
        # self.imageName: This variable contains path to the input image
        # self.inputImage: This variable should contain the input image as nupy array after reading and preprocessing.
        # The shape of self.inputImage should be (height, width, channels)
````

#### An example of reading an image stored as a numpy array and preprocessing is demostrated below
````
from CnnVisualizationApp import FeatVis
import numpy as np

class FeatVis_child(FeatVis):
    def __init__(self, name="CNN Feature Visualization"):
        FeatVis.__init__(self, name)

    def imageReader(self):
        # Read Image
        self.inputImage = np.load(self.imageName)
        # Preprocess image 
        self.inputImage = (self.inputImage - np.mean(self.inputImage))/np.std(self.inputImage)


gui = FeatVis_child()
gui()

````

Save the above code as python file and run. This will instantiate a GUI that will use the **imageReader()** defined above to read the input image.

## Additional Notes
This GUI requires a trained keras model file in *.hdf5 or *.h5 format.

For the pretrained models from keras application you can generate keras model as:

````
from keras.applications.vgg16 import VGG16
model = VGG16(include_top=False, weights='imagenet')
model.save('model-vgg16.hdf5')
````








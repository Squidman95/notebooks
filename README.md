# Tensorflow Object Detection Walkthrough

## Installation Steps

<br />
<b>Video:</b> Go through the installation here https://www.youtube.com/watch?v=dZh_ps8gKgs
It's very important that you use the correct versions. You'll need Anaconda for Python. These versions are confirmed to work together:

<b>Anaconda Download (Jupyter):</b> "Anaconda3-2019.07-Windows-x86_64.exe" from https://repo.anaconda.com/archive.
It's okay to add to path despite the warning.

Visual Studio Community 2019 from https://my.visualstudio.com/Downloads?q=Visual%20Studio%202019
CUDA Only support specific versions, and 10.1 only supports 2019.

CUDA 10.1 from: https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal. You click Windows, x86_64, 10, exe (local)

CUDNN 7.6.5 from: https://developer.nvidia.com/rdp/cudnn-archive
Click "Download cuDNN v7.6.5 (November 5th, 2019), for CUDA 10.1" to show the options.
Download "cuDNN Library for Windows 10".
Go to the path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1
You can see that this folder contains a bin, include and lib folder, just like the zip you just downloaded.
Now copy the contents of "bin" and "include" of the zip file, into the folders with the same name. For "lib" you must go into the folder and into x64 and copy the contents of this into the x64 folder in lib in the other folder.

Download Protocol Buffers from: https://github.com/protocolbuffers/protobuf/releases
Select "protoc-3.19.4-win64.zip" or whichever version is there.
Extract the folder in C or in a folder close to C and add the path "C:\AdditionalPackages\protoc\bin" to your user variables

Clone the repo: https://github.com/tensorflow/models

<pre>
git clone https://github.com/tensorflow/models
</pre>

Open command prompt as admin and navigate into:

<pre>
\models\research
</pre>

Run commands:

<pre>
protoc object_detection/protos/*.proto --python_out=.
copy object_detection/packages/tf2/setup.py .
python -m pip install .
</pre>

<pre>

</pre>

<pre>

</pre>

<br/><br/>
<br/><br/>

# Steps

<br />
<b>Step 1.</b> Clone this repository: https://github.com/nicknochnack/TFODCourse and navigate into it in cmd
<br/><br/>
<b>Step 2.</b> Create a new virtual environment 
<pre>
python -m venv tfod
</pre> 
<br/>
<b>Step 3.</b> Activate your virtual environment
<pre>
source tfod/bin/activate # Linux
.\tfod\Scripts\activate # Windows 
</pre>
<br/>
<b>Step 4.</b> Install dependencies and add virtual environment to the Python Kernel
<pre>
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfodj
</pre>
<br/>

<b>Step 4.5</b> Open jupyter notebook by typing

<pre>
jupyter notebook
</pre>

A URL will be written in the command prompt, which you can copy into the browser and Jupyter will open. Alternatively you should be able to just select the browser if it asks which program to open.
If you click "new" in Jupyter notebook, you should now be able to see "tfod" under "Python 3". This proves that we have the virtual environment available in Jupyter.
<br/><br/>

## Collect and label images in Jupyter

### Collecting and setup:

<b>Step 5.</b> Collect images using the Notebook <a href="https://github.com/nicknochnack/TFODCourse/blob/main/1.%20Image%20Collection.ipynb">1. Image Collection.ipynb</a> - ensure you change the kernel to the virtual environment as shown below
<img src="https://i.imgur.com/8yac6Xl.png">
<br/>

<b>Step 5.5</b>
Run through steps 1-4 in the notebook. Step 4 should open a popup that will start taking pictures from your webcam and put them into the folders that was created in step 3, according to the labels that was given in step 2.
<br/><br/>

### Labelling

This part is about labelling. Labeling means drawing boxes around the things we need to be able to detect.
This tutorial uses labelImg, which is a package that can be found here:
https://github.com/tzutalin/labelImg
This covers several installation steps, but these are covered in the Notebook
<br/><br/>
<b>Label step (5)</b>
<br/>
Go through step 5 in the Notebook which will clone the git repo above and eventually a program should open. In this program, click "Open Dir" and navigate to the folder where the images were saved, e.g: <br/>
C:\Users\sebas\development\TutorialProjects\TFODCourse\Tensorflow\workspace\images\collectedimages
<br/>
Start with a folder, such as "livelong" and click "select folder" (vælg mappe). Sometimes it won't actually open the folder, but will ask you to select the folder again, but you can just click cancel, since all that matters is that you can see the files (images) from the folder in the bottom right corner under "File list".
You can now click "w" and mark the area of the image that you want to train it for (the hand in this example). Once an area has been marked, a pop-up should appear where you can give it a name. The name is important, and you just use this name for every image in this folder that you select. Example could be "LiveLong". Click "Ok". Save it (Ctrl S) and just hit enter here and an XML file will be saved that incapsulates the selected area for the image. Continue with all other images in the folder and keep naming them "LiveLong" and keep using the default name when saving the XML.

<b>Step 6.</b>

<br/>
Manually divide collected images into two folders train and test. So now all folders and annotations should be split between the following two folders. <br/>
\TFODCourse\Tensorflow\workspace\images\train<br />
\TFODCourse\Tensorflow\workspace\images\test <br />
You should have about 80% of your images in "train" and the last 20% in "test". It's important that the annotations (xml files) that belong to the images are copied with.
<br/><br/>

<b>Step 7.</b> Begin training process by opening <a href="https://github.com/nicknochnack/TFODCourse/blob/main/2.%20Training%20and%20Detection.ipynb">2. Training and Detection.ipynb</a>, this notebook will walk you through installing Tensorflow Object Detection, making detections, saving and exporting your model.
<br /><br/>

<b>Step 7.5</b>
Google "tensorflow model zoo tf2" or go to this link: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
Here we can see a bunch of different models. In the beginning of the Notebook we have open, there are 2 lines that we would have to change to select a different model.
<br/>

<pre>
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
</pre>

You could right click one of the other models in the git repo and copy the git repo and replace the second line with this link, and then replace the name of the model on the first line.
<br />
This is the TensorFlow model garden:
https://github.com/tensorflow/models
This is what is cloned in through the Notebook in step 8. When people refer to the "TensorFlow Object Detection API", they refer to everything inside the following folder in the repo: <br/>
research/object_detection <br/>

<b>Step 8.</b> During this process the Notebook will install Tensorflow Object Detection. You should ideally receive a notification indicating that the API has installed successfully at Step 8 with the last line stating OK.  
<img src="https://i.imgur.com/FSQFo16.png">
If not, resolve installation errors by referring to the <a href="https://github.com/nicknochnack/TFODCourse/blob/main/README.md">Error Guide.md</a> in this folder.
<br /> <br/>
<b>Step 9.</b> Once you get to step 6. Train the model, inside of the notebook, you may choose to train the model from within the notebook. I have noticed however that training inside of a separate terminal on a Windows machine you're able to display live loss metrics.
<img src="https://i.imgur.com/K0wLO57.png">
<br />
<b>Step 10.</b> You can optionally evaluate your model inside of Tensorboard. Once the model has been trained and you have run the evaluation command under Step 7. Navigate to the evaluation folder for your trained model e.g.

<pre> cd Tensorlfow/workspace/models/my_ssd_mobnet/eval</pre>

and open Tensorboard with the following command

<pre>tensorboard --logdir=. </pre>

Tensorboard will be accessible through your browser and you will be able to see metrics including mAP - mean Average Precision, and Recall.
<br />

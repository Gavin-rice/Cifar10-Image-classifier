# Cifar10-Image-classifier
A CNN project that uses the Keras dataset cifar10

You can upload photos into the images folder and then, by running 'python classifier.py' you can get an output of the top 5 classes that the model believes the image belongs too
in order from most to least confident.

## Installation and running tips

My environment is configured using miniconda and I used Python 3.9 in my virtual environment.
To run properly you should consider running this in a conda or miniconda environment. 

```bash
conda create --name myenv python==3.9
```
Now we activate the venv.
```bash
conda activate myenv
```
Alternatively you can use
```bash
source c:/Users/my_user/your_parent_directory/dev/Scripts/activate
```

Now you are able to install the dependencies.

### installing dependencies
Navigate to the directory containing your classifier.py file and then install the modules using pip.

```bash
pip install tensorflow
pip install matplotlib
pip install numpy
pip install scikit-image
```

After installations, you should be able to use the scipt to perform your own classifications.

# ART-ificial intelligence: CNN to predict the artist behind a painting

This is the README from my final project from Ironhack's data analytics bootcamp. My project is a neural network that predicts the artist that has done a painting, the dataset was obtained from Kaggle. This dataset contained a list of the 50 most influential artists, my neural network predicts for 11 artists out of the 50 existing in the dataset, because these are the ones that had larger amounts of paintings to train with. 


![Image](https://upload.wikimedia.org/wikipedia/commons/c/cd/VanGogh-starry_night.jpg)

---

## **Dataset**

https://www.kaggle.com/ikarus777/best-artworks-of-all-time


### :raising_hand: **ART-ificial intelligence** 
ART-ificial intelligence: this project is a mix of art and science, I have trained with almost 3000 artworks my CNN network, using as the pretrained architecture ResNet50. Finally, achieving an 83% of accuracy. 

### :running: **Demo-pipeline**
My code needs 2 input arguments, the folder where you have stored the artwork images' and the number of artworks that you want the network to predict 

### :computer: **Technology stack**
Python, Pandas, Scipy, Scikit-learn, Keras, Tensorflow, Matplotlib and Seaborn. The training of the network was developed in Colab-Pro as the images used where too large to work with in a standard laptop.

### :boom: **Core technical concepts and inspiration**
The inspiration comes from my personal experience being a person with a non-science background and doing a Data Analytics bootcamp, thus experiencing myself the mix of art and science.
 
This CNN predicts the author behind an artwork, there are other projects that predict the type of art where each artwork belongs, but my project is focused on predicting the painter and differenciating the style that each painter has.

### :wrench: **Configuration**
Requeriments: all libraries described in technology stack and computer with graphical memory to run the CNN

### :see_no_evil: **Usage**
Parameters: p-path_to_images_folder & n-number of predictions

### :file_folder: **Folder structure**
```
└── project
    ├── __trash__
    ├── .gitignore
    ├── .env
    ├── requeriments.txt
    ├── README.md
    ├── main_script.py
    ├── p_acquisition
    ├── p_analysis
    ├── p_reporting
    ├── notebooks
    │   ├── acquisition.ipynb
    │   ├── analysis.ipynb
    │   ├── training.ipynb
    │   └── demo.ipynb
    ├── package1
    │   ├── acquisition.py
    │   ├── plots.py
    │   └── demo.py
    └── data
        ├── raw
        ├── processed
        └── results
```

### :shit: **ToDo**
Next steps: improve actual accuracy, include new painters and their artworks in the network.

### :information_source: **Further info**
Credits: kernel from the dataset in Kaggle (https://www.kaggle.com/supratimhaldar/deepartist-identify-artist-from-art).

### :love_letter: **Contact info**
Mail: juliafroch@gmail.com Getting help, getting involved, hire me please.

---


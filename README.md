# Segmentation of Overlapping Pixels in Multi-Spectral Document Images by Statistical Techniques

The utilization of hyperspectral data has emerged as a potent tool in the field of document image processing, presenting unparalleled levels of information and insights across diverse applications. Traditionally, tasks like detecting document forgery, estimating ink age, and extracting text from deteriorated or damaged documents have heavily relied on RGB images. However, with the growing necessity for more refined tasks, such as multiclass signature segmentation, there is a pressing demand for a more comprehensive source of information to prevent data loss. This implementation introduces a novel approach for segmenting signatures with class overlap in hyperspectral document images. Unlike conventional RGB methods, our proposed technique is expressly tailored for hyperspectral data. To validate the efficacy of our methodology, we subject the handwritten signatures segmented using our approach to rigorous testing against ground truth images. The results affirm that our method not only proves effective but also demonstrates efficiency and precision in the challenging task of segmenting signatures with class overlap in hyperspectral document images.

Our implemented pipeline is outlined in the flow below
![image](https://github.com/user-attachments/assets/95acc682-39be-4e46-b3e5-699f77e73fe7)
Our Data consists of HSI document images with pixels belonging to three classes i.e.

* Printed Text: Pixels comprising the printed text.
* Handwritten Signature: Pixels comprising the handwritten signature.
* Overlapping Pixels:e: Pixels on the intersections of handwritten and printed text pixels.
* Background Pixels: Pixels comprising the background i.e. not belonging to any of the above class.

A few samples of the HSI document images dataset are below

![image](https://github.com/user-attachments/assets/39db0098-3f96-4f83-b549-6f897921c0e7)

Each Sample in our dataset consists of 240 channels making each pixel to an array of 240 indices. The above formulated pipeline focuses on segmenting the handwritten pixels along with the overlapping pixels belonging to the handwritten signature from the printed text and extracting the complete handwritten signature without any data loss at the overlappling points. The classes present in the dataset are also visualized in the figure below.

# Pixel Classes present in the HSI document image cube
![image](https://github.com/user-attachments/assets/fef8cfd2-80fc-40c2-92da-5a40afe54c90)

The image (A) in the figure represents pixels comprising of the printed text pixels class, the image (B) represents the pixels comprising of the handwritten pixels class and image (C) comprises of pixels comprising the overlapping pixels class for this particular HSI sample.

# Implementation

To get started with the implementation

* Clone the repository:
`git clone 'https://github.com/Document-Data-Analyst/Signature-Segmentation-in-HSI-Images.git`

* Download the dataset: Download and unzip the the HSI dataset from https://drive.google.com/drive/folders/1FGNaeoz-N8xBWcCXx8ukU28B_0LU1NBR?usp=sharing into the HSI_document_images_signature_segmentation/data/ folder

## Python Implementation

* Install the required dependencies:
pip install -r requirements.txt

* Run the .py file: **Before running the python file update the paths to the dataset files in the code**
'!py thesis_python_code.py'

## Jupyter Notebook Implementation

* Download the ThesisCodeFinal.ipynb notebook and update the paths to the data based on the folder the data is unzipped.
* Run the .ipynb file to generate the step by step results.

# Results
The step-by-step visualization of the handwritten signature extraction process is depicted in the figure below.

![image](https://github.com/user-attachments/assets/10a79347-6c06-4dd9-b7c2-6c7918bd2b69)

Results for signature segmentation for multiple HSI samples and their evaluation metrics are the following

![image](https://github.com/user-attachments/assets/544545ff-34f4-4e9f-836c-4b3a08d2656e)









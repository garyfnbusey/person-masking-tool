Person Masking Tool
A GUI-based Python application that identifies and isolates a specific person in images, masking out everything else. Built on SAM-HQ and enhanced with face recognition and fallback DNN-based detection.

🚀 Features
✅ Identify a specific human in images using a reference photo

✅ Mask everything except the target person

✅ Save output as .png with transparency

✅ Batch process entire image folders

✅ GPU acceleration (CUDA) support

✅ Fallback face detection using OpenCV DNN (for profile/side views)

✅ Simple GUI for non-technical users

📂 Folder Structure
Your img directory must contain:

graphql
Copy
Edit
img/
├── reference/       # contains ONE image of the target person
├── image1.jpg
├── image2.png
├── ...
Output images will be saved with the name pattern: originalname_masked.png.

🖼️ Reference Image
The app uses the image inside img\reference to detect the target person. Only one image should be present in this folder.

🧠 Models Used
SAM-HQ (ViT-H) — for segmentation

face_recognition — to match the person in the reference image

OpenCV DNN (ResNet SSD) — as a fallback for profile face detection

Ensure the following files are present:

sam_hq_vit_h.pth — Download from SAM-HQ GitHub

res10_300x300_ssd_iter_140000.caffemodel and deploy.prototxt — Download from OpenCV

🧰 Requirements
Python 3.8+

torch (with CUDA if using GPU)

face_recognition

numpy

opencv-python

Pillow

tkinter

Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

txt
Copy
Edit
torch
face_recognition
opencv-python
numpy
Pillow
🖥️ How to Use
Launch the tool:

bash
Copy
Edit
python person_masking_tool.py
In the GUI:

Select your img directory (must contain a reference subfolder).

Optionally, browse to the SAM-HQ checkpoint (.pth file).

Choose your output folder.

Click Start.

Processed .png images will appear in the output directory with transparent backgrounds.

🛠 Default Configuration
SAM-HQ path defaults to:

pgsql
Copy
Edit
C:\Users\LocalAdmin\Downloads\GitHub\un-stable-diffusion\stable-diffusion-webui\extensions\sd-webui-segment-anything\models\sam\sam_hq_vit_h.pth
DNN fallback model paths:

Copy
Edit
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
These must be in the same directory as the script or modify the paths accordingly.

🙋 Support & Contributions
I may only check a few times per year... but feel free to open issues or submit pull requests to enhance functionality, fix bugs, or improve performance.

# Person Masking Tool

`batch_mask_img.py`
A GUI-based Python application that identifies and isolates a specific person in images, masking out everything else. Built on **SAM-HQ** and enhanced with face recognition and fallback DNN-based detection.

`batch_mask_img_all.py`
same tool but no reference used so all humans are masked out instead.

This tool is created for personal use. There are a couple of files which are needed:

1. You will need to download sam_hq_vit_h.pth and select it in the GUI.

SAM-HQ path defaults to:

`C:\Users\LocalAdmin\Downloads\GitHub\un-stable-diffusion\stable-diffusion-webui\extensions\sd-webui-segment-anything\models\sam\sam_hq_vit_h.pth`

DNN fallback model paths (for detected side view faces)
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
These must be in the same directory as the script, or you can modify the script to change the paths.



---

## 🚀 Features

- ✅ Identify a specific human in images using a reference photo  
- ✅ Mask everything *except* the target person  
- ✅ Save output as `.png` with transparency  
- ✅ Batch process entire image folders  
- ✅ GPU acceleration (CUDA) support  
- ✅ Fallback face detection using OpenCV DNN (for profile/side views)  
- ✅ Simple GUI for non-technical users

---

## 📂 Folder Structure

Your `img` directory must contain:

img/
├── reference/ # contains ONE image of the target person
├── image1.jpg
├── image2.png
├── ...


Output images will be saved with the name pattern: `originalname_masked.png`.

---

## 🖼️ Reference Image

The app uses the image inside `img\reference` to detect the target person. Only one image should be present in this folder.

---

## 🧠 Models Used

- **SAM-HQ (ViT-H)** — for segmentation  
- **face_recognition** — to match the person in the reference image  
- **OpenCV DNN** (ResNet SSD) — as a fallback for profile face detection

Ensure the following files are present:

- `sam_hq_vit_h.pth` — [Download from SAM-HQ GitHub](https://github.com/SysCV/sam-hq)
- `res10_300x300_ssd_iter_140000.caffemodel` and `deploy.prototxt` — [Download from OpenCV](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

---

## 🧰 Requirements

- Python 3.8+
- torch (with CUDA if using GPU)
- face_recognition
- numpy
- opencv-python
- Pillow
- tkinter

Install dependencies:

```bash
pip install -r requirements.txt

torch
face_recognition
opencv-python
numpy
Pillow
```

🖥️ How to Use
1. Launch the tool:

```
python person_masking_tool.py
```

2. In the GUI:

Select your img directory (must contain a reference subfolder).

Optionally, browse to the SAM-HQ checkpoint (.pth file).

3. Choose your output folder.

4. Click Start.

Processed .png images will appear in the output directory with transparent backgrounds.


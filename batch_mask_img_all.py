# Looks in img\reference\ for the only reference image (any extension).
# Iterates over all .png, .jpg, and .jpeg files in img\.
# need the following:
# sam_HQ
# deploy.prototxt
# res10_300x300_ssd_iter_140000.caffemodel
import os
import glob
import cv2
import face_recognition
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry as sam_hq_model_registry, SamPredictor
import tkinter as tk
from tkinter import filedialog, messagebox
import threading

stop_event = threading.Event()

# --- DEFAULTS ---
MODEL_TYPE = 'vit_h'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_SAM_PATH = r'C:\\Users\\LocalAdmin\\Downloads\\GitHub\\un-stable-diffusion\\stable-diffusion-webui\\extensions\\sd-webui-segment-anything\\models\\sam\\sam_hq_vit_h.pth'
DEFAULT_DNN_PROTO = 'deploy.prototxt'
DEFAULT_DNN_MODEL = 'res10_300x300_ssd_iter_140000.caffemodel'

# --- GUI Functions ---
def load_sam(checkpoint_path):
    sam = sam_hq_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    return SamPredictor(sam)

def apply_mask_and_save(image_path, mask, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_rgb.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    image_rgba = np.dstack((image_rgb, mask.astype(np.uint8) * 255))
    Image.fromarray(image_rgba).save(output_path)

def fallback_face_detection(image_bgr, log_fn):
    net = cv2.dnn.readNetFromCaffe(DEFAULT_DNN_PROTO, DEFAULT_DNN_MODEL)
    h, w = image_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append((startY, endX, endY, startX))
    if boxes:
        log_fn("Fallback DNN faces used.")
    return boxes

def process_images(img_dir, checkpoint_path, output_dir, log_fn):
    try:
        predictor = load_sam(checkpoint_path)
        os.makedirs(output_dir, exist_ok=True)

        valid_ext = ('.png', '.jpg', '.jpeg')
        image_paths = [f for f in glob.glob(os.path.join(img_dir, '*')) if f.lower().endswith(valid_ext) and os.path.isfile(f)]

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            output_filename = os.path.splitext(filename)[0] + '_masked.png'
            output_path = os.path.join(output_dir, output_filename)

            if os.path.exists(output_path):
                log_fn(f"Skipping {filename}, already processed.")
                continue

            log_fn(f"Processing {filename}...")
            image = face_recognition.load_image_file(image_path)
            face_locations = face_recognition.face_locations(image)
            image_bgr = cv2.imread(image_path)

            if not face_locations:
                face_locations = fallback_face_detection(image_bgr, log_fn)
                if not face_locations:
                    log_fn(f"No faces found in {filename}.")
                    continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(image_rgb)

            all_masks = []
            for box in face_locations:
                top, right, bottom, left = box
                expand_y = int((bottom - top) * 0.6)
                expand_x = int((right - left) * 0.2)
                top_exp = max(0, top - expand_y)
                bottom_exp = min(image_rgb.shape[0], bottom + int(0.2 * expand_y))
                left_exp = max(0, left - expand_x)
                right_exp = min(image_rgb.shape[1], right + expand_x)
                input_box = np.array([left_exp, top_exp, right_exp, bottom_exp])
                center_point = np.array([(left + right) // 2, (top + bottom) // 2])

                masks, _, _ = predictor.predict(
                    box=input_box,
                    point_coords=np.array([center_point]),
                    point_labels=np.array([1]),
                    multimask_output=False
                )
                all_masks.append(masks[0])

            if all_masks:
                combined_mask = np.any(np.stack(all_masks), axis=0)
                apply_mask_and_save(image_path, combined_mask, output_path)
                log_fn(f"Saved: {output_filename}")

        log_fn("\nProcessing complete.")
    except Exception as e:
        log_fn(f"Error: {e}")

# --- GUI ---
def run_gui():
    def start_processing():
        img_dir = img_entry.get()
        checkpoint = checkpoint_entry.get() or DEFAULT_SAM_PATH
        output_dir = output_entry.get()
        if not os.path.isdir(img_dir):
            messagebox.showerror("Invalid Path", "Please select a valid image directory.")
            return
        if not os.path.isfile(checkpoint):
            messagebox.showerror("Invalid Path", f"SAM-HQ checkpoint not found at:\n{checkpoint}")
            return
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        log_box.delete(1.0, tk.END)
        threading.Thread(target=process_images, args=(img_dir, checkpoint, output_dir, lambda msg: log_box.insert(tk.END, msg + "\n")), daemon=True).start()

    def browse_dir(entry):
        path = filedialog.askdirectory()
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def browse_file(entry):
        path = filedialog.askopenfilename(filetypes=[("Model Checkpoint", "*.pth")])
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    root = tk.Tk()
    root.title("Person Masking Tool - SAM-HQ")

    tk.Label(root, text="Image Folder:").grid(row=0, column=0, sticky='w')
    img_entry = tk.Entry(root, width=50)
    img_entry.grid(row=0, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_dir(img_entry)).grid(row=0, column=2)

    tk.Label(root, text="SAM-HQ Checkpoint (.pth):").grid(row=1, column=0, sticky='w')
    checkpoint_entry = tk.Entry(root, width=50)
    checkpoint_entry.insert(0, DEFAULT_SAM_PATH)
    checkpoint_entry.grid(row=1, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_file(checkpoint_entry)).grid(row=1, column=2)

    tk.Label(root, text="Output Folder:").grid(row=2, column=0, sticky='w')
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=2, column=1)
    tk.Button(root, text="Browse", command=lambda: browse_dir(output_entry)).grid(row=2, column=2)

    tk.Button(root, text="Start", command=start_processing, bg="green", fg="white").grid(row=3, column=1, pady=10)

    log_box = tk.Text(root, width=80, height=20)
    log_box.grid(row=4, column=0, columnspan=3)

    root.mainloop()

if __name__ == '__main__':
    run_gui()
#!/usr/bin/env python3

import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

import glob
import csv

import threading
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetPowerUsage


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATASET_DIR = "imagenet500/images/"
LABELS_PATH = 'imagenet500/words.txt'
VAL_LABELS_PATH = 'imagenet500/val.txt'

FUSED_MODEL_PATH = "test_1_fused_model.pt"

TEST_A0_RESULT_CSV_PATH         = "test_A0_gpu_results.csv"
TEST_A0_POWER_RESULT_CSV_PATH   = "test_A0_gpu_power_results.csv"

TEST_B1_RESULT_CSV_PATH         = "test_B1_gpu_results.csv"
TEST_B1_POWER_RESULT_CSV_PATH   = "test_B1_gpu_power_results.csv"

TEST_B2_RESULT_CSV_PATH         = "test_B2_gpu_results.csv"
TEST_B2_POWER_RESULT_CSV_PATH   = "test_B2_gpu_power_results.csv"

_gpu_log_thread = None
_gpu_log_stop = None


# -----------------------------------------------------------------------------
# MODEL CLASSES
# -----------------------------------------------------------------------------
class ResnetFusionSaver:
    def __init__(self, output_path, device):
        self.device = device
        self.model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        fusion_dict = {
            'model1': self.model1,
            'model2': self.model2
        }
        torch.save(fusion_dict, output_path)
        print(f"Resnet Fusion model saved at: {output_path}")

class ResnetFusionLoader(nn.Module):
    """Loads two ResNet50 models from a saved fusion dictionary."""

    def __init__(self, fusion_model_path, device):
        super().__init__()
        self.device = device
        fusion_dict = torch.load(fusion_model_path, map_location=device, weights_only=False)
        self.model1 = fusion_dict['model1']
        self.model2 = fusion_dict['model2']

        self.model1.eval()
        self.model2.eval()

    def forward(self, img1, img2):
        return self.model1(img1), self.model2(img2)

    def infer_model_1(self, img):
        return self.model1(img)

    def infer_model_2(self, img):
        return self.model2(img)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_labels(path):
    """Load labels from text file."""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_val_labels(path):
    """Load validation labels mapping."""
    val_mapping = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                val_mapping[parts[0]] = int(parts[1])
    return val_mapping

def preprocess_pil(img):
    """PIL-based preprocessing for PyTorch models."""

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return preprocess(img).unsqueeze(0)

def postprocess_topk(logits, k=5):
    """Convert logits to probabilities and return top-k."""
    probs = torch.nn.functional.softmax(logits, dim=1)
    topk_probs, topk_indices = torch.topk(probs, k, dim=1)
    return topk_indices.cpu().numpy()[0], topk_probs.cpu().numpy()[0]


# ---------------------------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------------------------
def process_images_gpu(image_paths, labels, val_mapping, model, device, results_csv):
    """Process images on GPU and generate detailed metrics."""

    with open(results_csv, "w", newline="") as f:

        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "pred1", "prob1", "gt1", "correct1",
            "pred2", "prob2", "gt2", "correct2",
            "load_time", "preprocess_time",
            "inference_time", "postprocess_time", "total_time"
        ])

        for i in range(0, len(image_paths), 2):
            if i + 1 >= len(image_paths):
                print(f"Skipping {image_paths[i]} (no pair)")
                break

            total_time_start = time.perf_counter()

            # -------------------------------
            # LOAD IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img1_path, img2_path = image_paths[i], image_paths[i + 1]
            img1_name = os.path.basename(img1_path)
            img2_name = os.path.basename(img2_path)

            # Get ground truth labels
            gt1 = val_mapping.get(img1_name, -1)
            gt2 = val_mapping.get(img2_name, -1)
            load_time = time.perf_counter() - t_start

            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # -------------------------------
            # PREPROCESS IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img1_tensor = preprocess_pil(img1).to(device)
            img2_tensor = preprocess_pil(img2).to(device)
            preprocess_time = time.perf_counter() - t_start

            # -------------------------------
            # INFERENCE
            # -------------------------------
            t_start = time.perf_counter()
            with torch.no_grad():
                output1, output2 = model(img1_tensor, img2_tensor)
            inference_time = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()
            idx1, prob1 = postprocess_topk(output1, k=1)
            idx2, prob2 = postprocess_topk(output2, k=1)
            postprocess_time = time.perf_counter() - t_start

            # -------------------------------
            # ACCURACY CALCULATION
            # -------------------------------
            correct1 = 1 if idx1[0] == gt1 else 0
            correct2 = 1 if idx2[0] == gt2 else 0

            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV
            # -------------------------------
            writer.writerow([
                labels[idx1[0]], float(prob1[0]), gt1, correct1,
                labels[idx2[0]], float(prob2[0]), gt2, correct2,
                load_time, preprocess_time,
                inference_time, postprocess_time, total_time
            ])


def TestB1(image_paths, labels, val_mapping, results_csv):
    """ Run TestB1 """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResnetFusionLoader(FUSED_MODEL_PATH, device=device)

    with open(results_csv, "w", newline="") as f:

        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2", "upload_time_1", "upload_time_2",
            "inference_time",
            "download_time_1", "download_time_2", "postprocess_time_1", "postprocess_time_2",
            "total_time"
        ])

        for i in range(0, len(image_paths)):

            print(i)

            img_path = image_paths[i]
            img_name = os.path.basename(img_path)
            gt = val_mapping.get(img_name, -1)
            gt_name = labels[gt] if gt != -1 else "unknown"

            total_time_start = time.perf_counter()

            # -------------------------------
            # LOAD IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img_1 = Image.open(img_path).convert('RGB')
            load_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            img_2 = Image.open(img_path).convert('RGB')
            load_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # PREPROCESS IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img1_tensor = preprocess_pil(img_1).to(device)
            preprocess_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            img2_tensor = preprocess_pil(img_2).to(device)
            preprocess_time_2 = time.perf_counter() - t_start


            # -------------------------------
            # UPLOAD
            # -------------------------------
            t_start = time.perf_counter()
            img1_tensor = img1_tensor.to(device)
            upload_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            img2_tensor = img2_tensor.to(device)
            upload_time_2 = time.perf_counter() - t_start


            # -------------------------------
            # INFERENCE
            # -------------------------------
            t_start = time.perf_counter()

            with torch.no_grad():
                output1, output2 = model(img1_tensor, img2_tensor)

            inference_time = time.perf_counter() - t_start

            # -------------------------------
            # DOWNLOAD
            # -------------------------------
            t_start = time.perf_counter()
            output1 = output1.cpu()
            download_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            output2 = output2.cpu()
            download_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()
            idx1, prob1 = postprocess_topk(output1, k=1)
            pred1_name = labels[idx1[0]] if idx1[0] < len(labels) else "unknown"
            postprocess_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            idx2, prob2 = postprocess_topk(output2, k=1)
            pred2_name = labels[idx1[0]] if idx1[0] < len(labels) else "unknown"
            postprocess_time_2 = time.perf_counter() - t_start

            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV
            # -------------------------------

            '''
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2", "upload_time_1", "upload_time_2",
            "inference_time",
            "download_time_1", "download_time_2", "postprocess_time_1", "postprocess_time_2",
            "total_time"
            '''

            writer.writerow([
                img_name,
                int(idx1[0]), pred1_name, float(prob1[0]),
                int(idx2[0]), pred2_name, float(prob2[0]),
                gt, gt_name,
                load_time_1, load_time_2, preprocess_time_1, preprocess_time_1, upload_time_1, upload_time_2,
                inference_time,
                download_time_1, download_time_2, postprocess_time_1, postprocess_time_2,
                total_time
            ])


def TestB2(image_paths, labels, val_mapping, results_csv):
    """ Run TestB2 """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResnetFusionLoader(FUSED_MODEL_PATH, device=device)

    with open(results_csv, "w", newline="") as f:

        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2", "upload_time_1", "upload_time_2",
            "inference_time_1", "inference_time_2",
            "download_time_1", "download_time_2", "postprocess_time_1", "postprocess_time_2",
            "total_time"
        ])

        for i in range(0, len(image_paths)):

            print(i)

            # gt
            img_path = image_paths[i]
            img_name = os.path.basename(img_path)
            gt = val_mapping.get(img_name, -1)
            gt_name = labels[gt] if gt != -1 else "unknown"

            total_time_start = time.perf_counter()

            # -------------------------------
            # LOAD IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img_1 = Image.open(img_path).convert('RGB')
            load_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            img_2 = Image.open(img_path).convert('RGB')
            load_time_2 = time.perf_counter() - t_start

            # ------------------------------------------------------- MODEL 0
            # -------------------------------
            # PREPROCESS 0 IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img_tensor_1 = preprocess_pil(img_1).to(device)
            preprocess_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # UPLOAD
            # -------------------------------
            t_start = time.perf_counter()
            img_tensor_1 = img_tensor_1.to(device)
            upload_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # INFERENCE
            # -------------------------------
            t_start = time.perf_counter()
            with torch.no_grad():
                output_1 = model.infer_model_1(img_tensor_1)
            inference_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # DOWNLOAD
            # -------------------------------
            t_start = time.perf_counter()
            output_1 = output_1.cpu()
            download_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()
            idx1, prob1 = postprocess_topk(output_1, k=1)
            pred1_name = labels[idx1[0]] if idx1[0] < len(labels) else "unknown"
            postprocess_time_1 = time.perf_counter() - t_start

            # ------------------------------------------------------- MODEL 2
            # -------------------------------
            # PREPROCESS 0 IMAGES
            # -------------------------------
            t_start = time.perf_counter()
            img_tensor_2 = preprocess_pil(img_2).to(device)
            preprocess_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # UPLOAD
            # -------------------------------
            t_start = time.perf_counter()
            img_tensor_2 = img_tensor_2.to(device)
            upload_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # INFERENCE
            # -------------------------------
            t_start = time.perf_counter()
            with torch.no_grad():
                output_2 = model.infer_model_2(img_tensor_2)
            inference_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # DOWNLOAD
            # -------------------------------
            t_start = time.perf_counter()
            output_2 = output_2.cpu()
            download_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()
            idx2, prob2 = postprocess_topk(output_2, k=1)
            pred2_name = labels[idx2[0]] if idx2[0] < len(labels) else "unknown"
            postprocess_time_2 = time.perf_counter() - t_start

            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV
            # -------------------------------

            '''
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2", "upload_time_1", "upload_time_2",
            "inference_time_1", "inference_time_2",
            "download_time_1", "download_time_2", "postprocess_time_1", "postprocess_time_2",
            "total_time"
            '''

            writer.writerow([
                img_name,
                int(idx1[0]), pred1_name, float(prob1[0]),
                int(idx2[0]), pred2_name, float(prob2[0]),
                gt, gt_name,
                load_time_1, load_time_2, preprocess_time_1, preprocess_time_2, upload_time_1, upload_time_2,
                inference_time_1, inference_time_2,
                download_time_1, download_time_2, postprocess_time_1, postprocess_time_2,
                total_time
            ])


# ---------------------------------------------------------------------------
# POWER LOG
# ---------------------------------------------------------------------------
def gpu_power_logger(csv_path, stop_event, interval=0.01):
    """Thread logger de energia GPU -> CSV"""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    start = time.time()
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "power_watts"])
        while not stop_event.is_set():
            now = time.time() - start
            power_w = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            writer.writerow([now, power_w])
            f.flush()
            time.sleep(interval)
    nvmlShutdown()

def start_gpu_power_log(csv_path):
    """Arranca o logger em thread separada"""
    global _gpu_log_thread, _gpu_log_stop
    _gpu_log_stop = threading.Event()
    _gpu_log_thread = threading.Thread(
        target=gpu_power_logger, args=(csv_path, _gpu_log_stop), daemon=True
    )
    _gpu_log_thread.start()
    print(f"[DEBUG] GPU power logging started -> {csv_path}")

def stop_gpu_power_log():
    """Para o logger e espera thread acabar"""
    global _gpu_log_thread, _gpu_log_stop
    if _gpu_log_stop:
        _gpu_log_stop.set()
    if _gpu_log_thread:
        _gpu_log_thread.join()
        print("[DEBUG] GPU power logging stopped")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    # BUILD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create, save model and load model
    ResnetFusionSaver(FUSED_MODEL_PATH, device)


    # Start power logger
    #start_gpu_power_log(TEST_B1_POWER_RESULT_CSV_PATH)
    start_gpu_power_log(TEST_B2_POWER_RESULT_CSV_PATH)

    # load dataset, labels and validation mapping
    image_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.JPEG")))
    labels = load_labels(LABELS_PATH)
    val_mapping = load_val_labels(VAL_LABELS_PATH)
    print(f"Found {len(image_files)} images")

    #TestB1(image_files, labels, val_mapping, TEST_B1_RESULT_CSV_PATH)
    TestB2(image_files, labels, val_mapping, TEST_B2_RESULT_CSV_PATH)

    # Stop power log
    stop_gpu_power_log()


if __name__ == '__main__':
    main()
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
# CONFIGURATION
# -----------------------------------------------------------------------------
DATASET_DIR = "imagenet500/images/"
LABELS_PATH = 'imagenet500/words.txt'
VAL_LABELS_PATH = 'imagenet500/val.txt'

FUSED_MODEL_PATH = "test_2_fused_model.pt"
FUSED_MODEL_ONNX_PATH = "test_2_fused_model.onnx"

TEST_B3_RESULT_CSV_PATH         = "test_B3_gpu_results.csv"
TEST_B3_POWER_RESULT_CSV_PATH   = "test_B3_gpu_power_results.csv"

_gpu_log_thread = None
_gpu_log_stop = None


# -----------------------------------------------------------------------------
# MODEL CLASSES
# -----------------------------------------------------------------------------
class Resnet50FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model2 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.dummy_fc = nn.ReLU(inplace=True)

    def forward(self, img1, img2):
        out1 = self.model1(img1)                        # [B, 1000]
        out2 = self.model2(img2)                        # [B, 1000]
        concat = torch.cat((out1, out2), dim=1)  # [B, 2000]
        fused = self.dummy_fc(concat)                   # [B, 2000]
        return fused

class Resnet50FusionSaver:
    def __init__(self, output_path_pt, output_path_onnx):
        fusion_model = Resnet50FusionModel()
        torch.save(fusion_model, output_path_pt)
        print(f"Resnet Fusion model (.pt) saved at: {output_path_pt}")

        dummy_img1 = torch.randn(1, 3, 224, 224)
        dummy_img2 = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
        fusion_model,
        (dummy_img1, dummy_img2),
        output_path_onnx,
        input_names=["img1", "img2"],
        output_names=["fused_output"],
        opset_version=12
        )
        print(f"Resnet Fusion model (.onnx) saved at: {output_path_onnx}")

class Resnet50FusionLoader:
    def __init__(self, fusion_model_path, device):
        self.device = device
        self.model = torch.load(fusion_model_path, map_location=device, weights_only=False)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def infer(self, img1, img2):
        return self.model(img1, img2)


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
    # Extrair as saídas individuais dos modelos (primeiras 1000 classes para cada modelo)
    output1 = logits[:, :1000]  # Saída do primeiro ResNet50
    output2 = logits[:, 1000:]  # Saída do segundo ResNet50

    probs1 = torch.nn.functional.softmax(output1, dim=1)
    probs2 = torch.nn.functional.softmax(output2, dim=1)

    topk_probs1, topk_indices1 = torch.topk(probs1, k, dim=1)
    topk_probs2, topk_indices2 = torch.topk(probs2, k, dim=1)

    return (topk_indices1.cpu().numpy()[0], topk_probs1.cpu().numpy()[0],
            topk_indices2.cpu().numpy()[0], topk_probs2.cpu().numpy()[0])


# ---------------------------------------------------------------------------
# PROCESS IMAGES
# ---------------------------------------------------------------------------
def process_images_gpu(image_paths, labels, val_mapping, model_loader, device, results_csv):
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
            fused_output = model_loader.infer(img1_tensor, img2_tensor)
            inference_time = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()
            idx1, prob1, idx2, prob2 = postprocess_topk(fused_output, k=1)
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

def TestB3(image_paths, labels, val_mapping, results_csv):
    """ Run TestB3 """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_loader = Resnet50FusionLoader(FUSED_MODEL_PATH, device)

    with open(results_csv, "w", newline="") as f:

        writer = csv.writer(f)
        # CSV header
        writer.writerow([
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2",
            "upload_time","inference_time",
            "download_time", "postprocess_time",
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
            img1_tensor = preprocess_pil(img_1)
            preprocess_time_1 = time.perf_counter() - t_start

            t_start = time.perf_counter()
            img2_tensor = preprocess_pil(img_2)
            preprocess_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # UPLOAD
            # -------------------------------
            t_start = time.perf_counter()
            img1_tensor = img1_tensor.to(device)
            img2_tensor = img2_tensor.to(device)
            upload_time = time.perf_counter() - t_start

            # -------------------------------
            # INFERENCE
            # -------------------------------
            t_start = time.perf_counter()
            fused_output = model_loader.infer(img1_tensor, img2_tensor)
            inference_time = time.perf_counter() - t_start

            # -------------------------------
            # DOWNLOAD
            # -------------------------------
            t_start = time.perf_counter()
            fused_output = fused_output.cpu()
            download_time = time.perf_counter() - t_start

            # -------------------------------
            # POST PROCESSING
            # -------------------------------
            t_start = time.perf_counter()

            idx1, prob1, idx2, prob2 = postprocess_topk(fused_output, k=1)
            pred1_name = labels[idx1[0]] if idx1[0] < len(labels) else "unknown"
            pred2_name = labels[idx2[0]] if idx2[0] < len(labels) else "unknown"

            postprocess_time = time.perf_counter() - t_start
            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV
            # -------------------------------

            '''
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "load_time_2", "preprocess_time_1", "preprocess_time_2",
            "upload_time","inference_time",
            "download_time", "postprocess_time",
            "total_time"
            '''

            writer.writerow([
                img_name,
                int(idx1[0]), pred1_name, float(prob1[0]),
                int(idx2[0]), pred2_name, float(prob2[0]),
                gt, gt_name,
                load_time_1, load_time_2, preprocess_time_1, preprocess_time_2,
                upload_time, inference_time,
                download_time, postprocess_time,
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
    Resnet50FusionSaver(FUSED_MODEL_PATH, FUSED_MODEL_ONNX_PATH)


    # Start power logger
    start_gpu_power_log(TEST_B3_POWER_RESULT_CSV_PATH)

    # load dataset, labels and validation mapping
    image_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.JPEG")))
    labels = load_labels(LABELS_PATH)
    val_mapping = load_val_labels(VAL_LABELS_PATH)
    print(f"Found {len(image_files)} images")

    TestB3(image_files, labels, val_mapping, TEST_B3_RESULT_CSV_PATH)

    # Stop power log
    stop_gpu_power_log()


if __name__ == '__main__':
    main()
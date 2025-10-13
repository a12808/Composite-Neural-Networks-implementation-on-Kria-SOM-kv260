#!/usr/bin/env python3

import os
import time
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights
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

RESNET50_0_MODEL_PATH           = "test_5_resnet50_0_model.pt"
RESNET50_0_MODEL_ONNX_PATH      = "test_5_resnet50_0_model.onnx"

RESNET50_1_MODEL_PATH           = "test_5_resnet50_1_model.pt"
RESNET50_1_MODEL_ONNX_PATH      = "test_5_resnet50_1_model.onnx"

VGG16_MODEL_PATH                = "test_5_vgg16_model.pt"
VGG16_MODEL_ONNX_PATH           = "test_5_vgg16_model.onnx"

TEST_A0_RESULT_CSV_PATH         = "test_A0_gpu_results.csv"
TEST_A0_POWER_RESULT_CSV_PATH   = "test_A0_gpu_power_results.csv"
TEST_A1_RESULT_CSV_PATH         = "test_A1_gpu_results.csv"
TEST_A1_POWER_RESULT_CSV_PATH   = "test_A1_gpu_power_results.csv"
TEST_B0_RESULT_CSV_PATH         = "test_B0_gpu_results.csv"
TEST_B0_POWER_RESULT_CSV_PATH   = "test_B0_gpu_power_results.csv"
TEST_C0_RESULT_CSV_PATH         = "test_C0_gpu_results.csv"
TEST_C0_POWER_RESULT_CSV_PATH   = "test_C0_gpu_power_results.csv"

_gpu_log_thread = None
_gpu_log_stop = None


# -----------------------------------------------------------------------------
# MODEL CLASSES
# -----------------------------------------------------------------------------
class Resnet50Saver:
    def __init__(self, output_path_pt, output_path_onnx):

        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        torch.save(self.model, output_path_pt)

        print(f"ResNet50 model (.pt) saved at: {output_path_pt}")

        dummy_img1 = torch.randn(1, 3, 224, 224)
        dummy_img2 = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
        self.model,
        dummy_img1,
        output_path_onnx,
        input_names=["img"],
        output_names=["out"],
        opset_version=12
        )
        print(f"ResNet50 model (.onnx) saved at: {output_path_onnx}")

class Vgg16Saver:
    def __init__(self, output_path_pt, output_path_onnx):

        self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        torch.save(self.model, output_path_pt)

        print(f"VGG16 model (.pt) saved at: {output_path_pt}")

        dummy_img1 = torch.randn(1, 3, 224, 224)

        torch.onnx.export(
        self.model,
        dummy_img1,
        output_path_onnx,
        input_names=["img"],
        output_names=["out"],
        opset_version=12
        )
        print(f"Vgg16 model (.onnx) saved at: {output_path_onnx}")

class ModelLoader:
    def __init__(self, model_path, device):
        self.device = device
        self.model = torch.load(model_path, map_location=device, weights_only=False)
        self.model.eval()

    @torch.no_grad()
    def infer(self, img):
        return self.model(img)


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
def Test_Baseline(model_path, image_paths, labels, val_mapping, results_csv):
    """Baseline single model inference"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "gt_lbl", "gt_name",
            "load_time_1", "preprocess_time_1", "upload_time_1",
            "inference_time_1", "download_time_1", "postprocess_time_1",
            "total_time"
        ])

        # MODEL LOAD
        model = ModelLoader(model_path, device=device)

        for i, img_path in enumerate(image_paths):

            print(i)

            img_name = os.path.basename(img_path)
            gt_lbl = val_mapping.get(img_name, -1)
            gt_name = labels[gt_lbl] if 0 <= gt_lbl < len(labels) else "UNK"

            total_time_start = time.perf_counter()

            # -------------------------------
            # LOAD IMAGE
            # -------------------------------
            t_start = time.perf_counter()
            img = Image.open(img_path).convert("RGB")
            load_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # PREPROCESS + INFERENCE
            # -------------------------------
            # preprocess
            t_start = time.perf_counter()
            tensor = preprocess_pil(img).to(device)
            preprocess_time_1 = time.perf_counter() - t_start

            # upload
            t_start = time.perf_counter()
            tensor = tensor.to(device)
            upload_time_1 = time.perf_counter() - t_start

            # inference
            t_start = time.perf_counter()
            output = model.infer(tensor)
            inference_time_1 = time.perf_counter() - t_start

            # download
            t_start = time.perf_counter()
            output = output.cpu()
            download_time_1 = time.perf_counter() - t_start

            # postprocess
            t_start = time.perf_counter()
            idx, prob = postprocess_topk(output, k=1)
            pred_lbl_1 = int(idx[0])
            prob_1 = float(prob[0])
            pred_name_1 = labels[pred_lbl_1] if pred_lbl_1 < len(labels) else "UNK"
            postprocess_time_1 = time.perf_counter() - t_start

            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV OUTPUT
            # -------------------------------
            writer.writerow([
                img_name,
                pred_lbl_1, pred_name_1, prob_1,
                gt_lbl, gt_name,
                load_time_1, preprocess_time_1, upload_time_1,
                inference_time_1, download_time_1, postprocess_time_1,
                total_time
            ])

def Test_Sequential(model_1_path, model_2_path, image_paths, labels, val_mapping, results_csv):
    """Two sequential model inference"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(results_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "img_name",
            "pred_lbl_1", "pred_name_1", "prob_1",
            "pred_lbl_2", "pred_name_2", "prob_2",
            "gt_lbl", "gt_name",
            "load_time_1", "model_load_time_1", "preprocess_time_1", "upload_time_1",
            "inference_time_1", "download_time_1", "postprocess_time_1",
            "load_time_2", "model_load_time_2", "preprocess_time_2", "upload_time_2",
            "inference_time_2", "download_time_2", "postprocess_time_2",
            "total_time"
        ])

        for i, img_path in enumerate(image_paths):

            print(i)

            img_name = os.path.basename(img_path)
            gt_lbl = val_mapping.get(img_name, -1)
            gt_name = labels[gt_lbl] if 0 <= gt_lbl < len(labels) else "UNK"

            total_time_start = time.perf_counter()

            # -------------------------------
            # LOAD IMAGE 1
            # -------------------------------
            t_start = time.perf_counter()
            img_1 = Image.open(img_path).convert("RGB")
            load_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # MODEL 1
            # -------------------------------
            t_start = time.perf_counter()
            model_1 = ModelLoader(model_1_path, device=device)
            model_load_time_1 = time.perf_counter() - t_start

            # preprocess
            t_start = time.perf_counter()
            tensor_1 = preprocess_pil(img_1).to(device)
            preprocess_time_1 = time.perf_counter() - t_start

            # upload
            t_start = time.perf_counter()
            tensor_1 = tensor_1.to(device)
            upload_time_1 = time.perf_counter() - t_start

            # inference
            t_start = time.perf_counter()
            output1 = model_1.infer(tensor_1)
            inference_time_1 = time.perf_counter() - t_start

            # download
            t_start = time.perf_counter()
            output1 = output1.cpu()
            download_time_1 = time.perf_counter() - t_start

            # postprocess
            t_start = time.perf_counter()
            idx1, prob1 = postprocess_topk(output1, k=1)
            pred_lbl_1 = int(idx1[0])
            prob_1 = float(prob1[0])
            pred_name_1 = labels[pred_lbl_1] if pred_lbl_1 < len(labels) else "UNK"
            postprocess_time_1 = time.perf_counter() - t_start

            # -------------------------------
            # LOAD IMAGE 2
            # -------------------------------
            t_start = time.perf_counter()
            img_2 = Image.open(img_path).convert("RGB")
            load_time_2 = time.perf_counter() - t_start

            # -------------------------------
            # MODEL 2
            # -------------------------------
            t_start = time.perf_counter()
            model_2 = ModelLoader(model_2_path, device=device)
            model_load_time_2 = time.perf_counter() - t_start

            # preprocess
            t_start = time.perf_counter()
            tensor_2 = preprocess_pil(img_2).to(device)
            preprocess_time_2 = time.perf_counter() - t_start

            # upload
            t_start = time.perf_counter()
            tensor_2 = tensor_2.to(device)
            upload_time_2 = time.perf_counter() - t_start

            # inference
            t_start = time.perf_counter()
            output2 = model_2.infer(tensor_2)
            inference_time_2 = time.perf_counter() - t_start

            # download
            t_start = time.perf_counter()
            output2 = output2.cpu()
            download_time_2 = time.perf_counter() - t_start

            # postprocess
            t_start = time.perf_counter()
            idx2, prob2 = postprocess_topk(output2, k=1)
            pred_lbl_2 = int(idx2[0])
            prob_2 = float(prob2[0])
            pred_name_2 = labels[pred_lbl_2] if pred_lbl_2 < len(labels) else "UNK"
            postprocess_time_2 = time.perf_counter() - t_start

            total_time = time.perf_counter() - total_time_start

            # -------------------------------
            # CSV OUTPUT
            # -------------------------------
            writer.writerow([
                img_name,
                pred_lbl_1, pred_name_1, prob_1,
                pred_lbl_2, pred_name_2, prob_2,
                gt_lbl, gt_name,
                load_time_1, model_load_time_1, preprocess_time_1, upload_time_1,
                inference_time_1, download_time_1, postprocess_time_1,
                load_time_2, model_load_time_2, preprocess_time_2, upload_time_2,
                inference_time_2, download_time_2, postprocess_time_2,
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
    # create and save model
    #Resnet50Saver(RESNET50_0_MODEL_PATH, RESNET50_0_MODEL_ONNX_PATH)
    #Resnet50Saver(RESNET50_1_MODEL_PATH, RESNET50_1_MODEL_ONNX_PATH)
    #Vgg16Saver(VGG16_MODEL_PATH, VGG16_MODEL_ONNX_PATH)


    # RUN
    # Start power logger
    #start_gpu_power_log(TEST_A0_POWER_RESULT_CSV_PATH)
    #start_gpu_power_log(TEST_A1_POWER_RESULT_CSV_PATH)
    #start_gpu_power_log(TEST_B0_POWER_RESULT_CSV_PATH)
    start_gpu_power_log(TEST_C0_POWER_RESULT_CSV_PATH)

    # load dataset, labels and validation mapping
    image_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.JPEG")))
    labels = load_labels(LABELS_PATH)
    val_mapping = load_val_labels(VAL_LABELS_PATH)
    print(f"Found {len(image_files)} images")

    #Test_Baseline(RESNET50_0_MODEL_PATH, image_files, labels, val_mapping, TEST_A0_RESULT_CSV_PATH)
    #Test_Baseline(VGG16_MODEL_PATH, image_files, labels, val_mapping, TEST_A1_RESULT_CSV_PATH)
    #Test_Sequential(RESNET50_0_MODEL_PATH, RESNET50_1_MODEL_PATH, image_files, labels, val_mapping, TEST_B0_RESULT_CSV_PATH)
    Test_Sequential(RESNET50_0_MODEL_PATH, VGG16_MODEL_PATH, image_files, labels, val_mapping, TEST_C0_RESULT_CSV_PATH)

    # Stop power log
    stop_gpu_power_log()


if __name__ == '__main__':
    main()
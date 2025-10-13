
import os
import cv2
from PIL import Image
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights, vgg16, VGG16_Weights

#from torchinfo import summary
from pytorch_nndct import Inspector
from pytorch_nndct.apis import torch_quantizer, dump_xmodel

import shutil

# ─── Configuration Class ──────────────────────────────────────────────────────
class Config:
    """Configuration class to manage all paths and settings"""
    
    def __init__(self):
        # Base paths
        self.base_dir = "claudino/projects"
        self.project_name = "Test_3"
        
        # Dataset paths
        self.dataset_dir = f"{self.base_dir}/imagenet500/images/"
        self.labels_path = f"{self.base_dir}/imagenet500/words.txt"
        self.val_labels_path = f"{self.base_dir}/imagenet500/val.txt"
        
        # Input images
        self.image_dir = f"{self.base_dir}/_images/"
        self.img_1_path = f"{self.image_dir}/plane.jpg"
        self.img_2_path = f"{self.image_dir}/plane.jpg"
        
        # Model paths
        self.project_path = f"{self.base_dir}/{self.project_name}"
        self.fused_model_path = f"{self.project_path}/test_3_fused_model.pt"
        self.fused_model_onnx_path = f"{self.project_path}/test_3_fused_model.onnx"
        
        # Output directories
        self.inspect_output_dir = f"{self.project_path}/output_inspect"
        self.quantize_output_dir = f"{self.project_path}/output_quantize"
        self.compilation_output_dir = f"{self.project_path}/output_compilation"
        
        # deploy source path
        self.compiled_fused_model_path = f"{self.compilation_output_dir}/test_3_compiled.xmodel"
        self.run_script_path = f"{self.project_path}/test_3_kv260_run.py"

        # Deploy directories
        self.deploy_dir = f"{self.project_path}/{self.project_name}"

        # Deploy destination path
        self.deploy_model_dest = f"{self.deploy_dir}/test_3_compiled.xmodel"
        self.deploy_run_script_dest = f"{self.deploy_dir}/test_3_kv260_run.py"
        
        # DPU configuration
        self.dpu_target = "DPUCZDX8G_ISA1_B4096"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(self.inspect_output_dir, exist_ok=True)
        os.makedirs(self.quantize_output_dir, exist_ok=True)
        os.makedirs(self.compilation_output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.deploy_dir, exist_ok=True)
        

# ─── Model Classes ───────────────────────────────────────────────────────────
class ResnetFusionSaver:
    def __init__(self, output_path, device='cpu'):
        self.device = device
        self.model1 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model2 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        fusion_dict = {
            'model1': self.model1,
            'model2': self.model2
        }
        torch.save(fusion_dict, output_path)
        print(f"Resnet Fusion model saved at: {output_path}")

class ResnetFusionLoader(nn.Module):
    """Loads two models (resnet 50 and vgg16) from a saved fusion dictionary."""

    def __init__(self, fusion_model_path, device='cpu'):
        super().__init__()
        self.device = device
        fusion_dict = torch.load(fusion_model_path, map_location=device)
        self.model1 = fusion_dict['model1']
        self.model2 = fusion_dict['model2']

        self.model1.eval()
        self.model2.eval()

    def forward(self, img1, img2):
        return self.model1(img1), self.model2(img2)
    
    
# ─── Utility Functions ───────────────────────────────────────────────────────
def load_and_preprocess_images(config):
    """Load and preprocess input images"""
    
    print("Loading and preprocessing images...")
    
    img_1 = Image.open(config.img_1_path).convert('RGB')
    img_2 = Image.open(config.img_2_path).convert('RGB')

    # ResNet preprocessing
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    img1_tensor = preprocess(img_1).unsqueeze(0)
    img2_tensor = preprocess(img_2).unsqueeze(0)
    
    return img1_tensor, img2_tensor, preprocess

def load_labels(config):
    """Load class labels"""
    
    with open(config.labels_path, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels

def test_model_predictions(model, img1_tensor, img2_tensor, labels):
    """Test model predictions"""
    
    print("Testing model predictions...")
    
    with torch.no_grad():
        output1, output2 = model(img1_tensor, img2_tensor)

    pred1 = output1.argmax(1).item()
    pred2 = output2.argmax(1).item()
    
    print(f"Model 1 prediction: {labels[pred1]} (class {pred1})")
    print(f"Model 2 prediction: {labels[pred2]} (class {pred2})")
    
    return output1, output2

def export_to_onnx(model, config, device):
    """Export model to ONNX format"""
    
    print("Exporting model to ONNX...")
    
    dummy_input1 = torch.rand(1, 3, 224, 224).to('cpu')
    dummy_input2 = torch.rand(1, 3, 224, 224).to('cpu')

    # Ensure model is on CPU for ONNX export
    model_cpu = model.to('cpu')
    d1_input, d2_input = dummy_input1.cpu(), dummy_input2.cpu()
    
    torch.onnx.export(
        model_cpu,
        (d1_input, d2_input),
        config.fused_model_onnx_path,
        input_names=["img1", "img2"],
        output_names=["out1", "out2"]
    )
    print(f"ONNX model exported to: {config.fused_model_onnx_path}")
    
    return model_cpu

def run_vitis_ai_inspection(model, config, device):
    """Run Vitis-AI inspection"""
    
    print("Running Vitis-AI inspection...")
    
    dummy_input1 = torch.rand(1, 3, 224, 224).to(device)
    dummy_input2 = torch.rand(1, 3, 224, 224).to(device)

    inspector = Inspector(config.dpu_target)
    inspector.inspect(
        model,
        (dummy_input1, dummy_input2),
        device=device,
        output_dir=config.inspect_output_dir,
        image_format="png"
    )
    print(f"Inspection diagram saved to: {config.inspect_output_dir}/inspect_{config.dpu_target}.png")

def quantize_model(model, config, img1_tensor, img2_tensor, device):
    """Quantize the model"""
    
    print("Quantizing model...")
    
    # Calibration phase
    quantizer = torch_quantizer(
        'calib',
        model,
        (img1_tensor, img2_tensor),
        output_dir=config.quantize_output_dir,
        quant_config_file=None,
        target=config.dpu_target,
        device=device
    )

    quant_model = quantizer.quant_model.to(device)
    quant_model.eval()

    print("Running quantization calibration...")
    for _ in range(10):
        _ = quant_model(img1_tensor, img2_tensor)
    
    quantizer.export_quant_config()
    print(f"Calibration config saved to: {config.quantize_output_dir}/quant_info.json")
    
    # Test phase
    q_test = torch_quantizer(
        'test',
        model,
        (img1_tensor, img2_tensor),
        output_dir=config.quantize_output_dir,
        quant_config_file=None,
        target=config.dpu_target,
        device=device
    )

    quant_test_model = q_test.quant_model.to(device)
    quant_test_model.eval()

    print("Running test forward pass...")
    _ = quant_test_model(img1_tensor, img2_tensor)

    # Save INT8 model
    int8_model_path = f"{config.quantize_output_dir}/quantized_fusion.pt"
    torch.save(quant_test_model.state_dict(), int8_model_path)
    print(f"Saved INT8 PyTorch model to: {int8_model_path}")
    
    return q_test, quant_test_model

def export_quantized_models(quantizer, config):
    """Export quantized models in various formats"""
    
    print("Exporting quantized models...")
    
    quantizer.export_torch_script(config.quantize_output_dir)
    quantizer.export_onnx_model(config.quantize_output_dir)
    quantizer.export_xmodel(config.quantize_output_dir, deploy_check=True)
    print("Exported TorchScript, ONNX, and XModel successfully.")

def compile_for_kv260(config):
    """Compile the model for KV260 DPU"""
    
    print("Compiling model for KV260...")
    
    # A quantização gera o ficheiro ResnetFusionLoader_int.xmodel com base no nome da classe ...
    xmodel_path = f"{config.quantize_output_dir}/ResnetFusionLoader_int.xmodel"
    arch_json_path = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
    
    compile_cmd = f"vai_c_xir -x {xmodel_path} -a {arch_json_path} -o {config.compilation_output_dir} -n test_3_compiled"
    print("Executing command: ", f"{compile_cmd}")
    os.system(compile_cmd)
    
    # Verify DPU subgraphs
    verify_cmd = f"xir png {config.compilation_output_dir}/test_3_compiled.xmodel {config.compilation_output_dir}/test_3_subgraphs.png"
    os.system(verify_cmd)
    
    print(f"Compilation completed. Results in: {config.compilation_output_dir}")    
        
def prepare_for_deployment(config):
    """Step 11: Copy files to deployment folder"""
    
    # Check if source files exist
    if not os.path.exists(config.compiled_fused_model_path):
        print(f"ERROR: Compiled model not found at: {config.compiled_fused_model_path}")
        return
    
    #if not os.path.exists(config.run_script_path):
    #    print(f"ERROR: Python script not found at: {config.run_script_path}")
    #    return
    
    # Copy files
    try:
        # Copy compiled model
        shutil.copy2(config.compiled_fused_model_path, config.deploy_model_dest)
        print(f"Copied compiled model to: {config.deploy_model_dest}")
        
        ## Copy Python script
        #shutil.copy2(config.run_script_path, config.deploy_run_script_dest)
        #print(f"Copied Python script to: {config.deploy_run_script_dest}")
        
        # List files in deployment directory
        print(f"\nFiles in deployment directory ({config.deploy_dir}):")
        for file in os.listdir(config.deploy_dir):
            print(f"  - {file}")
        
        print(f"\nDeployment preparation completed successfully!")
        print(f"Deployment folder: {config.deploy_dir}")

    except Exception as e:
        print(f"ERROR: Failed to copy files: {e}")
    
        
# ─── Main Function ───────────────────────────────────────────────────────────
def main():
    """Main execution function"""
    
    # Initialize configuration
    config = Config()
    
    # Set device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device =torch.device("cpu")
    print(f"Using device: {device}")
    
    try:
        # Step 1: Create and save fused model
        print("\n" + "="*50)
        print("STEP 1: Creating fused model")
        print("="*50)
        ResnetFusionSaver(config.fused_model_path, device)
        
        # Step 2: Load fused model
        print("\n" + "="*50)
        print("STEP 2: Loading fused model")
        print("="*50)
        resnet_fused_model = ResnetFusionLoader(config.fused_model_path, device)
        
        # Step 3: Load and preprocess images
        print("\n" + "="*50)
        print("STEP 3: Load and preprocess images")
        print("="*50)
        img1_tensor, img2_tensor, preprocess = load_and_preprocess_images(config)
        
        # Step 4: Load labels
        print("\n" + "="*50)
        print("STEP 4: Load labels")
        print("="*50)
        labels = load_labels(config)
        
        # Step 5: Test model predictions
        print("\n" + "="*50)
        print("STEP 5: Test model predictions")
        print("="*50)
        test_model_predictions(resnet_fused_model, img1_tensor, img2_tensor, labels)
        
        # Step 6: Export to ONNX
        print("\n" + "="*50)
        print("STEP 6: Exporting to ONNX")
        print("="*50)
        model_cpu = export_to_onnx(resnet_fused_model, config, device)
        
        # Step 7: Vitis-AI inspection
        print("\n" + "="*50)
        print("STEP 7: Vitis-AI inspection")
        print("="*50)
        run_vitis_ai_inspection(model_cpu, config, device)
        
        # Step 8: Quantization
        print("\n" + "="*50)
        print("STEP 8: Quantization")
        print("="*50)
        quantizer, quant_model = quantize_model(model_cpu, config, img1_tensor, img2_tensor, device)
        
        # Step 9: Export quantized models
        print("\n" + "="*50)
        print("STEP 9: Exporting quantized models")
        print("="*50)
        export_quantized_models(quantizer, config)
            
        # Step 10: Compile for KV260
        print("\n" + "="*50)
        print("STEP 10: Compiling for KV260")
        print("="*50)
        compile_for_kv260(config)
        
        # Step 11: Prepare for deployment
        print("\n" + "="*50)
        print("STEP 11: Preparing for Deployment")
        print("="*50)
        prepare_for_deployment(config)
        
        print("\n" + "="*50)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
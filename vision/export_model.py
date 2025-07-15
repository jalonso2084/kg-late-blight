import torch
import timm
import onnx
import onnxruntime
import numpy as np

# --- CONFIGURATION ---
# This should match the configuration from your training script
CONFIG = {
    "model_name": "vit_tiny_patch16_224",
    "num_classes": 2,
    "image_size": 224,
    "pytorch_model_path": "vision/mosvit_blite.pt",
    "onnx_model_path": "vision/mosvit_blite.onnx"
}

def export_to_onnx():
    """Loads a trained PyTorch model and exports it to the ONNX format."""

    print(f"Loading PyTorch model from {CONFIG['pytorch_model_path']}...")
    # First, create the model architecture
    model = timm.create_model(
        CONFIG["model_name"], 
        pretrained=False, # We are loading our own weights, not pre-trained ones
        num_classes=CONFIG["num_classes"]
    )

    # Load the saved weights (the state dictionary)
    model.load_state_dict(torch.load(CONFIG["pytorch_model_path"]))

    # Set the model to evaluation mode
    model.eval()

    # Create a dummy input tensor with the correct shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"])

    print(f"Exporting model to ONNX format at {CONFIG['onnx_model_path']}...")

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        CONFIG["onnx_model_path"],
        export_params=True,
        opset_version=14, # Use version 14 as required by the model's operator
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
    )

    print("Model exported successfully.")

    # --- VERIFICATION STEP ---
    print("Verifying the ONNX model...")
    onnx_model = onnx.load(CONFIG["onnx_model_path"])
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession(CONFIG["onnx_model_path"])
    input_name = ort_session.get_inputs()[0].name

    # Create a dummy numpy array for inference
    dummy_np_input = dummy_input.numpy()

    # Run inference
    ort_outs = ort_session.run(None, {input_name: dummy_np_input})

    print("ONNX model verified successfully. It can be loaded and run.")
if __name__ == "__main__":
    export_to_onnx()
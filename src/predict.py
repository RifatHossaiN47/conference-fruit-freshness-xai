"""
Command-Line Interface (CLI) for Multi-Task Fruit & Freshness Prediction
Usage:
    python src/predict.py --image path/to/sample.jpg --gradcam
"""

import argparse
import os
import sys

# Ensure src package can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import FRUIT_CLASSES, FRESHNESS_CLASSES

def preprocess_image(image_path, target_size=(224, 224)):
    """Loads and preprocesses image for ResNet152V2 model inference."""
    from PIL import Image
    import numpy as np

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")
        
    img = Image.open(image_path).convert("RGB")
    original_np = np.array(img)
    img_resized = img.resize(target_size)
    
    # Rescale to [0, 1]
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    batch_array = np.expand_dims(img_array, axis=0)
    
    return original_np, img_resized, batch_array

def run_prediction(image_path, weights_path="models/ResNet152.keras", save_gradcam=False, output_dir="results"):
    """Performs multi-task inference and optional Grad-CAM generation."""
    import numpy as np
    import matplotlib.pyplot as plt
    from src.model import load_trained_model
    from src.gradcam import MultiTaskGradCAM

    print("=" * 60)
    print("🍎 Multi-Task Fruit & Freshness Classifier (IEEE ICCIT 2025)")
    print("=" * 60)
    print(f"Loading weights from: {weights_path}")
    
    model = load_trained_model(weights_path)
    original_np, img_resized, batch_array = preprocess_image(image_path)
    
    # Run multi-task forward pass
    preds = model.predict(batch_array, verbose=0)
    fruit_probs = preds[0][0]
    fresh_prob = float(preds[1][0][0])
    
    fruit_idx = int(np.argmax(fruit_probs))
    fruit_name = FRUIT_CLASSES[fruit_idx] if fruit_idx < len(FRUIT_CLASSES) else f"Class_{fruit_idx}"
    fruit_conf = float(fruit_probs[fruit_idx]) * 100.0
    
    # Binary freshness threshold (0.5)
    is_rotten = fresh_prob >= 0.5
    freshness_label = "Rotten" if is_rotten else "Fresh"
    freshness_conf = (fresh_prob if is_rotten else (1.0 - fresh_prob)) * 100.0
    
    print("\n--- Prediction Results ---")
    print(f"🍓 Fruit Variety : {fruit_name} (Confidence: {fruit_conf:.2f}%)")
    print(f"✨ Freshness     : {freshness_label} (Confidence: {freshness_conf:.2f}%)")
    print(f"⚠️ Rotten Score  : {fresh_prob * 100.0:.2f}%")
    print("=" * 60)
    
    if save_gradcam:
        os.makedirs(output_dir, exist_ok=True)
        print("\nGenerating Explainable AI (Grad-CAM) Visualizations...")
        explainer = MultiTaskGradCAM(model)
        
        # Heatmap for fruit identification
        fruit_heatmap = explainer.generate_heatmap(batch_array, task_output="fruit_output", class_index=fruit_idx)
        fruit_overlay = explainer.overlay_heatmap(fruit_heatmap, np.array(img_resized))
        
        # Heatmap for freshness assessment
        fresh_heatmap = explainer.generate_heatmap(batch_array, task_output="fresh_output", class_index=0)
        fresh_overlay = explainer.overlay_heatmap(fresh_heatmap, np.array(img_resized))
        
        # Plot comparative Grad-CAM
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title("Input Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        axes[1].imshow(fruit_overlay)
        axes[1].set_title(f"Fruit Attention: {fruit_name}\n({fruit_conf:.1f}% Conf)", fontsize=12, fontweight="bold", color="navy")
        axes[1].axis("off")
        
        state_color = "darkred" if is_rotten else "darkgreen"
        axes[2].imshow(fresh_overlay)
        axes[2].set_title(f"Freshness Attention: {freshness_label}\n({freshness_conf:.1f}% Conf)", fontsize=12, fontweight="bold", color=state_color)
        axes[2].axis("off")
        
        plt.suptitle("ResNet152V2 Multi-Task Dual-Head Explainability", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout()
        
        out_filename = os.path.join(output_dir, f"gradcam_{os.path.splitext(os.path.basename(image_path))[0]}.png")
        plt.savefig(out_filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Grad-CAM visual saved to: {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Fruit Type and Freshness with ResNet152V2 Multi-Task Model")
    parser.add_argument("--image", type=str, required=True, help="Path to input fruit image")
    parser.add_argument("--weights", type=str, default="models/ResNet152.keras", help="Path to model weights file")
    parser.add_argument("--gradcam", action="store_true", help="Generate and save Grad-CAM explainability heatmaps")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output visuals")
    
    args = parser.parse_args()
    try:
        run_prediction(args.image, args.weights, args.gradcam, args.output_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

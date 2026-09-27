"""
Explainable AI (XAI) Grad-CAM Module
Generates task-specific gradient-weighted class activation heatmaps
for both fruit identification and freshness assessment branches.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

class MultiTaskGradCAM:
    """
    Grad-CAM interpreter tailored for dual-head multi-task CNN architectures.
    """
    def __init__(self, model, target_layer_name="conv5_block3_3_conv"):
        self.model = model
        self.target_layer_name = target_layer_name
        self._find_target_layer()

    def _find_target_layer(self):
        """Locates the layer inside the model or nested backbone."""
        try:
            self.target_layer = self.model.get_layer(self.target_layer_name)
        except ValueError:
            # Check if layer is inside a nested base_model
            for layer in self.model.layers:
                if hasattr(layer, "layers"):
                    try:
                        self.target_layer = layer.get_layer(self.target_layer_name)
                        return
                    except ValueError:
                        continue
            # Fallback to last 4D conv layer
            for layer in reversed(self.model.layers):
                if len(layer.output_shape) == 4 and "conv" in layer.name:
                    self.target_layer = layer
                    self.target_layer_name = layer.name
                    return

    def generate_heatmap(self, img_array, task_output="fruit_output", class_index=0):
        """
        Computes Grad-CAM heatmap for a given task output head.

        Args:
            img_array (np.ndarray): Preprocessed batch image (1, H, W, 3).
            task_output (str): 'fruit_output' or 'fresh_output'.
            class_index (int): Target class index for categorical softmax output.

        Returns:
            np.ndarray: Normalized 2D activation heatmap in range [0, 1].
        """
        grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.target_layer.output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if task_output == "fruit_output":
                # Multi-class output (Softmax)
                task_preds = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
                loss = task_preds[0, class_index]
            else:
                # Binary freshness output (Sigmoid)
                task_preds = predictions[1] if isinstance(predictions, (list, tuple)) else predictions
                loss = task_preds[0, 0]

        # Calculate gradients of score with respect to feature maps
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU to focus only on features that positively contribute to the class
        heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap, original_image, alpha=0.55, colormap=cv2.COLORMAP_JET):
        """
        Overlays activation heatmap onto the original RGB image.

        Args:
            heatmap (np.ndarray): 2D activation heatmap [0, 1].
            original_image (np.ndarray): Original image in RGB (H, W, 3), range [0, 255] or [0, 1].
            alpha (float): Transparency factor for overlay.
            colormap (int): OpenCV colormap.

        Returns:
            np.ndarray: Blended RGB image.
        """
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = original_image.astype(np.uint8)

        h, w = original_image.shape[:2]
        resized_heatmap = cv2.resize(heatmap, (w, h))
        uint8_heatmap = np.uint8(255 * resized_heatmap)

        colored_heatmap = cv2.applyColorMap(uint8_heatmap, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(colored_heatmap, alpha, original_image, 1 - alpha, 0)
        return blended

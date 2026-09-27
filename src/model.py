"""
ResNet152V2 Multi-Task Model Architecture
Dual-head deep learning architecture for joint fruit type and freshness classification.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet152V2

def build_multitask_resnet152(input_shape=(224, 224, 3), num_fruit_classes=9, fine_tune_layers=30):
    """
    Constructs the proposed ResNet152V2 Multi-Task architecture.
    
    Args:
        input_shape (tuple): Dimensions of input image (H, W, C).
        num_fruit_classes (int): Number of fruit variety classes (default: 9).
        fine_tune_layers (int): Number of top layers in ResNet152V2 backbone to unfreeze.
        
    Returns:
        keras.Model: Configured multi-task neural network model.
    """
    inputs = layers.Input(shape=input_shape, name="input_image")
    
    # Pre-trained ResNet152V2 feature extraction backbone
    base_model = ResNet152V2(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )
    
    # Freeze lower layers and fine-tune top layers
    if fine_tune_layers > 0:
        for layer in base_model.layers[:-fine_tune_layers]:
            layer.trainable = False
        for layer in base_model.layers[-fine_tune_layers:]:
            layer.trainable = True
    else:
        base_model.trainable = False
        
    # Shared feature extraction trunk
    shared_features = base_model.output
    x = layers.GlobalAveragePooling2D(name="shared_gap")(shared_features)
    x = layers.Dense(512, activation="relu", name="shared_dense_512")(x)
    x = layers.BatchNormalization(name="shared_bn")(x)
    x = layers.Dropout(0.5, name="shared_dropout")(x)
    shared_trunk = layers.Dense(256, activation="relu", name="shared_256_res")(x)
    
    # Branch 1: Fruit Type Classification (9 classes, Softmax)
    fruit_branch = layers.Dense(128, activation="relu", name="fruit_dense_128")(shared_trunk)
    fruit_branch = layers.Dropout(0.3, name="fruit_dropout")(fruit_branch)
    fruit_output = layers.Dense(num_fruit_classes, activation="softmax", name="fruit_output")(fruit_branch)
    
    # Branch 2: Freshness Detection (Binary: Fresh vs. Rotten, Sigmoid)
    fresh_branch = layers.Dense(64, activation="relu", name="fresh_dense_64")(shared_trunk)
    fresh_branch = layers.Dropout(0.3, name="fresh_dropout")(fresh_branch)
    fresh_output = layers.Dense(1, activation="sigmoid", name="fresh_output")(fresh_branch)
    
    model = models.Model(
        inputs=inputs,
        outputs=[fruit_output, fresh_output],
        name="ResNet152V2_MultiTask_Classifier"
    )
    
    return model

def compile_multitask_model(model, learning_rate=0.0005, alpha=0.7, beta=0.3):
    """
    Compiles the multi-task model with dual loss functions and specified weights.
    
    Args:
        model (keras.Model): Model to compile.
        learning_rate (float): Optimizer learning rate.
        alpha (float): Loss weight for fruit classification (default: 0.7).
        beta (float): Loss weight for freshness classification (default: 0.3).
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    losses = {
        "fruit_output": "categorical_crossentropy",
        "fresh_output": "binary_crossentropy"
    }
    
    loss_weights = {
        "fruit_output": alpha,
        "fresh_output": beta
    }
    
    metrics = {
        "fruit_output": "accuracy",
        "fresh_output": "accuracy"
    }
    
    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    return model

def load_trained_model(weights_path="models/ResNet152.keras"):
    """
    Loads saved model weights with safe fallback.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights not found at '{weights_path}'. "
            "Please check models/README.md for instructions on downloading model weights."
        )
    return keras.models.load_model(weights_path)

"""
Diabetic Retinopathy Detection - Hybrid Deep Learning Architecture
Combines:
1. Custom Deep Convolutional Neural Network (CNN) Branch for local micro-lesion detection
   (microaneurysms, hemorrhages, hard/soft exudates).
2. Deep Transfer Learning Branch (EfficientNetB0 / ResNet) for high-level global semantic representations.
3. Feature Fusion Layer (Multi-modal feature concatenation + dense projection).
4. Deep Classification Head with Softmax activation for 5 DR severity classes:
   - Class 0: No DR (Normal)
   - Class 1: Mild Non-Proliferative DR
   - Class 2: Moderate Non-Proliferative DR
   - Class 3: Severe Non-Proliferative DR
   - Class 4: Proliferative Diabetic Retinopathy (PDR)
"""

import os
import io

def build_hybrid_model(input_shape=(224, 224, 3), num_classes=5, pretrained_backbone='efficientnet'):
    """
    Constructs the dual-branch Hybrid Deep Learning Model.
    
    Args:
        input_shape: tuple of (height, width, channels), default (224, 224, 3)
        num_classes: integer, default 5
        pretrained_backbone: 'efficientnet' or 'resnet50'
    
    Returns:
        keras.Model: Compiled or uncompiled hybrid Keras model
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, Model

    inputs = layers.Input(shape=input_shape, name="fundus_input")

    # ==========================================
    # BRANCH 1: Custom Deep CNN Feature Extractor
    # Tailored for local retinal lesion textures
    # ==========================================
    c1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="custom_conv1")(inputs)
    c1 = layers.BatchNormalization(name="custom_bn1")(c1)
    c1 = layers.MaxPooling2D(pool_size=(2, 2), name="custom_pool1")(c1)

    c2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="custom_conv2")(c1)
    c2 = layers.BatchNormalization(name="custom_bn2")(c2)
    c2 = layers.MaxPooling2D(pool_size=(2, 2), name="custom_pool2")(c2)

    c3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="custom_conv3")(c2)
    c3 = layers.BatchNormalization(name="custom_bn3")(c3)
    c3 = layers.MaxPooling2D(pool_size=(2, 2), name="custom_pool3")(c3)

    c4 = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name="custom_conv4")(c3)
    c4 = layers.BatchNormalization(name="custom_bn4")(c4)
    c4 = layers.GlobalAveragePooling2D(name="custom_gap")(c4)
    c_features = layers.Dense(256, activation='relu', name="custom_features")(c4)
    c_features = layers.Dropout(0.4, name="custom_dropout")(c_features)

    # =======================================================
    # BRANCH 2: Pre-trained Deep Feature Extractor
    # (EfficientNetB0 or ResNet50 backbone)
    # =======================================================
    try:
        if pretrained_backbone == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs
            )
        else:
            backbone = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_tensor=inputs
            )
        # Freeze backbone initial layers for feature stability
        backbone.trainable = False
        p_gap = layers.GlobalAveragePooling2D(name="backbone_gap")(backbone.output)
    except Exception as e:
        print(f"[HybridModel] Notice: Pretrained weights download skipped or offline ({e}). Using initialized backbone.")
        if pretrained_backbone == 'resnet50':
            backbone = tf.keras.applications.ResNet50(
                include_top=False,
                weights=None,
                input_tensor=inputs
            )
        else:
            backbone = tf.keras.applications.EfficientNetB0(
                include_top=False,
                weights=None,
                input_tensor=inputs
            )
        p_gap = layers.GlobalAveragePooling2D(name="backbone_gap")(backbone.output)

    p_features = layers.Dense(256, activation='relu', name="backbone_features")(p_gap)
    p_features = layers.Dropout(0.4, name="backbone_dropout")(p_features)

    # =======================================================
    # FEATURE FUSION LAYER
    # Concatenates local textural representations and global semantic features
    # =======================================================
    fused = layers.Concatenate(name="feature_fusion")([c_features, p_features])

    # =======================================================
    # CLASSIFICATION HEAD
    # Multi-stage dense projection with regularized dropout
    # =======================================================
    fc1 = layers.Dense(256, activation='relu', name="dense_fc1")(fused)
    fc1 = layers.BatchNormalization(name="fc1_bn")(fc1)
    fc1 = layers.Dropout(0.4, name="fc1_dropout")(fc1)

    fc2 = layers.Dense(128, activation='relu', name="dense_fc2")(fc1)
    fc2 = layers.Dropout(0.3, name="fc2_dropout")(fc2)

    output = layers.Dense(num_classes, activation='softmax', name="severity_output")(fc2)

    model = Model(inputs=inputs, outputs=output, name="Hybrid_DR_Classifier")

    return model


def get_model_summary_text(model):
    """
    Returns string summary of the model architecture.
    """
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    return stream.getvalue()


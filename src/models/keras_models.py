import tensorflow as tf
from tensorflow.keras import layers, Model

def build_fpn(backbone, layer_names, feature_channels=256):
    # Extract intermediate feature maps
    feature_maps = [backbone.get_layer(name).output for name in layer_names]

    # Apply lateral 1x1 conv to each
    lateral_convs = [layers.Conv2D(feature_channels, 1, padding='same')(fm) 
                     for fm in feature_maps]

    # Build top-down pathway
    # Start with the top-most feature map
    P6 = lateral_convs[-1]
    # Iterate backwards (excluding the last added P6)
    pyramid_features = [P6]
    for conv_feature in reversed(lateral_convs[:-1]):
        upsampled = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(pyramid_features[-1])
        merged = layers.Add()([conv_feature, upsampled])
        pyramid_features.append(merged)
    # Reverse to restore ordering from lowest level to highest
    pyramid_features = pyramid_features[::-1]

    # Optionally, apply additional 3x3 conv smoothing on the lowest level feature map
    p_final = layers.Conv2D(feature_channels, 3, padding='same', activation='relu')(pyramid_features[0])
    return p_final

def build_model(input_shape=(256, 256, 3), num_keypoints=24):
    inputs = layers.Input(shape=input_shape)

    # Create MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights=None
    )

    # Specify layer names to extract feature maps from MobileNetV2.
    # For example, these layers correspond to different spatial resolutions.
    layer_names = [
        'block_3_expand_relu',   # higher resolution feature map
        'block_6_expand_relu',   # middle resolution feature map
        'block_13_expand_relu',  # lower resolution feature map
        'out_relu'            # final feature map from backbone
    ]

    # Build a simple FPN using the selected feature maps
    fpn_feature = build_fpn(backbone, layer_names, feature_channels=256)

    heatmaps = layers.Conv2D(num_keypoints, kernel_size=1, activation='sigmoid')(fpn_feature)
    model = Model(inputs=inputs, outputs=heatmaps)
    return model

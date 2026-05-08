import os
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input
from keras.models import Model


class ModelFactory:
    def build_cnn(
        image_shape,
        conv_config_layers: list ,
        fnn_config_layers: list ,
        output_dir: str,
        output_units: int = 2,
        head_config_layers: list[dict] = None,
        aux_input_shapes: dict[str,tuple] = None
        ) -> keras.Model:

        """ Builds a configurable convolutional neural network (CNN) model for regression tasks,
        with optional support for auxiliary scalar inputs and customizable layer configurations.

        Parameters
        ----------
        image_shape : tuple
            Shape of the input image, e.g., (height, width, channels).
        conv_config_layers : list of dict
            List of dictionaries specifying the configuration of each Conv2D layer.
            Each dict can include keys: "filters", "kernel_size", "stride", "padding",
            "activation", "batchnorm", "pool", "pool_size".
        fnn_config_layers : list of dict
            List of dictionaries specifying the configuration of each fully-connected (Dense) layer
            after the convolutional blocks. Each dict can include: "units", "activation",
            "batchnorm", "dropout", "dropout_rate".
        output_dir : str
            Directory where the model summary text file will be saved.
        output_units : int, optional
            Number of output units (default is 2), e.g., for [mean, variance].
        head_config_layers : list of dict, optional
            List of dictionaries specifying additional Dense "head" layers applied after concatenating
            auxiliary features. Same dict keys as `fnn_config_layers`. If None, a default two-layer
            head is used.
        aux_input_shapes : dict of str to tuple, optional
            Dictionary mapping auxiliary input names to their shapes (e.g., {"redshift": (1,)}).
            If provided, the corresponding inputs are added and concatenated to the flattened CNN features.

        Returns
        -------
        keras.Model
            Compiled Keras model instance ready for training.

        Notes
        -----
        - If `aux_input_shapes` is provided, the model will have multiple inputs (images and auxiliaries).
        - The model outputs two values per sample, with the second output (variance/uncertainty)
        passed through a sigmoid activation to constrain its range.
        - The model summary will be saved as "model_summary.txt" in `output_dir`.
        """
        
        img_in = Input(shape=image_shape, name="image_input")
        x = img_in

        for cfg in conv_config_layers:
            x = layers.Conv2D(
                filters=cfg["filters"],
                kernel_size=cfg["kernel_size"],
                strides=cfg.get("stride", (1, 1)),
                padding=cfg.get("padding", "valid"),
                activation=cfg.get("activation", "relu")
            )(x)
            if cfg.get("batchnorm", False):
                x = layers.BatchNormalization()(x)
            if cfg.get("pool", False):
                x = layers.MaxPooling2D(pool_size=cfg.get("pool_size", (2, 2)))(x)
        
        x = layers.Flatten()(x)

        for cfg in fnn_config_layers:
            x = layers.Dense(
                units=cfg["units"],
                activation=cfg.get("activation", "relu")
            )(x)
            if cfg.get("batchnorm", False):
                x = layers.BatchNormalization()(x)
            if cfg.get("dropout", False):
                x = layers.Dropout(rate=cfg.get("dropout_rate", 0.3))(x)

        # prepare inputs and concatenation list
        inputs    = [img_in]
        to_concat = [x]

        if aux_input_shapes:
            for name, shape in aux_input_shapes.items():
                aux_in = Input(shape=shape, name=f"{name}_input")
                inputs.append(aux_in)
                to_concat.append(aux_in)

        # merge if needed
        if aux_input_shapes:
            merged = layers.Concatenate()(to_concat)
            # use head_config_layers if provided, else fall back to default two‑layer head
            head_cfg = head_config_layers or [
                {"units": 32, "activation": "relu"},
                {"units": 16, "activation": "relu"},
            ]
            h = merged
            for cfg in head_cfg:
                h = layers.Dense(
                    cfg["units"],
                    activation=cfg.get("activation","relu"))(h)
                if cfg.get("batchnorm", False):
                    h = layers.BatchNormalization()(h)
                if cfg.get("dropout", False):
                    h = layers.Dropout(cfg.get("dropout_rate",0.3))(h)

            out = layers.Dense(2, name="raw_outputs")(h)
            out = layers.Lambda(lambda x: tf.stack([x[:, 0], tf.sigmoid(x[:, 1])], axis=1), name="final_output")(out)
            model = Model(inputs=inputs, outputs=out, name="cnn_with_aux")
        else:
            out = layers.Dense(2, name="raw_outputs")(x)
            out = layers.Lambda(lambda x: tf.stack([x[:, 0], tf.sigmoid(x[:, 1])], axis=1), name="final_output")(out)
            model = Model(inputs=img_in, outputs=out, name="cnn")

        # save summary
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "model_summary.txt"), "w") as f:
            model.summary(print_fn=lambda line: f.write(line + "\n"))

        return model





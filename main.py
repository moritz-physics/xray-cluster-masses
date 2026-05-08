import os
import argparse
from data_loader import Loader
from preprocessing import Preprocessor
from models import ModelFactory
from training import Trainer  
from keras.optimizers import Adam


def run_pipeline(extra_features: list[str], output_dir: str, augment_images: bool = False):
    """Run end-to-end: load data, preprocess, build CNN, train with NLL loss, evaluate, save artefacts."""

    # data path — catalogue is feather-format (see README on converting from FITS)
    image_path = "physics_data/eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle"
    catalog_path = "physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f"

    # load data
    loader = Loader(image_path, catalog_path)
    images, features = loader.get_data()  # features is a dict e.g. {"mass":…, "redshift":…, "index":…}



    # preprocess data 
    prep = Preprocessor(images, features)
    prep.normalize_images(mode="per_image")
    prep.normalize_feature("mass") 
    for feat in extra_features:
        prep.normalize_feature(feat) 
    
    train_set, val_set, test_set = prep.split()

    # Augment the images of the train set
    if augment_images:
        train_set.augment_images()

    # build training, validation and test inputs

    x_train, y_train = train_set.get_xy("mass")  # x_train: images (N, H, W, C); y_train: normalised log-masses
    x_val,   y_val   = val_set.get_xy("mass")
    x_test,  y_test  = test_set.get_xy("mass")

    inputs_train = {"image_input": x_train}
    inputs_val   = {"image_input": x_val}
    inputs_test  = {"image_input": x_test}

    aux_shapes = {}  # will feed into build_cnn

    for feat in extra_features:
        add_train = train_set.get_feature(feat, as_column=True)
        add_val = val_set.get_feature(feat, as_column=True)
        add_test = test_set.get_feature(feat, as_column=True)

        # create new keys for the input dictionaries with new features
        inputs_train[f"{feat}_input"] = add_train
        inputs_val[  f"{feat}_input"] = add_val
        inputs_test[ f"{feat}_input"] = add_test

        # ensure input shape for added features
        aux_shapes[feat] = add_train.shape[1:]  # (1,)

    input_shape = x_train.shape[1:]  # (H, W, C) — height, width, channels

    # configuration of model
    # conv_layers = [
    #     {"filters": 16, "kernel_size": (10, 10), "padding": "same", "activation": "relu","pool": True, "pool_size": (2, 2), "batchnorm": True},
    #     {"filters": 32, "kernel_size": (5, 5), "padding": "same", "activation": "relu", "batchnorm": True},
    #     {"filters": 64, "kernel_size": (5, 5), "padding": "same", "activation": "relu", "pool": True, "pool_size": (2, 2), "batchnorm": True},
    #     {"filters": 128, "kernel_size": (2, 2), "padding": "same", "activation": "relu", "pool": True, "pool_size": (2, 2), "batchnorm": True},
    # ]


    # fnn_layers = [
    #     {"units": 64, "activation": "relu", "dropout": True, "dropout_rate": 0.5, "batchnorm": True},
    #     {"units": 32, "activation": "relu", "dropout": True, "dropout_rate": 0.2},
    # ]

    # head_config_layers = [
    #     {"units": 16, "activation": "relu", "dropout": True, "dropout_rate": 0.5, "batchnorm": True},
    #     {"units": 4, "activation": "relu", "dropout": True, "dropout_rate": 0.2},
    # ]



    # ------------------------- OPTUNA STUDY RESULTS ------------------------- #
    conv_layers = [
    {"filters": 32, "kernel_size": (5, 5), "padding": "same", "activation": "relu", "pool": False, "batchnorm": False},
    {"filters": 32, "kernel_size": (2, 2), "padding": "same", "activation": "relu", "pool": False, "batchnorm": False},
    {"filters": 128, "kernel_size": (2, 2), "padding": "same", "activation": "relu", "pool": False, "batchnorm": True},
    ]
    fnn_layers = [
        {"units": 64, "activation": "relu", "dropout": True, "dropout_rate": 0.4509, "batchnorm": False},
        {"units": 256, "activation": "relu", "dropout": True, "dropout_rate": 0.4036, "batchnorm": False},
    ]
    head_config_layers = [
        {"units": 256, "activation": "relu", "dropout": False, "batchnorm": False},
    ]
    learning_rate = 0.0002615

    # Best Optuna config used: redshift + kT + image augmentation
    # ------------------------- OPTUNA STUDY RESULTS ------------------------- #


    #  build the model
    model = ModelFactory.build_cnn(
        image_shape        = input_shape,
        conv_config_layers = conv_layers,
        fnn_config_layers  = fnn_layers,
        output_dir         = output_dir,
        head_config_layers=head_config_layers,
        aux_input_shapes   = aux_shapes or None,
    )

    # training
    trainer = Trainer(model, output_dir)
    trainer.compile(
        # optimizer=Adam(learning_rate=learning_rate), 
        optimizer="adam",
        loss=Trainer.loss_nll, 
        metrics=[Trainer.mse_mean]
        )
    history = trainer.train(
        inputs_train, y_train,
        inputs_val,   y_val,
        batch_size=64, epochs=100, patience=20
    )

    # evaluation + Prediction
    test_loss, test_mse = trainer.evaluate(inputs_test, y_test)
    # Predict (normalized)
    y_pred = trainer.predict(inputs_test)       # shape (N, 2)

    print(f"\nFinal test MSE: {test_mse:.4f}")

    # save + Plot
    model.save(os.path.join(output_dir, "model.keras"))
    Trainer.plot_training_curves(history, output_dir)
    Trainer.plot_predictions(y_pred, y_test, output_dir, prep=prep)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a CNN on x-ray cluster images, optionally with extra scalar features"
    )
    parser.add_argument(
        "--features", 
        nargs="+", 
        # choices=["redshift", "kT"], 
        default=[],
        help="List of extra scalar features to include alongside the image"
    )
    parser.add_argument(
        "--out-dir", 
        default="model_results", 
        help="Where to save model, summary, and plots"
    )
    parser.add_argument(
        "--augment-images",  
        action="store_true", 
        help="Apply image augmentation to training data"
    )
    args = parser.parse_args()
    run_pipeline(extra_features=args.features, augment_images=args.augment_images, output_dir=f"final_results/{args.out_dir}")



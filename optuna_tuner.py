import os
import argparse
import optuna
import logging
import sys
from keras.optimizers import Adam
from data_loader import Loader
from preprocessing import Preprocessor
from models import ModelFactory
from training import Trainer
from optuna.samplers import TPESampler

CHECKPOINT_EVERY = 10


def save_checkpoint(study, output_dir):
    """Save trial results to CSV."""
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df.to_csv(os.path.join(output_dir, "optuna_checkpoint.csv"), index=False)
    print(f"[Checkpoint] Saved CSV after {len(study.trials)} trials.")


def objective(trial, output_dir):
    try:
        # Load data
        image_path = "physics_data/eFEDS_01to18-3dImgs-7946clus-300pix-50pix-ext_det_thr.pickle"
        catalog_path = "physics_data/eFEDS_mock_clusters_catalog_01to18-ext_det_thr.f"

        loader = Loader(image_path, catalog_path)
        images, features = loader.get_data()

        # Preprocessing
        possible_feature_sets = [
            [],
            ["redshift"],
            ["kT"],
            ["Lx_soft"],
            ["redshift", "kT"],
            ["redshift", "Lx_soft"],
            ["kT", "Lx_soft"],
            ["redshift", "kT", "Lx_soft"]
        ]

        extra_features = trial.suggest_categorical("extra_features", possible_feature_sets)

        prep = Preprocessor(images, features)
        prep.normalize_images(mode="per_image")
        prep.normalize_feature("mass")
        for feat in extra_features:
            prep.normalize_feature(feat)

        train_set, val_set, _ = prep.split()
        x_train, y_train = train_set.get_xy("mass")
        x_val, y_val = val_set.get_xy("mass")

        inputs_train = {"image_input": x_train}
        inputs_val = {"image_input": x_val}
        aux_shapes = {}

        for feat in extra_features:
            f_train = train_set.get_feature(feat, as_column=True)
            f_val = val_set.get_feature(feat, as_column=True)
            inputs_train[f"{feat}_input"] = f_train
            inputs_val[f"{feat}_input"] = f_val
            aux_shapes[feat] = f_train.shape[1:]

        input_shape = x_train.shape[1:]

        # Conv layers
        conv_config = []
        for i in range(trial.suggest_int("n_conv_layers", 2, 4)):
            kernel_options = [(10, 10), (5, 5), (3, 3)] if i == 0 else [(5, 5), (3, 3), (2, 2)]
            conv_config.append({
                "filters": trial.suggest_categorical(f"filters_{i}", [16, 32, 64, 128]),
                "kernel_size": trial.suggest_categorical(f"kernel_{i}", kernel_options),
                "pool": trial.suggest_categorical(f"pool_{i}", [True, False]),
                "batchnorm": trial.suggest_categorical(f"bn_{i}", [True, False])
            })


        # FNN layers
        fnn_config = []
        n_fnn = trial.suggest_int("n_fnn_layers", 1, 2)
        for i in range(n_fnn):
            layer = {
                "units": trial.suggest_categorical(f"fnn_units_{i}", [64, 128, 256]),
                "batchnorm": trial.suggest_categorical(f"fnn_bn_{i}", [True, False]),
                "dropout": trial.suggest_categorical(f"fnn_dropout_{i}", [True, False])
            }
            if layer["dropout"]:
                layer["dropout_rate"] = trial.suggest_float(f"fnn_dropout_rate_{i}", 0.2, 0.5)
            fnn_config.append(layer)

        # Head layers
        head_config = []
        n_head = trial.suggest_int("n_head_layers", 1, 2)
        for i in range(n_head):
            layer = {
                "units": trial.suggest_categorical(f"head_units_{i}", [64, 128, 256]),
                "batchnorm": trial.suggest_categorical(f"head_bn_{i}", [True, False]),
                "dropout": trial.suggest_categorical(f"head_dropout_{i}", [True, False])
            }
            if layer["dropout"]:
                layer["dropout_rate"] = trial.suggest_float(f"head_dropout_rate_{i}", 0.2, 0.5)
            head_config.append(layer)

        learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model = ModelFactory.build_cnn(
            image_shape=input_shape,
            conv_config_layers=conv_config,
            fnn_config_layers=fnn_config,
            head_config_layers=head_config,
            aux_input_shapes=aux_shapes or None,
            output_dir=output_dir,
            output_units=2
        )

        # image augmentation 
        use_augment = trial.suggest_categorical("use_augmentation", [True, False])
        if use_augment:
            train_set.augment_images()


        trainer = Trainer(model, output_dir="optuna_tmp")
        trainer.compile(optimizer=Adam(learning_rate), loss=Trainer.loss_nll, metrics=[Trainer.mse_mean])
        history = trainer.train(
            inputs_train, y_train,
            inputs_val, y_val,
            batch_size=64,
            epochs=100,
            patience=20
        )

        return history.history["val_mse_mean"][-1]
    except Exception as e:
        print(f"[Trial failed] {e}")
        return float('inf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optuna optimizer")
    parser.add_argument("--out-dir", default="optuna_results", help="Directory to save results")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--study-name", default="cnn_hyperparam_search", help="Name of study")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)


    # Config to save results of study
    STUDY_NAME = args.study_name
    OUTPUT_DIR = args.out_dir  
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    STORAGE_NAME = f"sqlite:///{os.path.join(OUTPUT_DIR, STUDY_NAME)}.db"

    # Logging Optuna output to stdout
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Persistent study with SQLite (saves automatically)
    sampler = TPESampler(n_startup_trials=20)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True
    )

    # Optimization loop with periodic CSV save
    for _ in range(args.n_trials):
        study.optimize(lambda trial: objective(trial, args.out_dir), n_trials=1)

        # Save CSV every CHECKPOINT_EVERY trials
        if len(study.trials) % CHECKPOINT_EVERY == 0:
            save_checkpoint(study, args.out_dir)

    # Final checkpoint
    save_checkpoint(study, args.out_dir)

    # Save best trial info
    best = study.best_trial
    with open(os.path.join(args.out_dir, "best_trial.txt"), "w") as f:
        f.write(f"Best trial:\n")
        f.write(f"  Trial number: {best.number}\n")
        f.write(f"  Validation MSE: {best.value:.4f}\n")
        f.write("  Hyperparameters:\n")
        for key, value in best.params.items():
            f.write(f"    {key}: {value}\n")

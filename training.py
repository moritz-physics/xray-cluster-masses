import os
import tensorflow as tf
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.gridspec as gridspec
from preprocessing import Preprocessor

class Trainer():

    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
        self.history = None
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def loss_nll(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Computes the negative log-likelihood (NLL)
        
        Parameters
        ----------
        y_true : tf.Tensor
            True target values. Shape: (batch_size,) or (batch_size, 1).
        y_pred : tf.Tensor
            Predicted mean and variance for each input, as output by the network.
            Shape: (batch_size, 2), where y_pred[:, 0] = mean, y_pred[:, 1] = variance.
        
        Returns
        -------
        tf.Tensor
            Scalar tensor representing the mean NLL loss over the batch.
        """
        y_true = tf.squeeze(y_true)  # shape (batch_size,)
        mu = y_pred[:, 0]            # predicted means
        var = y_pred[:, 1]           # predicted variances (must be positive)
        jitter = 1e-8                # numerical stability
        nll = 0.5 * tf.math.log(var + jitter) + 0.5 * tf.square(y_true - mu) / (var + jitter)
        return tf.reduce_mean(nll)


    @staticmethod
    def mse_mean(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the mean squared error (MSE) between true and predicted values.

        Parameters
        ----------
        y_true : tf.Tensor or np.ndarray
            True target values, shape (batch,) or (batch, 1).
        y_pred : tf.Tensor or np.ndarray
            Model predictions, shape (batch, 2). Only the first column (mean) is used.

        Returns
        -------
        mse : tf.Tensor
            Mean squared error between predicted means and true values.
        """
        y_true = tf.squeeze(y_true)       # handle (batch, 1) or (batch,)
        mu = y_pred[:, 0]
        return tf.reduce_mean(tf.square(y_true - mu))



    def compile(
        self,
        optimizer: str | tf.keras.optimizers.Optimizer = "adam",
        loss: str | callable = "mse",
        metrics: list = ["mse"]
            ):
        """
        Compile the Keras model with the specified optimizer, loss function, and metrics.

        Parameters
        ----------
        optimizer : str or tf.keras.optimizers.Optimizer, default="adam"
            Optimizer to use for training.
        loss : str or callable, default="mse"
            Loss function to optimize.
        metrics : list, default=["mse"]
            List of metrics to compute during training.
        """
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def train(
        self,
        x_train: np.ndarray | dict,
        y_train: np.ndarray,
        x_val: np.ndarray | dict,
        y_val: np.ndarray,
        batch_size: int = 64,
        epochs: int = 40,
        patience: int = 10
        ) -> tf.keras.callbacks.History:
        """
        Train the model on the provided data with early stopping.

        Parameters
        ----------
        x_train : np.ndarray or dict
            Training inputs.
        y_train : np.ndarray
            Training targets.
        x_val : np.ndarray or dict
            Validation inputs.
        y_val : np.ndarray
            Validation targets.
        batch_size : int, default=64
            Batch size for training.
        epochs : int, default=40
            Number of training epochs.
        patience : int, default=10
            Early stopping patience.

        Returns
        -------
        tf.keras.callbacks.History
            Training history object.
        """
        early_stopping = EarlyStopping(
            patience=patience,
            restore_best_weights=True
        )        
        
        fit_args = dict(
            x=x_train, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stopping])

        print("========= Start of training =========\n")
        self.history = self.model.fit(**fit_args)
        print("========= End of training =========\n")

        return self.history

    def evaluate(self, x_val: np.ndarray | dict, y_val: np.ndarray) -> tuple[float, float]:
        """Evaluate the model on validation data.

        Parameters
        ----------
        x_val : np.ndarray or dict
            Validation inputs.
        y_val : np.ndarray
            Validation targets.

        Returns
        -------
        tuple[float, float]
            Validation loss and validation mean squared error.
        """

        val_loss, val_mse = self.model.evaluate(x_val,  y_val, verbose=2)
        print(f"Val loss: {val_loss:.4f}, Test MSE: {val_mse:.4f}")
        return val_loss, val_mse


    def predict(self, x: np.ndarray | dict) -> np.ndarray:
        """
        Obtain model predictions for input data.

        Parameters
        ----------
        x : np.ndarray or dict
            Input data.

        Returns
        -------
        np.ndarray
            Model predictions.
        """

        return self.model.predict(x)

    # --------------------------------------------------------------------
    # Plotters
    # --------------------------------------------------------------------  


    @staticmethod
    def plot_training_curves(
        history: tf.keras.callbacks.History,
        output_dir: str,
        filename: str = "metrics_vs_epoch.png"
        ) -> None:
        """
        Plot the training and validation MSE and NLL loss curves over training epochs.

        Parameters
        ----------
        history : tf.keras.callbacks.History
            Training history object returned by model.fit().
        output_dir : str
            Directory to save the plot.
        filename : str, default="metrics_vs_epoch.png"
            Name of the output plot file.

        Returns
        -------
        None
        """        
        val_mse = history.history["val_mse_mean"]
        train_mse = history.history["mse_mean"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(val_mse) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Left: MSE curves
        axs[0].set_title("Training and Validation MSE over Epochs")
        axs[0].plot(epochs, train_mse, label="Train MSE")
        axs[0].plot(epochs, val_mse, label="Val MSE", linestyle='--')
        best_epoch = np.argmin(val_mse)
        best_val = val_mse[best_epoch]
        axs[0].scatter(best_epoch + 1, best_val, color="red")
        axs[0].annotate(
            f"Best Val MSE\n{best_val:.4f} at epoch {best_epoch + 1}",
            xy=(best_epoch + 1, best_val),
            xytext=(best_epoch + 1, best_val + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10
        )
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Mean Squared Error")
        axs[0].legend()
        axs[0].grid(True)

        # Right: Loss curves
        axs[1].set_title("Training and Validation NLL Loss over Epochs")
        axs[1].plot(epochs, loss, label="Train NLL")
        axs[1].plot(epochs, val_loss, label="Val NLL", linestyle='--')
        best_loss_epoch = np.argmin(val_loss)
        best_val_loss = val_loss[best_loss_epoch]
        axs[1].scatter(best_loss_epoch + 1, best_val_loss, color="red")
        axs[1].annotate(
            f"Best Val NLL\n{best_val_loss:.4f} at epoch {best_loss_epoch + 1}",
            xy=(best_loss_epoch + 1, best_val_loss),
            xytext=(best_loss_epoch + 1, best_val_loss + 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05),
            fontsize=10
        )
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Negative Log-Likelihood")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

        
    @staticmethod
    def check_uncertainty(y_test: np.ndarray, y_pred: np.ndarray, sigma_pred: np.ndarray) -> float:
        """Computes the coverage of predicted uncertainty intervals.

        For each test example, checks if the true value lies within the 
        predicted interval [y_pred - sigma_pred, y_pred + sigma_pred].
        Returns the fraction of samples where this is true, i.e., the 
        coverage within ±1σ.

        Parameters
        ----------
        y_test : np.ndarray
            Array of true target values
        y_pred : np.ndarray
            Array of predicted mean values
        sigma_pred : np.ndarray
            Array of predicted standard deviations

        Returns
        -------
        coverage : float
            Fraction of samples for which the true value falls within the 
            predicted ±1σ interval.

        Prints
        ------
        Coverage within ±σ as a percentage.
        """
        lower_bound = y_pred - sigma_pred
        upper_bound = y_pred + sigma_pred

        within_interval = (y_test >= lower_bound) & (y_test <= upper_bound)
        coverage = np.mean(within_interval)

        print(f"Coverage within ±σ: {coverage * 100:.2f}%")
        return coverage


    @staticmethod
    def plot_predictions(
        y_pred: np.ndarray,
        y_true: np.ndarray,
        output_dir: str,
        plot_name: str = "mass_pred_test.png",
        prep: object = None
            ) -> None:
        """
        Plot predicted vs true values, error bars, ±1σ coverage, and pull distribution.

        Parameters
        ----------
        y_pred : np.ndarray
            Model predictions, shape (N, 2). First column is mean, second is variance.
        y_true : np.ndarray
            True target values, shape (N,).
        output_dir : str
            Directory to save the generated plot.
        plot_name : str, default="mass_pred_test.png"
            Filename for the output plot.
        prep : object, optional
            Preprocessor instance, used to denormalize values if provided.

        Returns
        -------
        None
        """
        y_pred = np.array(y_pred)
        y_true = np.array(y_true).flatten()
        means = y_pred[:, 0]
        vars_ = y_pred[:, 1]
        stds = np.sqrt(vars_)

        # Denormalize if needed
        if prep is not None:
            means = prep.denormalize_feature("mass", means)
            y_true = prep.denormalize_feature("mass", y_true)
            stds = stds * (prep.feature_stats["mass"]["std"] + 1e-8)
        
        # Calculate coverage
        coverage = Trainer.check_uncertainty(y_true, means, stds) 

        # Sort by true mass for a smooth band
        idx_sort = np.argsort(y_true)
        x_sorted = y_true[idx_sort]
        mean_sorted = means[idx_sort]
        std_sorted = stds[idx_sort]

        # Plot
        fig = plt.figure(figsize=(7, 7))
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
        ax_main = plt.subplot(gs[0])

        # Shaded region for ±1σ region around y=x
        ax_main.fill_between(
            x_sorted, x_sorted - std_sorted, x_sorted + std_sorted,
            color="#dbeeff", alpha=0.4, label="±1σ region (around y=x)"
        )

        # Scatter points, coloring based on whether inside/outside
        inside = (y_true >= means - stds) & (y_true <= means + stds)
        outside = ~inside
        ax_main.errorbar(
            y_true[inside], means[inside], yerr=stds[inside],
            fmt='o', color="#539ecd", alpha=0.5, label="Within ±1σ"
        )
        ax_main.errorbar(
            y_true[outside], means[outside], yerr=stds[outside],
            fmt='o', color="#ffb877", alpha=0.5, label="Outside ±1σ"
        )

        # y=x line
        ax_main.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", label="y = x")

        ax_main.set_ylabel(r"Predicted $\log_{10}(M_{500c}/M_\odot)$")
        ax_main.set_title(f"Predicted vs True Mass\nCoverage within ±σ: {coverage * 100:.1f}%")
        ax_main.legend()
        ax_main.grid(True)
        ax_main.set_xticklabels([])

        # Lower plot: Pull
        ax_res = plt.subplot(gs[1], sharex=ax_main)
        pull = (means - y_true) / stds
        ax_res.axhline(0, color="black", linestyle="--")
        ax_res.scatter(y_true, pull, alpha=0.5, s=10)
        # ax_res.errorbar(y_true, pull, yerr=np.std(pull))
        ax_res.set_xlabel(r"True $\log_{10}(M_{500c}/M_\odot)$")
        ax_res.set_ylim(-2, 2)
        ax_res.set_ylabel("Pull")
        ax_res.grid(True)
        plt.savefig(os.path.join(output_dir, plot_name))
        plt.close()

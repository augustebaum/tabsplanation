"""
Port of REVISE algorithm from CARLA repository catalog.
This is because installing the `carla-recourse` package into
this project doesn't work because of conflicts. The
dependency hell is strong with this one.

You can find the original implementation here:
<https://github.com/carla-recourse/CARLA/blob/9595d4f6609ff604bc22d9b8e6cd728ecf18737b/carla/recourse_methods/catalog/revise/model.py>
"""

from typing import Dict, Optional

import torch
from torch import nn

from tabsplanation.models.autoencoder import AutoEncoder
from tabsplanation.models.classifier import Classifier
from tabsplanation.types import ExplanationPath, InputOutputPair, InputPoint


class Revise:
    """
    Implementation of Revise from Joshi et.al. [1]_.

    Parameters
    ----------
    mlmodel : carla.model.MLModel
        Black-Box-Model
    data: carla.data.Data
        Dataset to perform on
    hyperparams : dict
        Dictionary containing hyperparameters. See notes below for its contents.

    Methods
    -------
    get_counterfactuals:
        Generate counterfactual examples for given factuals.

    Notes
    -----
    - Hyperparams
        Hyperparameter contains important information for the recourse method to
        initialize.
        Please make sure to pass all values as dict with the following keys.

        * "data_name": str
            name of the dataset
        * "lambda": float, default: 0.5
            Decides how similar the counterfactual is to the factual
        * "optimizer": {"adam", "rmsprop"}
            Optimizer for generation of counterfactuals.
        * "lr": float, default: 0.1
            Learning rate for Revise.
        * "max_iter": int, default: 1000
            Number of iterations for Revise optimization.
        * "target_class": List, default: [0, 1]
            List of one-hot-encoded target class.
        * "binary_cat_features": bool, default: True
            If true, the encoding of x is done by drop_if_binary.
        * "vae_params": Dict
            With parameter for VAE.

            + "layers": list
                Number of neurons and layer of autoencoder.
            + "train": bool
                Decides if a new autoencoder will be learned.
            + "lambda_reg": flot
                Hyperparameter for variational autoencoder.
            + "epochs": int
                Number of epochs to train VAE
            + "lr": float
                Learning rate for VAE training
            + "batch_size": int
                Batch-size for VAE training

    .. [1] Shalmali Joshi, Oluwasanmi Koyejo, Warut Vijitbenjaronk, Been Kim, and Joydeep Ghosh.2019.
            Towards Realistic Individual Recourse and Actionable Explanations in Black-Box Decision Making Systems.
            arXiv preprint arXiv:1907.09615(2019).
    """

    # _DEFAULT_HYPERPARAMS = {
    #     "data_name": None,
    #     "lambda": 0.5,
    #     "optimizer": "adam",
    #     "lr": 0.1,
    #     "max_iter": 1000,
    #     "target_class": [0, 1],
    #     "binary_cat_features": True,
    #     "vae_params": {
    #         "layers": None,
    #         "train": True,
    #         "lambda_reg": 1e-6,
    #         "epochs": 5,
    #         "lr": 1e-3,
    #         "batch_size": 32,
    #     },
    # }

    def __init__(
        self, classifier: Classifier, autoencoder: AutoEncoder, hparams: Dict
    ) -> None:

        # super().__init__(classifier)
        # self._params = merge_default_parameters(hyperparams, self._DEFAULT_HYPERPARAMS)

        # self._target_column = data.target
        self._distance_reg = hparams["distance_regularization"]
        self._optimizer = hparams["optimizer"]
        self._lr = hparams["lr"]
        self._max_iter = hparams["max_iter"]
        # self._target_class = self._params["target_class"]
        # self._binary_cat_features = self._params["binary_cat_features"]

        # vae_params = self._params["vae_params"]
        # self.vae = VariationalAutoencoder(
        #     self._params["data_name"], vae_params["layers"], ml_model.get_mutable_mask()
        # )

        # if vae_params["train"]:
        #     self.vae.fit(
        #         xtrain=data.df[ml_model.feature_input_order],
        #         lambda_reg=vae_params["lambda_reg"],
        #         epochs=vae_params["epochs"],
        #         lr=vae_params["lr"],
        #         batch_size=vae_params["batch_size"],
        #     )
        # else:
        #     try:
        #         self.vae.load(data.df.shape[1] - 1)
        #     except FileNotFoundError as exc:
        #         raise FileNotFoundError(
        #             "Loading of Autoencoder failed. {}".format(str(exc))
        #         )
        self.classifier = classifier
        self.autoencoder = autoencoder

    # def get_counterfactuals(self, factuals: pd.DataFrame) -> pd.DataFrame:
    def get_counterfactuals(
        self,
        input: InputPoint,
        target_class: Optional[int],
    ) -> ExplanationPath:

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # factuals = self.classifier.get_ordered_features(factuals)

        # pay attention to categorical features
        # encoded_feature_names = self.classifier.data.encoder.get_feature_names(
        #     self.classifier.data.categorical
        # )
        # cat_features_indices = [
        #     factuals.columns.get_loc(feature) for feature in encoded_feature_names
        # ]

        # list_cfs = self._counterfactual_optimization(
        #     cat_features_indices, device, factuals
        # )
        cfs = self._counterfactual_optimization(input, target_class)
        return cfs

        # cf_df = check_counterfactuals(self.classifier, list_cfs, factuals.index)
        # cf_df = self.classifier.get_ordered_features(cf_df)
        # return cf_df

    # def _counterfactual_optimization(self, cat_features_indices, device, df_fact):
    def _counterfactual_optimization(self, input, target_class):
        # prepare data for optimization steps
        # test_loader = torch.utils.data.DataLoader(
        #     df_fact.values, batch_size=1, shuffle=False
        # )

        cfs = []
        # for query_instance in test_loader:
        # query_instance = query_instance.float()

        # target = torch.FloatTensor(self._target_class).to(device)
        # target_prediction = np.argmax(np.array(self._target_class))

        # encode the mutable features
        # z = self.autoencoder.encode(query_instance[:, self.vae.mutable_mask])[0]
        z = self.autoencoder.encode(input.reshape(1, -1))
        # add the immutable features to the latents
        # z = torch.cat([z, query_instance[:, ~self.vae.mutable_mask]], dim=-1)
        z = z.clone().detach().requires_grad_(True)

        if self._optimizer == "adam":
            optim = torch.optim.Adam([z], self._lr)
        else:
            optim = torch.optim.RMSprop([z], self._lr)

        # candidate_counterfactuals = []  # all possible counterfactuals
        # distance of the possible counterfactuals from the intial value -
        # considering distance as the loss function (can even change it just the distance)
        # candidate_distances = []
        # losses = []

        for _ in range(self._max_iter):

            cf_x = self.autoencoder.decode(z)
            cfs.append(InputOutputPair(cf_x, self.classifier.predict_proba(cf_x)))

            # add the immutable features to the reconstruction
            # temp = query_instance.clone()
            # temp[:, self.vae.mutable_mask] = cf
            # cf = temp

            # cf = reconstruct_encoding_constraints(
            #     cf, cat_features_indices, self._params["binary_cat_features"]
            # )
            # output = self.classifier.predict_proba(cf)[0]
            # _, predicted = torch.max(output, 0)

            # z.requires_grad = True
            loss, logs = self._compute_loss(input, cf_x, target_class)
            # losses.append(logs)

            # if predicted == target_prediction:
            #     candidate_counterfactuals.append(
            #         cf.cpu().detach().numpy().squeeze(axis=0)
            #     )
            #     candidate_distances.append(loss.cpu().detach().numpy())

            loss.backward()
            optim.step()
            optim.zero_grad()
            cf_x.detach_()

            # Choose the nearest counterfactual
            # if len(candidate_counterfactuals):
            #     # log.info("Counterfactual found!")
            #     array_counterfactuals = np.array(candidate_counterfactuals)
            #     array_distances = np.array(candidate_distances)
            #     index = np.argmin(array_distances)
            #     list_cfs.append(array_counterfactuals[index])
            # else:
            #     # log.info("No counterfactual found")
            #     list_cfs.append(query_instance.cpu().detach().numpy().squeeze(axis=0))

        return ExplanationPath(
            explained_input=InputOutputPair(
                input, self.classifier.predict_proba(input)
            ),
            target_class=target_class,
            shift_step=None,
            max_iter=self._max_iter,
            cfs=cfs,
        )

    def _compute_loss(self, original_input, cf, target_class):

        # For multi-class tasks
        loss_function = nn.CrossEntropyLoss()
        output = self.classifier.predict_proba(original_input)

        classification_loss = loss_function(output, target_class)
        distance_loss = torch.norm((original_input - cf), 1)

        loss = classification_loss + self._distance_reg * distance_loss
        logs = {
            "loss": loss,
            "classification_loss": classification_loss,
            "distance_loss": distance_loss,
        }

        return loss, logs

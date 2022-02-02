import sklearn
import numpy as np

from tqdm import tqdm


class ShadowModelBundle(sklearn.base.BaseEstimator):
    
    MODEL_ID_FMT = "shadow_%d"

    def __init__(
        self, model_fn, shadow_dataset_size, num_models=20, seed=42, serializer=None
    ):
        super().__init__()
        self.model_fn = model_fn
        self.shadow_dataset_size = shadow_dataset_size
        self.num_models = num_models
        self.seed = seed
        self.serializer = serializer
        self._reset_random_state()

    def fit_transform(self, X, y, verbose=False, fit_kwargs=None):
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs)
        return self._transform(verbose=verbose)

    def _reset_random_state(self):
        self._prng = np.random.RandomState(self.seed)

    def _get_model_iterator(self, indices=None, verbose=False):
        if indices is None:
            indices = range(self.num_models)
        if verbose:
            indices = tqdm(indices)
        return indices

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = ShadowModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.shadow_models_[model_index]
        return model

    def _fit(self, X, y, verbose=False, pseudo=False, fit_kwargs=None):
        self.shadow_train_indices_ = []
        self.shadow_test_indices_ = []

        if self.serializer is None:
            self.shadow_models_ = []

        fit_kwargs = fit_kwargs or {}
        indices = np.arange(X.shape[0])

        for i in self._get_model_iterator(verbose=verbose):
            # Pick indices for this shadow model.
            shadow_indices = self._prng.choice(
                indices, 2 * self.shadow_dataset_size, replace=False
            )
            train_indices = shadow_indices[: self.shadow_dataset_size]
            test_indices = shadow_indices[self.shadow_dataset_size :]
            X_train, y_train = X[train_indices], y[train_indices]
            self.shadow_train_indices_.append(train_indices)
            self.shadow_test_indices_.append(test_indices)

            if pseudo:
                continue

            # Train the shadow model.
            shadow_model = self.model_fn()
            shadow_model.fit(X_train, y_train)
            if self.serializer is not None:
                self.serializer.save(ShadowModelBundle.MODEL_ID_FMT % i, shadow_model)
            else:
                self.shadow_models_.append(shadow_model)

        self.X_fit_ = X
        self.y_fit_ = y
        self._reset_random_state()
        return self

    def _pseudo_fit(self, X, y, verbose=False, fit_kwargs=None):
        self._fit(X, y, verbose=verbose, fit_kwargs=fit_kwargs, pseudo=True)

    def _transform(self, shadow_indices=None, verbose=False):
        shadow_data_array = []
        shadow_label_array = []

        model_index_iter = self._get_model_iterator(
            indices=shadow_indices, verbose=verbose
        )

        for i in model_index_iter:
            shadow_model = self._get_model(i)
            train_indices = self.shadow_train_indices_[i]
            test_indices = self.shadow_test_indices_[i]

            train_data = self.X_fit_[train_indices], self.y_fit_[train_indices]
            test_data = self.X_fit_[test_indices], self.y_fit_[test_indices]
            shadow_data, shadow_labels = prepare_attack_data(
                shadow_model, train_data, test_data
            )

            shadow_data_array.append(shadow_data)
            shadow_label_array.append(shadow_labels)

        X_transformed = np.vstack(shadow_data_array).astype("float32")
        y_transformed = np.hstack(shadow_label_array).astype("float32")
        return X_transformed, y_transformed


def prepare_attack_data(model, data_in, data_out):
    X_in, y_in = data_in
    X_out, y_out = data_out
    y_hat_in = model.predict(X_in)
    y_hat_out = model.predict(X_out)

    labels = np.ones(y_in.shape[0])
    labels = np.hstack([labels, np.zeros(y_out.shape[0])])
    # TODO: this does not work for non-one-hot labels.
    # data = np.hstack([y_hat_in, y_in])
    data = np.c_[y_hat_in, y_in]
    data = np.vstack([data, np.c_[y_hat_out, y_out]])
    return data, labels


class AttackModelBundle(sklearn.base.BaseEstimator):
    MODEL_ID_FMT = "attack_%d"

    def __init__(
        self, model_fn, num_classes, serializer=None, class_one_hot_coded=True
    ):
        self.model_fn = model_fn
        self.num_classes = num_classes
        self.serializer = serializer
        self.class_one_hot_coded = class_one_hot_coded

    def fit(self, X, y, verbose=False, fit_kwargs=None):
        X_total = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]
        
        inv_classes = np.zeros((classes.shape[0], 1))
        for i in range(classes.shape[0]):
            inv_classes[i][0] = int(not classes[i][0])
        classes = np.append(classes, inv_classes, axis=1)
        
        datasets_by_class = []
        data_indices = np.arange(X_total.shape[0])
        for i in range(self.num_classes):
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]

            datasets_by_class.append((X_total[class_indices], y[class_indices]))

        if self.serializer is None:
            self.attack_models_ = []

        dataset_iter = datasets_by_class
        if verbose:
            dataset_iter = tqdm(dataset_iter)
        for i, (X_train, y_train) in enumerate(dataset_iter):
            model = self.model_fn()
            fit_kwargs = fit_kwargs or {}
            model.fit(X_train, y_train, **fit_kwargs)

            if self.serializer is not None:
                model_id = AttackModelBundle.MODEL_ID_FMT % i
                self.serializer.save(model_id, model)
            else:
                self.attack_models_.append(model)

    def _get_model(self, model_index):
        if self.serializer is not None:
            model_id = AttackModelBundle.MODEL_ID_FMT % model_index
            model = self.serializer.load(model_id)
        else:
            model = self.attack_models_[model_index]
        return model

    def predict_proba(self, X):
        result = np.zeros((X.shape[0], 2))
        shadow_preds = X[:, : self.num_classes]
        classes = X[:, self.num_classes :]

        inv_classes = np.zeros((classes.shape[0], 1))
        for i in range(classes.shape[0]):
            inv_classes[i][0] = int(not classes[i][0])
        classes = np.append(classes, inv_classes, axis=1)
        
        data_indices = np.arange(shadow_preds.shape[0])
        for i in range(self.num_classes):
            model = self._get_model(i)
            if self.class_one_hot_coded:
                class_indices = data_indices[np.argmax(classes, axis=1) == i]
            else:
                class_indices = data_indices[np.squeeze(classes) == i]
            
            membership_preds = model.predict(shadow_preds[class_indices])
            for j, example_index in enumerate(class_indices):
                prob = np.squeeze(membership_preds[j])
                result[example_index, 1] = prob
                result[example_index, 0] = 1 - prob

        return result

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return probs > 0.5

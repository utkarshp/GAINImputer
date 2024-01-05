from torch import manual_seed as torch_manual_seed
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import gc
torch_manual_seed(421)
np.random.seed(421)
def cross_validate_imputer(imputer, base_imputer, X_complete, missing_rate=0.3, n_splits=10):
    kf = TimeSeriesSplit(n_splits=n_splits)
    rsquares = []
    # X_complete = MinMaxScaler().fit_transform(X_complete)
    scaler = MinMaxScaler()
    for it, (train_index, test_index) in enumerate(kf.split(X_complete)):
        gc.collect()
        # Split data into training and testing sets
        X_train_, X_test_ = X_complete.iloc[train_index], X_complete.iloc[test_index]

        # We want to artificially introduce missing data in X_test. But first, impute any already missing data.
        X_train_ = scaler.fit_transform(X_train_)
        base_imputer.fit(X_train_)  # We do not transform X_train. Only fit to get the reference base_imputer.
        X_test_base = base_imputer.transform(scaler.transform(X_test_))
        missing_samples = np.random.choice(X_test_.shape[0], int(np.floor(missing_rate * X_test_.shape[0])), replace=False)
        missing_features = np.random.choice(X_test_.shape[1], int(np.floor(missing_rate * X_test_.shape[1])), replace=False)
        X_test_missing = X_test_base.copy()
        X_test_missing.iloc[missing_samples, missing_features] = np.nan
        
        imputer.reset()
        # Fit the imputer on the training set and transform the test set
        imputer.fit(X_train_)
        X_test_imputed = imputer.transform(X_test_missing)
        # print(pd.DataFrame(X_test_imputed, columns=X_test_base.columns, index=X_test_base.index), X_test_base)
        # Calculate RMSE on the artificially missing part of the test set
        mask = from_numpy(X_test_missing.to_numpy()).isnan()
        # print(mask, X_test_base, X_test_imputed, X_test_imputed[mask], X_test_base.to_numpy()[mask])
        mse_ = mean_squared_error(X_test_imputed.to_numpy()[mask],
                                  X_test_base.to_numpy()[mask])
        r2_ = 1 - mse_ / np.var(X_test_base.to_numpy()[mask])
        rsquares.append(r2_)
        print(it, r2_, np.sqrt(mse_))
    return rsquares
test_frame  = pd.read_csv("Spam.csv", dtype=np.float32)
cross_validate_imputer(NNImputer(len(test_frame.columns), epochs=4000, verbose=False, batch_size=128), SimpleImputer(strategy='mean'),
                       test_frame,
                       )

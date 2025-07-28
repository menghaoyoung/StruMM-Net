import os
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras import Model
import numpy as np
import random
from sklearn.metrics import mean_absolute_error


def load_model(pretrained_model_dir, flag):
    if flag:
        model = tf.keras.models.load_model(pretrained_model_dir, compile=False)
    else:
        base_model = tf.keras.models.load_model(pretrained_model_dir, compile=False)
        conductivity_out = base_model.get_layer("conductivity_output").output
        model = Model(inputs=base_model.inputs, outputs=conductivity_out)
    print("Model loaded successfully!\n")
    return model


def load_scalers():
    scaler_scalar = joblib.load(os.path.join("pretrained_models", "scalar_scaler.pkl"))
    scaler_sine = joblib.load(os.path.join("pretrained_models", "sine_scaler.pkl"))
    scaler_acsf = joblib.load(os.path.join("pretrained_models", "acsf_scaler.pkl"))
    sine_encoder = tf.keras.models.load_model(os.path.join("pretrained_models", "sine_encoder.h5"), compile=False)
    acsf_encoder = tf.keras.models.load_model(os.path.join("pretrained_models", "acsf_encoder.h5"), compile=False)

    return scaler_scalar, scaler_sine, scaler_acsf, sine_encoder, acsf_encoder


def predict(pretrained_model, flag):
    pretrained_model_dir = os.path.join("pretrained_models", "SINNet_" + pretrained_model)
    scaler_scalar, sine_scaler, acsf_scaler, sine_encoder, acsf_encoder = load_scalers()
    folder_path = "extracted_features/"
    rough_est_path = os.path.join("predict_by_formula", pretrained_model+".csv")
    df_predict_by_formula = pd.read_csv(rough_est_path)
    est_map = df_predict_by_formula.set_index('name')['pred'].to_dict()

    train_list = pd.read_csv("train.csv")['structure files']

    x_train_xrd = pd.DataFrame()
    x_train_mRDF = pd.DataFrame()
    x_train_SineMatrix = pd.DataFrame()
    x_train_mACSF = pd.DataFrame()
    x_train_scalar = pd.DataFrame()
    y_train = pd.DataFrame()

    for item in train_list:
        name = item[:-4]
        path = os.path.join(folder_path, name, 'origin.txt')
        crude_est = est_map[item]

        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        y_train = pd.concat([y_train, pd.DataFrame(lines[0].split()[1:], dtype=float).T], ignore_index=True)
        x_train_xrd = pd.concat([x_train_xrd, pd.DataFrame(lines[2].split()[1:], dtype=float).T], ignore_index=True)
        x_train_mRDF = pd.concat([x_train_mRDF, pd.DataFrame(lines[1].split()[1:], dtype=float).T], ignore_index=True)
        x_train_SineMatrix = pd.concat([x_train_SineMatrix, pd.DataFrame(lines[3].split()[1:], dtype=float).T],
                                       ignore_index=True)
        x_train_mACSF = pd.concat([x_train_mACSF, pd.DataFrame(lines[4].split()[1:], dtype=float).T],
                                  ignore_index=True)
        x_train_scalar = pd.concat([x_train_scalar, pd.DataFrame([crude_est] + lines[5].split()[1:], dtype=float).T],
                                   ignore_index=True)

    val_list = pd.read_csv("val.csv")['structure files']

    x_val_xrd = pd.DataFrame()
    x_val_mRDF = pd.DataFrame()
    x_val_SineMatrix = pd.DataFrame()
    x_val_mACSF = pd.DataFrame()
    x_val_scalar = pd.DataFrame()
    y_val = pd.DataFrame()

    for item in val_list:
        name = item[:-4]
        path = os.path.join(folder_path, name, 'origin.txt')
        crude_est = est_map[item]
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        y_val = pd.concat([y_val, pd.DataFrame(lines[0].split()[1:], dtype=float).T], ignore_index=True)
        x_val_xrd = pd.concat([x_val_xrd, pd.DataFrame(lines[2].split()[1:], dtype=float).T], ignore_index=True)
        x_val_mRDF = pd.concat([x_val_mRDF, pd.DataFrame(lines[1].split()[1:], dtype=float).T], ignore_index=True)
        x_val_SineMatrix = pd.concat([x_val_SineMatrix, pd.DataFrame(lines[3].split()[1:], dtype=float).T],
                                     ignore_index=True)
        x_val_mACSF = pd.concat([x_val_mACSF, pd.DataFrame(lines[4].split()[1:], dtype=float).T],
                                ignore_index=True)
        x_val_scalar = pd.concat([x_val_scalar, pd.DataFrame([crude_est] + lines[5].split()[1:], dtype=float).T],
                                 ignore_index=True)

    test_list = pd.read_csv("test.csv")['structure files']

    x_test_xrd = pd.DataFrame()
    x_test_mRDF = pd.DataFrame()
    x_test_SineMatrix = pd.DataFrame()
    x_test_mACSF = pd.DataFrame()
    x_test_scalar = pd.DataFrame()
    y_test = pd.DataFrame()

    for item in test_list:
        name = item[:-4]
        path = os.path.join(folder_path, name, 'origin.txt')
        crude_est = est_map[item]
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        y_test = pd.concat([y_test, pd.DataFrame(lines[0].split()[1:], dtype=float).T], ignore_index=True)
        x_test_xrd = pd.concat([x_test_xrd, pd.DataFrame(lines[2].split()[1:], dtype=float).T], ignore_index=True)
        x_test_mRDF = pd.concat([x_test_mRDF, pd.DataFrame(lines[1].split()[1:], dtype=float).T], ignore_index=True)
        x_test_SineMatrix = pd.concat([x_test_SineMatrix, pd.DataFrame(lines[3].split()[1:], dtype=float).T],
                                      ignore_index=True)
        x_test_mACSF = pd.concat([x_test_mACSF, pd.DataFrame(lines[4].split()[1:], dtype=float).T],
                                 ignore_index=True)
        x_test_scalar = pd.concat([x_test_scalar, pd.DataFrame([crude_est] + lines[5].split()[1:], dtype=float).T],
                                  ignore_index=True)

    rough_est_test = x_test_scalar.iloc[:, 0].to_numpy()

    XRD_train = np.array(x_train_xrd).reshape((-1, 4001, 1)).astype(np.float32)
    XRD_val = np.array(x_val_xrd).reshape((-1, 4001, 1)).astype(np.float32)
    XRD_test = np.array(x_test_xrd).reshape((-1, 4001, 1)).astype(np.float32)

    RDF_train = np.array(x_train_mRDF).reshape((-1, 400, 1)).astype(np.float32)
    RDF_val = np.array(x_val_mRDF).reshape((-1, 400, 1)).astype(np.float32)
    RDF_test = np.array(x_test_mRDF).reshape((-1, 400, 1)).astype(np.float32)

    SINE_train = np.array(x_train_SineMatrix).reshape((-1, 40000, 1)).astype(np.float32)
    SINE_val = np.array(x_val_SineMatrix).reshape((-1, 40000, 1)).astype(np.float32)
    SINE_test = np.array(x_test_SineMatrix).reshape((-1, 40000, 1)).astype(np.float32)
    sine_train_2d = SINE_train.reshape(-1, 40000)
    sine_train_norm = sine_scaler.transform(sine_train_2d)
    SINE_train_norm = sine_train_norm.reshape(-1, 40000, 1)
    sine_val_2d = SINE_val.reshape(-1, 40000)
    SINE_val_norm = sine_scaler.transform(sine_val_2d).reshape(-1, 40000, 1)
    sine_test_2d = SINE_test.reshape(-1, 40000)
    SINE_test_norm = sine_scaler.transform(sine_test_2d).reshape(-1, 40000, 1)

    ACSF_train = np.array(x_train_mACSF).reshape((-1, 5 * 3021)).astype(np.float32)
    ACSF_val = np.array(x_val_mACSF).reshape((-1, 5 * 3021)).astype(np.float32)
    ACSF_test = np.array(x_test_mACSF).reshape((-1, 5 * 3021)).astype(np.float32)
    ACSF_train_norm = acsf_scaler.transform(ACSF_train)
    ACSF_val_norm = acsf_scaler.transform(ACSF_val)
    ACSF_test_norm = acsf_scaler.transform(ACSF_test)
    ACSF_train_norm = np.array(ACSF_train_norm).reshape((-1, 5, 3021)).astype(np.float32)
    ACSF_val_norm = np.array(ACSF_val_norm).reshape((-1, 5, 3021)).astype(np.float32)
    ACSF_test_norm = np.array(ACSF_test_norm).reshape((-1, 5, 3021)).astype(np.float32)

    Scalar_train_scaled = scaler_scalar.transform(x_train_scalar)
    Scalar_val_scaled = scaler_scalar.transform(x_val_scalar)
    Scalar_test_scaled = scaler_scalar.transform(x_test_scalar)

    SINE_train_norm = sine_encoder.predict(SINE_train_norm)
    SINE_val_norm = sine_encoder.predict(SINE_val_norm)
    SINE_test_norm = sine_encoder.predict(SINE_test_norm)
    ACSF_train_norm = acsf_encoder.predict(ACSF_train_norm)
    ACSF_val_norm = acsf_encoder.predict(ACSF_val_norm)
    ACSF_test_norm = acsf_encoder.predict(ACSF_test_norm)

    model = load_model(pretrained_model_dir, flag)
    if flag:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_conductivity_output_mae', patience=5,
                                                      restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_conductivity_output_mae',
            factor=0.2,
            patience=8,
            verbose=1,
            min_lr=1e-6
        )

        # 由于模型有3个输出，这里构造训练及验证的目标字典
        y_train_dict = {
            'conductivity_output': y_train,
            'aux_1': y_train,
            'aux_2': y_train
        }
        y_val_dict = {
            'conductivity_output': y_val,
            'aux_1': y_val,
            'aux_2': y_val
        }

        y_train_np = y_train.values.flatten()
        weight = 2.0
        train_weights = np.where(y_train_np < -11, weight, 1.0)

        # 2）因为你的模型有三个输出，我们用 dict 的形式分别传给每个输出
        sample_weight_dict = {
            'conductivity_output': train_weights,
            'aux_1': train_weights,
            'aux_2': train_weights
        }
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.002),
            loss={
                'conductivity_output': 'mse',
                'aux_1': 'mse',
                'aux_2': 'mse'
            },
            loss_weights={
                'conductivity_output': 1.0,
                'aux_1': 0.2,
                'aux_2': 0.1
            },
            metrics={'conductivity_output': ['mae']},
            weighted_metrics=[]
        )
        model.fit(
            {
                'xrd_input': XRD_train,
                'rdf_input': RDF_train,
                'sinematrix_input': SINE_train_norm,
                'acsf_input': ACSF_train_norm,
                'scalar_input': Scalar_train_scaled
            },
            y_train_dict,
            sample_weight=sample_weight_dict,
            epochs=100,
            batch_size=16,
            validation_data=(
                {
                    'xrd_input': XRD_val,
                    'rdf_input': RDF_val,
                    'sinematrix_input': SINE_val_norm,
                    'acsf_input': ACSF_val_norm,
                    'scalar_input': Scalar_val_scaled
                },
                y_val_dict,
            ),
            callbacks=[early_stop],
            verbose=1
        )

    test_input = {
        'xrd_input': XRD_test,
        'rdf_input': RDF_test,
        'sinematrix_input': SINE_test_norm,
        'acsf_input': ACSF_test_norm,
        'scalar_input': Scalar_test_scaled
    }

    test_preds = model.predict(test_input)
    if flag:
        test_preds = test_preds[0].reshape(-1)
    else:
        test_preds = test_preds.flatten()
    test_mae = mean_absolute_error(y_test, test_preds)
    print(f"\nMAE of prediction: {test_mae:.4f}")
    est_mae = mean_absolute_error(y_test, rough_est_test)
    print(f"\nMAE of rough estimation: {est_mae:.4f}")
    df_out = pd.DataFrame({
        'name': test_list.str[:-4],
        'rough_estimate_by_formula': rough_est_test,
        'true': y_test.values.flatten(),
        'predict': test_preds
    })
    df_out.to_csv("trueVSpredictVSrough_est.csv", index=False)
    print("\nResults have been saved in 'trueVSpredictVSrough_est.csv' !")


if __name__ == '__main__':
    seed = 22
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # pretrained models to choose: 'catboost', 'crabnet', 'roost'
    model = 'catboost'

    # whether to fine-tune the model or not
    # if set fine_tuning = True, all parameters in lines 159 through 231 of the code can be adjusted
    fine_tuning = False

    predict(model, fine_tuning)

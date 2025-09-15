# StruMM-Net
A general structure-based model to predict ionic conductivity of lithium-containing inorganic compounds

# Installation
```python
conda create -n your-env-name python=3.8.20
pip install -r requirements.txt
```

# Usage
### Data preprocessing
The processed data used for training this model are stored in the `extracted_features/` folder.

To preprocess your own data, run data_processing.py
```python
if __name__ == "__main__":
    np.random.seed(224)
    random.seed(224)
    cif_folder_path = "to_your_cif_folder_path/"
    out_path = "output/"
    process(cif_path=cif_folder_path, out_path=out_path)
```
Please place all CIF files in the same folder, specify the input and output folder paths, and then run the script.


### Predicting with Pretrained Models
Using the provided dataset and pretrained models, you can run and reproduce the prediction results with `predict.py`.
```python
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
```
Select the pretrained model you wish to use. Models to choose: 'catboost', 'crabnet', 'roost'

### Fine-tuning
If you want to fine-tuning the model on your own dataset, please prepare `train.csv`, `val.py` and `test.py` files, which should contain at least two columns: the log value of ionic conductivity `log_target` and the corresponding CIF file names `structure files`. The predicted results from composition-only models also should be prepared and placed in `predict_by_formula/`.

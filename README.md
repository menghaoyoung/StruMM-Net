# StruMM-Net
A general structure-based model to predict ionic conductivity of lithium-containing inorganic compounds

# Installation
```python
conda create -n your-env-name python=3.8.20
pip install -r requirements.txt
```

# Usage
### Data preprocessing
The processed data used for training this model are stored in the `extracted_features` folder.

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

### Fine-tuning

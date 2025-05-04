To determine the mean and standard deviation of dataset features:
```
python determine_affinity_stats.py -save_filepath /affinity/ds_stats.json
```

To partition the dataset into training, validation and test partitions:
```
python partition_dataset.py -data_folder /affinity/processed/ -save_folder /affinity/
```

To embed the dataset proteins with ProtBERT and dataset ligands with ChemBERTa:
```
python process_plapt_ds.py -stat_json_path /affinity/ds_stats.json -save_dir /affinity/processed/
```

To train a PLAPT model to predict binding affinity:
```
python train_plapt.py -data_folder /affinity/processed/ -csv_folder /affinity/csv/ -batch_size 64 -lr 1e-3 -num_epochs 64 -result_folder /affinity/models/ -model_save_name model_1.pth.tar -patience 5 -prot_hidden 512 -lig_hidden 512
```

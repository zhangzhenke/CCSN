### Cancer Cell instance Segmentation Network 用于苏木精-伊红染色的癌细胞实例分割网络



## CellFinder
细胞检测器，产生目标框供SAM模型提示产生精确的分割掩码。
```
📦CellFinder
 ┣ 📂PublicData
 ┃ ┣ 📜CellFinder.py
 ┃ ┣ 📜datasets.py
 ┃ ┣ 📜engine.py
 ┃ ┣ 📜label_flow_mask.py
 ┃ ┣ 📜main_train.py
 ┃ ┣ 📜transforms.py
 ┃ ┃ ┗ 📜ytils.py
 ┣ 📂PreTrain
 ┃ ┣ 📜engine_pretrain.py
 ┃ ┣ 📜main_pretrain.py
 ┃ ┗ 📜models_mae.py
 ┗ 📂CellBinDB
```
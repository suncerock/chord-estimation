# chord-estimation

A DNN-based chord estimation project

A FDUROP project

## Data Preparation

chord-estimation
|-- billboard
|   |-- simple-lab
|       |-- 0003
|       |-- 0004
|       |-- 0006
|       |-- ....
|   |-- chordino
|       |-- 0003
|       |-- 0004
|       |-- 0006
|       |-- ....
|-- preprocessed_data  // empty dir, data generated by data_preprocessed.py
|-- data_preprocess.py
|-- dataloader.py

Data can be downloaded from [The McGill Billboard Project](https://ddmal.music.mcgill.ca/research/The_McGill_Billboard_Project_(Chord_Analysis_Dataset)/). Only [billboard-2.0.1-mirex.tar.gz](https://www.dropbox.com/s/f88s73bmivlvbiy/billboard-2.0.1-mirex.tar.gz?dl=1) and [billboard-2.0-chordino.tar.gz](https://www.dropbox.com/s/f88s73bmivlvbiy/billboard-2.0.1-mirex.tar.gz?dl=1) are required for this project.

After downloading the data, uncompress and then rename them as is mentioned above. After that, generate the data by running

```
python data_preprocess.py
```

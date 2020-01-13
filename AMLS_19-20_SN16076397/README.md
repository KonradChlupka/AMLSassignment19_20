# README

### Project structure
Running "tree --filelimit 20" from the location of this README.md, we get the following structure.

```
.
├── AMLS_19-20_SN16076397
│   ├── A1
│   │   └── a1.py -> Contains the code which created CNN, trains, test.
│   ├── A2
│   │   └── a2.py -> Contains the code which created CNN, trains, test.
│   ├── B1
│   │   └── b1.py -> Contains the code which created CNN, trains, test.
│   ├── B2
│   │   └── b2.py -> Contains the code which created CNN, trains, test.
│   ├── Datasets
│   │   ├── cartoon_set
│   │   │   ├── img [10000 entries exceeds filelimit, not opening dir]
│   │   │   └── labels.csv
│   │   ├── cartoon_set_test
│   │   │   ├── img [2500 entries exceeds filelimit, not opening dir]
│   │   │   └── labels.csv
│   │   ├── celeba
│   │   │   ├── img [5000 entries exceeds filelimit, not opening dir]
│   │   │   └── labels.csv
│   │   └── celeba_test
│   │       ├── img [1000 entries exceeds filelimit, not opening dir]
│   │       └── labels.csv
│   ├── README.md -> The file you're reading now.
│   ├── common.py -> Contains functions used across a1.py, a2.py, b1,py, b2.py.
│   ├── main.py -> Runs a1.py, a2.py, b1,py, b2.py, prints results.
│   ├── main_jupyter.ipynb -> *The main file in which extensive data and model exploration are made.*
│   └── shape_predictor_68_face_landmarks.dat -> Used by dlib.
└── README.md -> The file you're reading now.

17 directories, 19 files
```
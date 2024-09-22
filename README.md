# AA_three-body_problem
Repository for the first assignment of Machine Learning Course

## Students:
<ul>
  <li>Filipe Colla David 70666</li>
  <li>Diogo Almeida 70140</li>
  <li>Duarte Rodrigues 70150</li>
</ul>


## Usage:
```bash
# Create a virtual env 
$ python3 -m venv venv
# Activate the virtual env (use tab to autocomplete)
$ source venv/bin/activate
# Install the requirements
(venv) $ pip install -r requirements.txt
```

The when opening the Jupyter Notebook, select the kernel of the virtual env

If any new libraries are added to the env
```bash
(venv) $ pip freeze > requirements.txt
```

## Dataset
The dataset will not be present in this repository, you must download it into the ```./data/``` folder<br>
The dataset is available at Kaggle.com in the [competition page](https://www.kaggle.com/competitions/machine-learning-nova-2024-the-three-body-proble/data)

### Kaggle token
A Kaggle token will allow you to easily download kaggle datasets.<br>

In the kaggle website, go to Profile -> Settings -> API -> Create New Token<br>
This will download a file called ```kaggle.json```.<br>
To see the expected file location run:
```bash
kaggle
```
Output
```txt
...
OSError: Could not find kaggle.json. Make sure it's located in [LOCATION]/.kaggle/
```
Place the file there, you might need to create a folder named ```.kaggle```.
You may need to change it's permissions.
```
chmod 600 [LOCATION]/.kaggle/kaggle.json
```
<b>Note:</b> This is your own private authentication key, don't share it with anyone.

Download the dataset
```
cd data
kaggle competitions download -c machine-learning-nova-2024-the-three-body-proble
unzip [dataset.zip]
```

### Alternative methods
Just download and place the data into the dataset folder, using the Download All option in the Data section of the kaggle competion.

### Structure of the data folder
<<<<<<< HEAD

=======
The following shows the expected structure of the data folder.
>>>>>>> 7054958 (Updated requirements.txt2)
```
.
├── mlNOVA
│   └── mlNOVA
│       ├── X_test.csv
│       └── X_train.csv
└── sample_submission.csv
```

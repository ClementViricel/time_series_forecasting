# time series forecasting
This repo is about teaching you time series forecasting using artificial neural network and though tensorflow 2.0.
If you want a detailled teaching class about this, please check this video :

https://www.youtube.com/watch?v=3eLpk8EAkM8&amp;ab_channel=JedhaBootcamp

## Package intallation
To install needed packages i.e `numpy`, `pandas`, `xlrd`, `tensorflow` and `plotly`,  
you can use any python package installer (this script should work with any version of the libs except maybe for tensorflow).
The list of packages and the current working versions are listed in `requirements.txt`.  
If you use `pip`, you can run the followinf command to install all the packages.  

`pip3 install -r requirements.txt`

## Run the training script
Main scripts are in `scripts`folder.  
* `preprocessing.py`contains functions to preprocess the input datas.  
* `models.py`contains model function (in tf.keras)
* `train.py`is the running script to train your model and get predictions.  

If you want to predict points from a list of points, you want to run `train.py`.  
This script requires some inputs. Let's see the detail of the required options:  
* -points_to_predict or -pp (int) : the number of points you want to predict. This will have an impact on the accuracy of the prediction. 
* -points_history or -ph (int) : the number of points. The number of points the model needs to know in order to do the prediction. It wiil deeply impact the prediction. Too many points will over-learn the model, too few points will make the learning really difficult. The number of history points depends on the data but it can be set to 30 at the beggining. 
* -rows (string) : the name of the input rows. If there is several rows, please also use the option target_row yo define which row is predicted. If there is only one row, you can just put the name of the row and it will automaticaly set the target_row to row.
* -f (xls, xlsx, csv, txt) : the input file. It must be either one of the following format excel or csv. It will impact the output format. if csv then csv ouput format, if excel then excel output format.

## Training options
There is a lots of other options that allow you to custom your traning.  
Warning : This options must be change when you are sure to understand the script. If it's not the case you can reach me by mail and i will explain the option for you with more details.  

List of options are available with `python3 train.py -h`  
Here is what you get:  
```
usage: train.py [-h] -file FILE -points_to_predict POINTS_TO_PREDICT
                -points_history POINTS_HISTORY [-rows ROWS]
                [-target_row TARGET_ROW] [-limit LIMIT]
                [-model_name MODEL_NAME] [-single] [-step STEP]
                [-epochs EPOCHS] [-batch_size BATCH_SIZE] [-loss LOSS]
                [-neurons NEURONS] [-optimizer OPTIMIZER]
                [-mean_models MEAN_MODELS] [-smooth SMOOTH] [-test_mode]
                [-verbose VERBOSE] [-plot_open]

optional arguments:
  -h, --help            show this help message and exit
  -file FILE, -f FILE   input file with the points (xls, csv or txt)
  -points_to_predict POINTS_TO_PREDICT, -pp POINTS_TO_PREDICT
                        Number of point to predict
  -points_history POINTS_HISTORY, -ph POINTS_HISTORY
                        Number of point which are history points
  -rows ROWS            name of the rows (if xls or csv) separate by a comma
                        for example "Free Gb,nb_of_data"
  -target_row TARGET_ROW
                        name of the row to predict
  -limit LIMIT          limit to the number of data
  -model_name MODEL_NAME, -m MODEL_NAME
                        Model name to save
  -single               single point prediction
  -step STEP            step between points
  -epochs EPOCHS        number of epochs (for model training)
  -batch_size BATCH_SIZE
                        number batch during training process (for model
                        training)
  -loss LOSS            which loss to optimize
  -neurons NEURONS      number of neurons in RNN layer
  -optimizer OPTIMIZER  which optimizer
  -mean_models MEAN_MODELS
                        Run the training mean_models times and take the max,
                        min and mean model
  -smooth SMOOTH        smooth the output graph with windows, polynomial
                        order. Use 'windows, int'
  -test_mode, -t        test mode or predict mode
  -verbose VERBOSE, -v VERBOSE
                        set verbosity
  -plot_open            auto open the html graph
  ```

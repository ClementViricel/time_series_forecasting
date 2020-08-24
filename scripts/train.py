import argparse
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

from datetime import datetime
from pathlib import Path
from models import time_series_model
from preprocessing import multivariate_data
from scipy import signal

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser()

    # Input data options
    parser.add_argument("-file", "-f",
                        help="input file with the points (xls, csv or txt)",
                        type=str,
                        required=True)
    parser.add_argument("-points_to_predict", "-pp",
                        help="Number of point to predict",
                        type=int,
                        required=True)
    parser.add_argument("-points_history", "-ph",
                        help="Number of point which are history points",
                        type=int,
                        required=True)
    parser.add_argument("-rows",
                        help="name of the rows (if xls or csv) separate by a comma\n"
                             + "for example \"Free Gb,nb_of_data\" ",
                        type=str,
                        default=[])
    parser.add_argument("-target_row",
                        help="name of the row to predict",
                        type=str,
                        default="")
    parser.add_argument("-limit",
                        help="limit to the number of data",
                        type=int,
                        default=0)

    # Model options
    parser.add_argument("-model_name", "-m",
                        help="Model name to save",
                        type=str,
                        default='')
    parser.add_argument("-single",
                        help="single point prediction",
                        action='store_true',
                        default=False)
    parser.add_argument("-step",
                        help="step between points",
                        type=int,
                        default=1)
    parser.add_argument("-epochs",
                        help="number of epochs (for model training)",
                        type=int,
                        default=50)
    parser.add_argument("-batch_size",
                        help="number batch during training process (for model training)",
                        type=int,
                        default=32)
    parser.add_argument("-loss",
                        help="which loss to optimize",
                        type=str,
                        default="mae")
    parser.add_argument("-neurons",
                        help="number of neurons in RNN layer",
                        type=int,
                        default=64)
    parser.add_argument("-optimizer",
                        help="which optimizer",
                        type=str,
                        default="adam")
    parser.add_argument("-mean_models",
                        help="Run the training mean_models times\n and take the max, min and mean model",
                        type=int,
                        default=10)
    parser.add_argument("-smooth",
                        help="smooth the output graph with windows, polynomial order. Use 'windows, int'",
                        type=str,
                        default="23, 3")

    # test mode
    parser.add_argument("-test_mode", "-t",
                        help="test mode or predict mode",
                        action="store_true",
                        default=False)
    parser.add_argument("-verbose", "-v",
                        help="set verbosity",
                        type=int,
                        default=0)

    parser.add_argument("-plot_open",
                        help="auto open the html graph",
                        action="store_true",
                        default=False)
    STARTTIME = datetime.now()
    print("Start computing")
    args = parser.parse_args()

    path_to_file = Path(args.file)
    if path_to_file.suffix == ".xls" or path_to_file.suffix == ".xlsx":
        df = pd.read_excel(args.file)
    elif path_to_file.suffix == ".csv":
        df = pd.read_csv(args.file)
    else:
        raise TypeError("Unknown type of file")

    days_to_predict = args.points_to_predict
    history_points = args.points_history
    rows = args.rows.split(",")
    target_row = args.target_row
    if len(rows) == 1:
        target_row = rows[0]
    else:
        if not target_row:
            raise ValueError("argument target_row must be specified when there are several rows.")
    epochs = args.epochs
    step = args.step
    single_step = args.single
    batch_size = args.batch_size
    model_name = args.model_name
    limit = args.limit
    # Model parameter
    neurons = args.neurons
    loss = args.loss
    optimizer = args.optimizer
    mean_models = args.mean_models
    auto_open = args.plot_open
    if args.smooth:
        smooth = [int(i) for i in args.smooth.split(',')]
        if smooth[0] > days_to_predict:
            smooth[0] = days_to_predict
    else:
        smooth = []
    test_mode = args.test_mode
    verbose = args.verbose
    if test_mode:
        verbose = 1
    if not model_name:
        model_name = path_to_file.stem
    if not os.path.exists(model_name):
        os.makedirs(model_name)
        os.makedirs(os.path.join(model_name, "checkpoints"))
    if limit:
        dataset = df[rows][:limit].values
        target = df[target_row][:limit].values
    else:
        dataset = df[rows].values
        target = df[target_row].values

    if test_mode:
        train_size = len(dataset) - (history_points + days_to_predict)
        print("Training size is {}".format(train_size))
        print("Test size is {}".format(len(dataset) - train_size))
        test = dataset[train_size: train_size + history_points]
        x_test = np.array(test)
        x_test = np.expand_dims(x_test, axis=-1)
    else:
        train_size = None
        test = dataset[len(dataset) - history_points: len(dataset)]
        x_test = np.array(test)
        x_test = np.expand_dims(x_test, axis=-1)

    x_train, y_train = multivariate_data(
        dataset=dataset,
        target=target,
        start_index=0,
        end_index=None,
        history_size=history_points,
        target_size=days_to_predict,
        step=step,
        single_step=single_step)

    if len(x_train.shape) == 2:
        x_train = np.expand_dims(x_train, axis=-1)

    if verbose:
        print(x_train.shape, y_train.shape)
    if mean_models:
        for iter_ in range(mean_models):
            save_model_name_iter = os.path.join(model_name, "checkpoints/{}".format(iter_))
            point_model, callbacks = time_series_model(days_to_predict=days_to_predict,
                                                       neurons=neurons,
                                                       optimizer=optimizer,
                                                       single_step=single_step,
                                                       model_save_path=save_model_name_iter,
                                                       loss=loss)
            history = point_model.fit(
                x_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                shuffle=False,
                callbacks=callbacks
            )
        point_model, _ = time_series_model(days_to_predict=days_to_predict,
                                           neurons=neurons,
                                           optimizer=optimizer,
                                           single_step=single_step,
                                           model_save_path="",
                                           loss=loss)
        preds = []
        for iter_ in range(mean_models):
            save_model_name_iter = os.path.join(model_name, "checkpoints/{}".format(iter_))
            point_model.load_weights(save_model_name_iter)
            y_pred = point_model.predict(x_test)
            pred = y_pred[0]
            preds.append(pred)
        preds = np.array(preds)
        max_pred = np.max(preds, axis=0)
        min_pred = np.min(preds, axis=0)
        mean_pred = np.mean(preds, axis=0)
        if smooth:
            max_pred = signal.savgol_filter(max_pred,
                                            smooth[0],  # window size used for filtering
                                            smooth[1])  # order of fitted polynomial
            min_pred = signal.savgol_filter(min_pred,
                                            smooth[0],  # window size used for filtering
                                            smooth[1])  # order of fitted polynomial
            mean_pred = signal.savgol_filter(mean_pred,
                                             smooth[0],  # window size used for filtering
                                             smooth[1])  # order of fitted polynomial

        df_results = pd.DataFrame()
        df_results['max_pred'] = list(target) + list(max_pred)
        df_results['min_pred'] = list(target) + list(min_pred)
        df_results['mean_pred'] = list(target) + list(mean_pred)
        if path_to_file.suffix == ".xls" or path_to_file.suffix == ".xlsx":
            df_results.to_excel("{}/results_{}_{}.xlsx".format(model_name,
                                                               history_points, days_to_predict),
                                index=False)

        elif path_to_file.suffix == ".csv":
            df_results.to_csv("{}/results_{}_{}.csv".format(model_name,
                                                            history_points, days_to_predict),
                              index=False)
        # Create traces
        dataset = dataset.squeeze()
        test = test.squeeze()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(dataset))),
                                 y=list(dataset),
                                 mode='lines',
                                 name='True'))
        if test_mode:
            x_range = list(range(len(dataset) - (days_to_predict + history_points),
                                 len(dataset) - days_to_predict))
        else:
            x_range = list(range(len(dataset) - history_points,
                                 len(dataset)))
        fig.add_trace(go.Scatter(x=x_range,
                                 y=test,
                                 mode='lines+markers',
                                 name='Test'))
        if single_step:
            if test_mode:
                x_range = [len(dataset) - 1]
            else:
                x_range = [len(dataset)]
            fig.add_trace(go.Scatter(x=x_range,
                                     y=list(mean_pred),
                                     mode='markers',
                                     name='Predict'))
            fig.add_trace(go.Scatter(x=x_range,
                                     y=list(max_pred),
                                     mode='markers',
                                     name='Upper', showlegend=False))
            fig.add_trace(go.Scatter(x=x_range,
                                     y=list(min_pred),
                                     mode='markers',
                                     name='Lower', showlegend=False))
        else:
            if test_mode:
                x_range = list(range(len(dataset) - days_to_predict - 1, len(dataset)))
            else:
                x_range = list(range(len(dataset) - 1, len(dataset) + days_to_predict))
            fig.add_trace(go.Scatter(x=x_range,
                                     y=[list(test)[-1]] + list(mean_pred),
                                     mode='lines+markers',
                                     name='Predict'))
            fig.add_trace(go.Scatter(x=x_range,
                                     y=[list(test)[-1]] + list(min_pred),
                                     mode='lines',
                                     name='Lower',
                                     line_color='grey', showlegend=False))
            fig.add_trace(go.Scatter(x=x_range,
                                     y=[list(test)[-1]] + list(max_pred),
                                     mode='lines',
                                     name='Upper',
                                     fill='tonexty',
                                     line_color='grey', showlegend=False))
        fig.update_layout(
            title="Neural Network prediction",
            xaxis_title="Days",
            yaxis_title=target_row,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        fig.write_html('{}/graphs_{}_{}.html'.format(model_name, history_points, days_to_predict),
                       auto_open=auto_open)
    else:
        save_model_name = os.path.join(model_name, os.path.join('checkpoints', model_name))
        point_model, callbacks = time_series_model(days_to_predict=days_to_predict,
                                                   neurons=neurons,
                                                   optimizer=optimizer,
                                                   single_step=single_step,
                                                   model_save_path=save_model_name,
                                                   loss=loss)
        history = point_model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=False,
            callbacks=callbacks
        )
        point_model.load_weights(save_model_name)
        y_pred = point_model.predict(x_test)
        pred = y_pred[0]
        if smooth:
            pred = signal.savgol_filter(pred,
                                        smooth[0],  # window size used for filtering
                                        smooth[1])  # order of fitted polynomial
        df_results = pd.DataFrame()
        df_results['pred'] = list(target) + list(pred)

        if path_to_file.suffix == ".xls" or path_to_file.suffix == ".xlsx":
            df_results.to_excel("{}/results_{}_{}.xlsx".format(model_name,
                                                               history_points, days_to_predict),
                                index=False)

        elif path_to_file.suffix == ".csv":
            df_results.to_csv("{}/results_{}_{}.csv".format(model_name,
                                                            history_points, days_to_predict),
                              index=False)

        dataset = dataset.squeeze()
        test = test.squeeze()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(dataset))),
                                 y=dataset,
                                 mode='lines',
                                 name='True'))
        fig.add_trace(go.Scatter(x=list(range(len(dataset) - (days_to_predict + history_points),
                                              len(dataset) - days_to_predict)),
                                 y=test,
                                 mode='lines+markers',
                                 name='Test'))
        if single_step:
            fig.add_trace(go.Scatter(x=[len(dataset) - 2, len(dataset) - 1],
                                     y=[dataset[-1]] + list(pred),
                                     mode='lines+markers',
                                     name='Predict'))
        else:
            fig.add_trace(go.Scatter(x=list(range(len(dataset) - days_to_predict - 1,
                                                  len(dataset))),
                                     y=[list(test)[-1]] + list(pred),
                                     mode='lines+markers',
                                     name='Predict'))
        fig.update_layout(
            title="Neural Network prediction",
            xaxis_title="Days",
            yaxis_title=target_row,
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
        fig.write_html('{}/graph_{}_{}.html'.format(model_name, history_points, days_to_predict),
                       auto_open=auto_open)

    End_time = datetime.now() - STARTTIME
    print("End : {}".format(End_time.total_seconds()))

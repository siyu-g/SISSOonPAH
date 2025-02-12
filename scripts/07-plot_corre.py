# This script read RMSE from test_rmse.txt and train_rmse.txt from folder and make the correlation plots
# Usage: python 07-plot_corre.py 
import numpy as np 
import pandas as pd
import re
import matplotlib.pyplot as plt
import string
import os
import math
import sklearn.metrics

def find_y(pred_file):
    '''Give a path to SISSO predict_Y.out file, output the y_true and y_pred values as np array'''
    # Read the content of SISSO.out
    print("Extracting y_true, y_pred from: %s\n" % pred_file)
    with open(pred_file, 'r') as f:
        s = f.read()

    # Split the content into lines
    lines = s.split('\n')


    # Initialize lists to store y and y_pred
    y_true = []
    y_pred = []

    # Initialize a variable to keep track of the current dimension
    current_dimension = None
    large_dim_index = None

    # Iterate through the lines to find the largest dimension
    for line_index, line in enumerate(lines):
        if line.startswith('Predictions (y,pred,y-pred) by the model of dimension:'):
            # Extract the dimension number
            current_dimension = int(line.split(":")[1])
            large_dim_index = line_index 
    print("Largest Dimension:%s\n" % current_dimension)

    # Iterate through the lines below 'Predictions (y,pred,y-pred) by the model of dimension: LARGESTDIMENSION'
    # to get the y_true and y_pred, skip the last two rows as they are RMSE and MaxAE values, and a blank line
    for line_index, line in enumerate(lines[(large_dim_index+1): -2]):

        # y_true and y_pred are first and second column
        line_split = line.split()
        y_true.append(float(line_split[0]))
        y_pred.append(float(line_split[1]))

    return np.array(y_true), np.array(y_pred)


def plot_corre(y_true, y_pred, outputloc, title="Correlation Plot", x_label=None, xy_min=0, xy_max=1):
    '''Plot correlation'''
    # Start making figure
    fig, ax = plt.subplots()
    # Figure size
    fig.set_size_inches(7.4, 7.4)
    ax.scatter(y_true, y_pred, s = 80,color="purple", alpha=0.8)
    title_1 = ax.set_title(title,fontsize = 20, y=0.97, pad=-14)
    title_1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white')) # Set title background to avoid overlap with plot

    # Calculate the RMSE
    mse = sklearn.metrics.mean_squared_error(y_pred, y_true)
    rmse = math.sqrt(mse)

    # Add RMSE to the plot
    ax.text(0.1, 0.8,"RMSE = %.2f eV" % rmse, fontsize = 20, transform=ax.transAxes)

    # Add x and y labels
    ax.set(ylabel='Model Prediction (eV)')
    ax.set(xlabel=x_label)

    # Set x, y ranges
    ax.axis(xmin = xy_min, xmax = xy_max, ymin= xy_min, ymax = xy_max)
    # Change tick font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    # change font size for x, y labels
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    # Make the plot square
    ax.set_box_aspect(1)

    # Add x = y line
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]   
    l2, = ax.plot(lims, lims,color='red', linestyle = "dashed")

    # Save figure
    plt.savefig(outputloc, bbox_inches='tight')

# Plot the train and test point in one figure
def plot_corre_one(y_true_train, y_pred_train, y_true_test, y_pred_test, outputloc, title="Correlation Plot", x_label=None, xy_min=0, xy_max=1):
    '''Plot correlation'''

    # Calculate the RMSE
    train_mse = sklearn.metrics.mean_squared_error(y_pred_train, y_true_train)
    train_rmse = math.sqrt(train_mse)
    test_mse = sklearn.metrics.mean_squared_error(y_pred_test, y_true_test)
    test_rmse = math.sqrt(test_mse)
    
    # Start making figure
    fig, ax = plt.subplots()
    # Figure size
    fig.set_size_inches(7.4, 7.4)
    ax.scatter(y_true_train, y_pred_train, s = 80,color="purple", alpha=0.8, label="Train RMSE: %.2f" % train_rmse)
    ax.scatter(y_true_test, y_pred_test, s = 80, facecolors='none', edgecolors='green', alpha=0.8, linewidth=2, label="Test RMSE: %.2f" % test_rmse)
       
    title_1 = ax.set_title(title,fontsize = 20, y=0.97, pad=-14)
    title_1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white')) # Set title background to avoid overlap with plot



    # Add RMSE to the plot
    # ax.text(0.1, 0.8,"RMSE = %.2f eV" % rmse, fontsize = 20, transform=ax.transAxes)

    # Add x and y labels
    ax.set(ylabel='Model Prediction (eV)')
    ax.set(xlabel=x_label)

    # Add legend
    plt.legend(loc="lower right", fontsize=20)

    # Set x, y ranges
    ax.axis(xmin = xy_min, xmax = xy_max, ymin= xy_min, ymax = xy_max)
    # Change tick font size
    ax.tick_params(axis='both', which='major', labelsize=18)
    # change font size for x, y labels
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    # Make the plot square
    ax.set_box_aspect(1)

    # Add x = y line
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]   
    l2, = ax.plot(lims, lims,color='red', linestyle = "dashed")

    # Save figure
    plt.savefig(outputloc, bbox_inches='tight')

def plot_corre_multi(data_frame, outputloc, print_type="train", x_label='GW+BSE $\Delta E_{ST}$ (eV)', target_name="est", xy_min=0, xy_max=1):
    '''Plot correlation with multiple panels, here assume 6 panels,
        The function will create train.png and test.png under outputloc'''
    # Start making figure
    fig, axs = plt.subplots(2, 3)
    # Flatten the 2D array of axes to a 1D array
    axs_flat = axs.flatten()

    # Adjust space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # Figure size, Assume 2 rows 3 columns, so height = 2 *7.4 = 14.8, wide = 3 * 7.4 = 22.2
    fig.set_size_inches(22.2, 14.8) 
    
    # Loop through rows to fetch the output files in folders with lowest Train RMSEs
    i = 0
    for index, row in data_frame.iterrows():  
        row_name = index  
        min_idx = int(row["min_idx"])  # Index of best cross validation
        # Prediction on Training data is stored in "train_predict_Y.out" file
        pred_file_train = "%s/cross_validate%s/train_predict_Y.out" % (row_name, int(row["min_idx"]))
        # Prediction on Test data is stored in "predict_Y.out" file
        pred_file_test = "%s/cross_validate%s/predict_Y.out" % (row_name, int(row["min_idx"]))

        # Get the values of y_true and y_pred for train/test data
        if print_type == "train":
            y_true, y_pred = find_y(pred_file=pred_file_train)
            title = row["anotations"]+" Train"
            # Set the name of the saved figure
            outputpath = outputloc + "train_correlation_%s.png" % target_name
        else:
            y_true, y_pred = find_y(pred_file=pred_file_test) 
            title = row["anotations"]+" Test"    
            # Set the name of the saved figure
            outputpath = outputloc + "test_correlation_%s.png" % target_name  
        
        # Starting to make subplot
        ax = axs_flat[i]
        ax.scatter(y_true, y_pred, s = 80,color="purple", alpha=0.8)
        
        # Set title
        title_1 = ax.set_title(title,fontsize = 20, y=0.97, pad=-14)
        title_1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white')) # Set title background to avoid overlap with plot


        # # Calculate the RMSE
        mse = sklearn.metrics.mean_squared_error(y_pred, y_true)
        rmse = math.sqrt(mse)

        # # Add RMSE to the plot

        ax.text(0.1, 0.8,"RMSE = %.2f eV" % rmse, fontsize = 20, transform=ax.transAxes)
        # ax.text(0.4, 1.2,"RMSE = %.2f eV" % rmse, fontsize = 20)

        # Add x and y labels
        ax.set(ylabel='Model Prediction (eV)')
        ax.set(xlabel=x_label)

        # Set x, y ranges
        ax.axis(xmin = xy_min, xmax = xy_max, ymin= xy_min, ymax = xy_max)
        # Change tick font size
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # change font size for x, y labels
        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
        # Make the plot square
        ax.set_box_aspect(1)

        # Add x = y line
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]   
        l2, = ax.plot(lims, lims,color='red', linestyle = "dashed")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in fig.get_axes():
            ax.label_outer()
        
        # Add annotate to subplots
        for  n, ax in enumerate(fig.get_axes()):
            ax.text(0.03, 0.94, string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20)
        i+=1


    # Save figure
    plt.savefig(outputpath, bbox_inches='tight')


def plot_corre_all(data_frame, outputloc, x_label='GW+BSE $\Delta E_{ST}$ (eV)', target_name="est", xy_min=0, xy_max=1):
    '''Plot correlation with multiple panels, here assume 12 panels,
        The function will create train/test correlation under outputloc'''
    # Start making figure
    fig, axs = plt.subplots(4, 3)
    # Flatten the 2D array of axes to a 1D array
    axs_flat = axs.flatten()

    # Adjust space between subplots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    # Figure size, Assume 4 rows 3 columns, so height = 4 *7.4 = 29.6, wide = 3 * 7.4 = 22.2
    fig.set_size_inches(22.2, 29.6) 
    
    # Loop through rows to fetch the output files in folders with lowest Train RMSEs
    i = 0
    for index, row in data_frame.iterrows():  
        row_name = index  
        min_idx = int(row["min_idx"])  # Index of best cross validation
        # Prediction on Training data is stored in "train_predict_Y.out" file
        pred_file_train = "%s/cross_validate%s/train_predict_Y.out" % (row_name, int(row["min_idx"]))
        # Prediction on Test data is stored in "predict_Y.out" file
        pred_file_test = "%s/cross_validate%s/predict_Y.out" % (row_name, int(row["min_idx"]))

        # Get the values of y_true and y_pred for train/test data
        y_true_train, y_pred_train = find_y(pred_file=pred_file_train)
        y_true_test, y_pred_test = find_y(pred_file=pred_file_test)

        # Set the title name based on anotation column
        title = row["anotations"]

        # Set output path
        outputpath = outputloc + "correlation_all12_%s.png" % target_name

        # Calculate the RMSE
        train_mse = sklearn.metrics.mean_squared_error(y_pred_train, y_true_train)
        train_rmse = math.sqrt(train_mse)
        test_mse = sklearn.metrics.mean_squared_error(y_pred_test, y_true_test)
        test_rmse = math.sqrt(test_mse)
    

        # Starting to make subplot
        ax = axs_flat[i]
        ax.scatter(y_true_train, y_pred_train, s = 80,color="purple", alpha=0.8, label="Train RMSE: %.2f" % train_rmse)
        ax.scatter(y_true_test, y_pred_test, s = 80, facecolors='none', edgecolors='green', alpha=0.8, linewidth=2, label="Test RMSE: %.2f" % test_rmse)
        
        # Add legend
        ax.legend(loc="lower right", fontsize=20)
        # Set title
        title_1 = ax.set_title(title,fontsize = 20, y=0.97, pad=-14)
        title_1.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white')) # Set title background to avoid overlap with plot


        # Add x and y labels
        ax.set(ylabel='Model Prediction (eV)')
        ax.set(xlabel=x_label)

        # Set x, y ranges
        ax.axis(xmin = xy_min, xmax = xy_max, ymin= xy_min, ymax = xy_max)
        # Change tick font size
        ax.tick_params(axis='both', which='major', labelsize=18)
        
        # change font size for x, y labels
        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
        # Make the plot square
        ax.set_box_aspect(1)

        # Add x = y line
        lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]   
        l2, = ax.plot(lims, lims,color='red', linestyle = "dashed")

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in fig.get_axes():
            ax.label_outer()
        
        # Add annotate to subplots
        for  n, ax in enumerate(fig.get_axes()):
            ax.text(0.03, 0.94, string.ascii_lowercase[n]+')', transform=ax.transAxes, 
                size=20)
            

        
        i+=1


    # Save figure
    plt.savefig(outputpath, bbox_inches='tight')


def find_outlier(y_true_train, y_pred_train, y_true_test, y_pred_test, 
                     name_train, name_test, numOutlierTrain = 5, numOutlierTest = 5):
    # Initialize dataframe, store the value of true and pred
    df = pd.DataFrame(columns=["name", "y", "y_pred", "ifTrain"])

    # Append rows to dataframe, Train first, Test after
    train_len = len(y_true_train)
    test_len = len(y_true_test)
    # Append train rows
    for i in range(0, train_len):
        # Name, y_true, y_pred, ifTrain= Train
        df.loc[i] = [name_train[i], y_true_train[i], y_pred_train[i], True]
    # Append test rows
    for i in range(0, test_len):
        # Name, y_true, y_pred, ifTrain= False
        df.loc[len(df)] = [name_test[i], y_true_test[i], y_pred_test[i], False]
    # Calculate the difference between true and pred values
    df["|y - y_pred|"] = abs(df["y"] - df["y_pred"])

    # Sort the dataframe to get the first few outliers, number of outlier is controled by numOutlierTrain/numOutlierTest
    top_train = df[df["ifTrain"] ==  True].sort_values(by="|y - y_pred|", ascending=False).iloc[:numOutlierTrain]
    top_test = df[df["ifTrain"] !=  True].sort_values(by="|y - y_pred|", ascending=False).iloc[:numOutlierTest]

    # Return the subDataframe 
    return top_train, top_test

if __name__ == "__main__":
    # Load lowest Train/Test RMSEs from text files
    train_rmse_min = pd.read_csv("train_rmse.txt", index_col=0, sep=" ")
    test_rmse_min = pd.read_csv("test_rmse.txt", index_col=0,  sep=" ")
    # Name mapping of the training data
    name_train = pd.read_csv("data_splits/predict-train.dat", sep=" ")["materials"].to_numpy()
    # Name mapping of the test data
    name_test = pd.read_csv("data_splits/predict-test.dat", sep=" ")["materials"].to_numpy()

    # Name the models by labels
    anotations = ["$M_{1,1}$", "$M_{2,1}$", "$M_{3,1}$", "$M_{4,1}$", "$M_{1,2}$", "$M_{2,2}$", "$M_{3,2}$", "$M_{4,2}$","$M_{1,3}$", "$M_{2,3}$", "$M_{3,3}$", "$M_{4,3}$"]

    ################################################################################################
    # Settings to adjust based on different target of the model
    ################################################################################################
    # List of best models to plot together based on the Pareto Front (Assume to be 6)
    # model_list = ["$M_{1,2}$", "$M_{2,1}$", "$M_{4,2}$", "$M_{2,3}$", "$M_{3,3}$", "$M_{4,3}$"]
    model_list = ["$M_{3,3}$", "$M_{4,2}$"]

    # Set x label of the plot
    x_label='GW+BSE $E_{S}$ (eV)'

    # Set target name of the plot, this influence the name of the saved plot
    target_name = 'es'

    # Set xy range of the correlation plots
    xy_min=0.5 
    xy_max=6
    ################################################################################################

    # Using DataFrame.insert() to add a column anotations
    train_rmse_min.insert(2, "anotations", anotations, True)
    
    # Filter the DataFrame based on model_list to plot toghether
    filtered_df = train_rmse_min[train_rmse_min['anotations'].isin(model_list)]

    # Reorder according to model_list
    filtered_df['order'] = pd.Categorical(filtered_df['anotations'], categories=model_list, ordered=True)
    filtered_df = filtered_df.sort_values('order').drop(columns='order')

    # Make a folder to store the correlation plots
    try:
        os.mkdir("correlation_plots/")
    except FileExistsError:
        pass
    
    
    # Loop through rows to fetch the output files in folders with lowest Train RMSEs
    i = 0
    for index, row in filtered_df.iterrows():  
        row_name = index  
        min_idx = int(row["min_idx"])  # Index of best cross validation
        # Prediction on Training data is stored in "train_predict_Y.out" file
        pred_file_train = "%s/cross_validate%s/train_predict_Y.out" % (row_name, int(row["min_idx"]))
        # Prediction on Test data is stored in "predict_Y.out" file
        pred_file_test = "%s/cross_validate%s/predict_Y.out" % (row_name, int(row["min_idx"]))


        # Get the values of y_true and y_pred for train/test data
        y_true_train, y_pred_train = find_y(pred_file=pred_file_train)
        y_true_test, y_pred_test = find_y(pred_file=pred_file_test)

        # # # Make single correlation plots
        # plot_corre(y_pred=y_pred_train, y_true=y_true_train, title=anotations[i]+" Train", outputloc="correlation_plots/%s_train.png" % anotations[i], x_label=x_label, xy_min=xy_min, xy_max=xy_max)
        # plot_corre(y_pred=y_pred_test, y_true=y_true_test, title=anotations[i]+" Test", outputloc="correlation_plots/%s_test.png" % anotations[i], x_label=x_label, xy_min=xy_min, xy_max=xy_max)

        # # Plot train/test together
        plot_corre_one(y_true_train=y_true_train, y_pred_train=y_pred_train, y_true_test=y_true_test, y_pred_test=y_pred_test, 
                       outputloc="correlation_plots/%s.png" % model_list[i], title=row["anotations"], x_label=x_label, xy_min=xy_min, xy_max=xy_max)
        # Find outlier from the correlation plot
        outliers_train, outliers_test = find_outlier(y_true_train=y_true_train, y_pred_train=y_pred_train, y_true_test=y_true_test, y_pred_test=y_pred_test, 
                     name_train=name_train, name_test=name_test)
        print("Outliers for model %s: \n" % row["anotations"], outliers_train, "\n", outliers_test)
        i += 1
        
    # # # Plot the correlation plots of the 6 best models toghether
    # plot_corre_multi(filtered_df, outputloc="correlation_plots/", print_type='train', x_label=x_label, target_name=target_name, xy_min=xy_min, xy_max=xy_max)
    # plot_corre_multi(filtered_df, outputloc="correlation_plots/", print_type='test', x_label=x_label, target_name=target_name, xy_min=xy_min, xy_max=xy_max)
    
    # # Plot train/test together
    # plot_corre_all(train_rmse_min, outputloc="correlation_plots/", x_label=x_label, target_name=target_name, xy_min=xy_min, xy_max=xy_max)



    



# This script read RMSE from test_rmse.txt and train_rmse.txt from each folder, and give a summary of all the different trials
# It outputs the minimum value of all the cross_validations and gives which train/valid split gives the minimum RMSE
# Usage: python 06-analyze_data.py > model_analyze.txt
# Output: train_rmse.txt / test_rmse.txt / model_analyze.txt


import numpy as np 
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
#from adjustText import adjust_text
import string


def read_data(file, col_name=None):
    '''Read last column as the largest dimensional data'''
    df = pd.read_csv(file, header=None, sep=" ")
    # Extract the first & last column into a new DataFrame
    first_column_df = df.iloc[:, :1]
    first_column_df.columns = ["trials"]
    last_column_df = df.iloc[:, -1:]
    last_column_df.columns = [col_name]

    # Stack the two columns together
    stacked_df = pd.concat([first_column_df, last_column_df], axis=1)
    return stacked_df

def find_min(dataframe, rmse_type="train"):
    '''Given a dataframe, find the min RMSE of each column/setting'''
    min_rmse = pd.DataFrame(dataframe.iloc[:,1:].min(), columns=["%s_rmse_min" % rmse_type])
    min_idx = dataframe.iloc[:,1:].idxmin()
    min_rmse["min_idx"] = min_idx
    return min_rmse

def find_model(outputpath):
    '''Give a path to SISSO training output, grep the model from the file
        Input: outputpath -- path to SISSO.out file
        Outout: descriptors_array -- Numpy array with highest dimensional descriptors
                coefficient_array -- Numpy array with coefficients of highest dimensional descriptors
    '''

    # Read the content of SISSO.out
    print("Extracting model from: %s\n" % outputpath)
    with open(outputpath, 'r') as f:
        s = f.read()

    # Split the content into lines
    lines = s.split('\n')

    # Initialize lists to store descriptors and coefficients
    descriptors = []
    coefficients = []

    # Initialize a variable to keep track of the current dimension
    current_dimension = None

    # Iterate through the lines to find the largest dimension
    for line_index, line in enumerate(lines):
        if line.startswith('Dimension:'):
            # Extract the dimension number
            current_dimension = int(line.split()[1])
    print("Largest Dimension:%s\n" % current_dimension)

    # Extract the descriptors and coefficients:
    for line_index, line in enumerate(lines):
        if line.startswith('Final model/descriptor !'):
            descriptors_idx = line_index + 3
            # Get descriptors from the line nD Descriptors:           
            while(len(lines[descriptors_idx])):
                descriptors.append(lines[descriptors_idx].split()[2])
                descriptors_idx+=1
            coefficients_idx = descriptors_idx + 2
            coefficients = (lines[coefficients_idx].split()[1:])
            # Need to add the constant term from the following line
            coefficients.append(lines[coefficients_idx +1].split()[1])


    # Convert lists to NumPy arrays
    descriptors_array = np.array(descriptors)
    coefficients_array = np.array(coefficients)

    # Print the arrays
    print(f"{current_dimension}D Descriptors:")
    print(descriptors_array)
    print("\nCoefficients (Last term constant):")
    print(coefficients_array)
    return descriptors_array, coefficients_array

def find_cost(costfile, descriptors_array):
    '''Find cost to compute the descriptors in the model
        Input: costfile -- Path to descriptor & cost mapping file
               descriptors_array -- Output of find_model(), array that contains descriptor strings
        Output: total cost (int)
    '''
    # Define a regular expression pattern to match operators and parenthesis
    pattern = r'\+|\-|\*|\/|exp|log|abs|\^-1|\^2|\^3|sqrt|cbrt|\(|\)'

    # Initialize a set to collect unique features
    unique_features = set()

    # Loop through each descriptor and split it based on the pattern
    for descriptor in descriptors_array:
        # Split the descriptor using the pattern and filter out empty strings
        features = filter(None, re.split(pattern, descriptor))
        # Remove numeric values or scientific notation from the split strings
        features = [re.sub(r'[-+]?\d*\.\d+E[-+]?\d+|\d+', '', feature) for feature in features]
        # Add unique features to the set
        unique_features.update(features)

    # Convert the set to a sorted list
    unique_features_list = sorted(unique_features)

    # Print the unique features
    print(unique_features_list)

    # Open cost file and get a mapping between feature and cost
    with open(costfile) as f:
        s = f.read()

    # Split the content into lines
    lines = s.split('\n')

    # Create the mapping by lines in costfile:
    cost_map = {}
    for line in lines:
        if len(line):
            (feature, cost) = line.split()
            cost_map[feature] = int(cost)
    # print(cost_map)

    # Special cases : Here CBdispC, VB_dispC, GapC are generated by one FHI-aims run, we want to remove duplicate when considering the cost
    # You can DIY the Special cases based on your runs

        # Check if any two or more of 'CBdispC', 'VBdispC', and 'GapC' are present
    if ('CBdispC' in unique_features_list and 'VBdispC' in unique_features_list) or \
    ('CBdispC' in unique_features_list and 'GapC' in unique_features_list) or \
    ('VBdispC' in unique_features_list and 'GapC' in unique_features_list):
        # Keep only one of them
        unique_features_list = [feature for feature in unique_features_list if feature not in ('CBdispC', 'VBdispC', 'GapC')]
        unique_features_list.append('CBdispC')  # Add one of them back (arbitrary choice)
    
    # Special case 2: Here ETC and DFC_ST are generated by one FHI-aims run, we want to remove duplicate when considering the cost
    if ('ETC' in unique_features_list and 'DFC_ST' in unique_features_list) or \
        ('CBdispC' in unique_features_list and 'DFC_ST' in unique_features_list) or \
        ('GapC' in unique_features_list and 'DFC_ST' in unique_features_list) or \
        ('VBdispC' in unique_features_list and 'DFC_ST' in unique_features_list):
        unique_features_list = [feature for feature in unique_features_list if feature not in ('ETC','CBdispC', 'VBdispC', 'GapC')]


     # Special case 3: Here ETS and GapS are generated by one FHI-aims run, we want to remove duplicate when considering the cost
    if ('GapS' in unique_features_list and 'ETS' in unique_features_list):
        unique_features_list = [feature for feature in unique_features_list if feature not in ('GapS')]  
    print("After deleting duplicate:", unique_features_list)

    #Calculate the cost for certain features:
    cost_tot = 0

    for feat in unique_features_list:
        cost_tot += cost_map[feat]
    return cost_tot


if __name__ == "__main__":
    # Find settings based on folder names
    settings = np.sort(glob.glob("fcomp*dim*"))
    costfile_path = "./cost.txt"
    plot_pareto = True # If run locally, setting this to True can make plots of accuracy-cost scatters

    # Merge Train/Test RMSE from all folders
    for i, setting in enumerate(settings):
        test_file = setting+'/test_rmse.txt'
        train_file = setting+'/train_rmse.txt'
        if i == 0:
            df_train = read_data(train_file, col_name=setting)
            df_test = read_data(test_file, col_name=setting)
        else:
            df_train = pd.merge(df_train, read_data(train_file, col_name=setting), on = "trials")
            df_test = pd.merge(df_test, read_data(test_file, col_name=setting), on = "trials")

    print("Train RMSE for each trial:\n", df_train)
    print("Test RMSE for each trial:\n", df_test)

    # Select the Model based on lowest Training RMSE
    train_rmse_min = find_min(df_train, rmse_type="train")
    test_rmse_min = find_min(df_test, rmse_type="test")

    #Find the corresponding Test RMSE for selected models

    for index, row in train_rmse_min.iterrows():
        row_name = index  
        min_idx = int(row["min_idx"])
        test_rmse_min["min_idx"][row_name] = min_idx
        test_rmse_min["test_rmse_min"][row_name] = df_test[row_name][min_idx]

    print(train_rmse_min)
    print(test_rmse_min)
    train_rmse_min.to_csv("train_rmse.txt", sep=" ")
    test_rmse_min.to_csv("test_rmse.txt", sep=" ")
    
    # Find models with lowest training RMSEs in each settings, compare their costs
    print("###########################################################")
    print("\nExtracting models by index with lowest training RMSEs...\n")
    print("###########################################################")
    # train_rmse = []
    train_costs = []
    for index, row in train_rmse_min.iterrows():
        row_name = index   
        descriptors_array, coefficients_array = find_model("%s/cross_validate%s/SISSO.out" % (row_name, int(row["min_idx"])))
        model_cost = find_cost(costfile=costfile_path, descriptors_array=descriptors_array)
        print("\nTrain RMSE: %.3f eV" % row["train_rmse_min"])
        print("Model Cost: %s\n" % model_cost)
        # train_rmse.append(row["train_rmse_min"])
        train_costs.append(model_cost)
    train_rmse = train_rmse_min["train_rmse_min"].to_numpy()
    test_rmse = test_rmse_min["test_rmse_min"].to_numpy()


    # Start plotting Training/Testing Pareto Plot:
    if plot_pareto == True:
        # Make scatter plots in two panels
        fig, ax = plt.subplots()
        # Spacing between subfigures
        # plt.subplots_adjust(wspace=0.05, hspace=0.15)

        # Figure size
        fig.set_size_inches(7.4,7.4)

        # Panel a, Pareto plot for Training RMSE v.s Cost
        ax.scatter(train_costs, train_rmse, s = 80, color="purple", alpha=0.8, label="Train")

        # Add annotations to the models
        anotations = ["$M_{1,1}$", "$M_{2,1}$", "$M_{3,1}$", "$M_{4,1}$", "$M_{1,2}$", "$M_{2,2}$", "$M_{3,2}$", "$M_{4,2}$","$M_{1,3}$", "$M_{2,3}$", "$M_{3,3}$", "$M_{4,3}$"]
         # anotations = ["M1,1", "M1,2", "M1,3", "M1,4", "M3,1", "M3,2", "M3,3", "M3,4","M7,1", "M7,2", "M7,3", "M7,4"]
        for i, txt in enumerate(anotations):
            if txt == "$M_{1,2}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i], train_rmse[i]+0.005),fontsize = 18)
            elif txt == "$M_{2,1}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i], train_rmse[i]),fontsize = 18)
            elif txt == "$M_{3,2}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i]-10, train_rmse[i]+0.005),fontsize = 18)
            elif txt == "$M_{3,1}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i]-10, train_rmse[i]-0.006),fontsize = 18)
            elif txt == "$M_{4,1}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i], train_rmse[i]-0.012),fontsize = 18)
            elif txt == "$M_{4,2}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i]-5, train_rmse[i]+0.003),fontsize = 18)
            elif txt == "$M_{4,3}$":
                ax.annotate(txt, (train_costs[i], train_rmse[i]), (train_costs[i]-10, train_rmse[i]+0.005),fontsize = 18)
            else:
                ax.annotate(txt, (train_costs[i], train_rmse[i]),(train_costs[i], train_rmse[i]), fontsize = 18)
        # Adjust the text to prevent overlapping
        # texts = [ax1.text(train_costs[i], train_rmse[i], anotations[i], ha='center', va='center') for i in range(len(train_costs))]
        # adjust_text(texts)
        # X and Y labels
        # ax1.set_ylim([0.07, 0.18])
        # ax.set_ylabel("Validation RMSE (eV)", fontsize =16)
        # ax1.set_xlabel("Cost", fontsize=16)

    # Panel b, Pareto plot for Test RMSE v.s Cost
        ax.scatter(train_costs, test_rmse, s = 80, facecolors='none', edgecolors='green', alpha=0.8, linewidth=2,label="Test")

        # Add annotations to the models
        # anotations = ["M1,1", "M1,2", "M1,3", "M1,4", "M3,1", "M3,2", "M3,3", "M3,4","M7,1", "M7,2", "M7,3", "M7,4"]

        for i, txt in enumerate(anotations):
            if txt == "$M_{4,2}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i]-15, test_rmse[i]),fontsize = 18)
            elif txt == "$M_{2,2}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i], test_rmse[i]+0.005),fontsize = 18)
            elif txt == "$M_{3,2}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i]+2, test_rmse[i]-0.005),fontsize = 18)
            elif txt == "$M_{1,2}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i], test_rmse[i]-0.005),fontsize = 18)
            elif txt == "$M_{2,3}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i], test_rmse[i]-0.005),fontsize = 18)
            elif txt == "$M_{3,3}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i], test_rmse[i]+0.005),fontsize = 18)
            elif txt == "$M_{4,3}$":
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i]-10, test_rmse[i]+0.005),fontsize = 18)
            else:
                ax.annotate(txt, (train_costs[i], test_rmse[i]), (train_costs[i], test_rmse[i]),fontsize = 18)
        # Adjust the text to prevent overlapping
        # X and Y labels
        # ax.set_xlim([0, 5])
        # ax.set_ylim([0.07, 0.18])
        ax.set_ylabel("RMSE (eV)", fontsize =16)
        ax.set_xlabel("Relative Cost", fontsize=16)

        # Title and legends
        # title_1 = ax1.set_title('Validation RMSE (eV)',fontsize = 20, y=0.97, pad=-14)
        # title_2 = ax.set_title('Test RMSE (eV)',fontsize = 20, y=0.97, pad=-14)
        # for title in [title_1, title_2]:
        #     title.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='white')) # Set title background to avoid overlap with plot
        
        # Axes and ticks
        # for ax in ax:
            # Set x, y ranges
        ax.axis(ymin= 0.09, ymax = 0.32)
            
            # Change tick font size
        ax.tick_params(axis='both', which='major', labelsize=18)
            
            # Set y_label to blank
        # ax.set_ylabel(None)
            # change font size for x, y labels
        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
        ax.set_box_aspect(1)  # Make the plot square

        ax.legend(fontsize = 20, loc = "upper right")
        # Hide x labels and tick labels for top plots and y ticks for right plots.
        # for ax in fig.get_axes():
        #     ax.label_outer()

        # # Add annotate to subplots
        # for n, ax in enumerate([ax1, ax2]):
        #     ax.text(0.03, 0.94, string.ascii_lowercase[n]+')', transform=ax.transAxes, 
        #             size=20)
        plt.subplots_adjust(left=0.05, right=1.05)
            
        plt.savefig("sisso_pareto_es.png")
        plt.clf() 
        






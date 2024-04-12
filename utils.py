# Data manipulation libraries
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

# ML packages
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score

# System and config packages
import os
import yaml
import urllib

# Visualization and charts
import matplotlib.pyplot as plt
import seaborn as sns

# Warning ignore package
import warnings
warnings.filterwarnings('ignore')


def load_data_from_s3(bucket_name, aws_region, file_path):
    """
    Load data from an S3 bucket.

    Args:
        bucket_name (str): The name of the S3 bucket.
        aws_region (str): The AWS region of the S3 bucket.
        aws_access_key (str): AWS access key ID.
        aws_secret_key (str): AWS secret access key.
        file_path (str): The path to the S3 file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    s3 = boto3.resource("s3", region_name=aws_region)
    s3_object = s3.Object(bucket_name, file_path)
    csv_content = s3_object.get()['Body'].read()
    df_input = pd.read_csv(BytesIO(csv_content),header=None,usecols=[1, 3],names=["label", "sentence"])
    return df_input


def push_df_to_s3(data, bucket_name, aws_region, aws_access_key, aws_secret_key, file_path):
    """
    Push data to an S3 bucket.

    Args:
        data: The data to push (e.g., pandas DataFrame, bytes data)
        bucket_name (str): The name of the S3 bucket.
        aws_region (str): The AWS region of the S3 bucket.
        aws_access_key (str): AWS access key ID.
        aws_secret_key (str): AWS secret access key.
        file_path (str): The path to the S3 file.

    Returns:
        None
    """
    s3 = boto3.resource("s3", aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name=aws_region)
    if isinstance(data, pd.DataFrame):
        csv_data = data.to_csv(index=False).encode('utf-8')
    else:
        csv_data = data
    s3_object = s3.Object(bucket_name, file_path)
    s3_object.put(Body=csv_data)
    print(f"Data successfully pushed to s3://{bucket_name}/{file_path}")

    
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches


def plot_dist_char(df, feature, title):
    # Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(20, 15))
    # Creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 color='#e78c3c')
    ax1.set(ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))

    plt.title(f'{title}', fontsize=18)
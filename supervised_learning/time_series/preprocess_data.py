#!/usr/bin/env python3
"""Preprocess the Bitcoin dataset before training the model."""

import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf


def preprocess():
    """Load, clean, normalize, and split the Bitcoin dataset."""

    """
    dataset_file = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'

    df = pd.read_csv(dataset_file)

    # Convert Unix timestamps to datetime objects.
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Replace missing volume information with zeros.
    df['Volume_(BTC)'].fillna(value=0, inplace=True)
    df['Volume_(Currency)'].fillna(value=0, inplace=True)
    df['Weighted_Price'].fillna(value=0, inplace=True)

    # Forward-fill missing market prices.
    df['Open'].fillna(method='ffill', inplace=True)
    df['High'].fillna(method='ffill', inplace=True)
    df['Low'].fillna(method='ffill', inplace=True)
    df['Close'].fillna(method='ffill', inplace=True)

    # Keep only records from 2017 onward.
    df = df[df['Timestamp'].dt.year >= 2017]
    df.reset_index(drop=True, inplace=True)

    # Reduce the dataset to hourly intervals.
    df = df[0::60]

    timestamps = pd.to_datetime(df.pop('Timestamp'))

    selected_features = [
        'Open',
        'High',
        'Low',
        'Close',
        'Volume_(BTC)',
        'Volume_(Currency)',
        'Weighted_Price'
    ]

    feature_data = df[selected_features]
    feature_data.index = timestamps

    # Plot all selected features.
    _ = feature_data.plot(subplots=True)

    feature_indices = {
        column: index for index, column in enumerate(df.columns)
    }

    total_samples = len(df)

    train_data = df[:int(total_samples * 0.7)]
    validation_data = df[int(total_samples * 0.7):int(total_samples * 0.9)]
    test_data = df[int(total_samples * 0.9):]

    number_of_features = df.shape[1]

    training_features = train_data[selected_features]
    training_features.index = timestamps[:int(total_samples * 0.7)]

    _ = training_features.plot(subplots=True)

    # Normalize the datasets using statistics from the training set.
    train_mean = train_data.mean(axis=0)
    train_std = train_data.std(axis=0)

    train_data = (train_data - train_mean) / train_std
    validation_data = (validation_data - train_mean) / train_std
    test_data = (test_data - train_mean) / train_std

    return train_data, validation_data, test_data
    """

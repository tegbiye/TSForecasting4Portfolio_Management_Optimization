import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch
from src.forecast import (
    create_sequences,
    TimeSeriesDataset,
    LSTMModel,
    calculate_metrics
)

# Sample data for testing
@pytest.fixture
def sample_data():
    return np.random.rand(100, 1)

@pytest.fixture
def sample_sequences():
    data = np.arange(100).reshape(-1, 1)
    X, y = create_sequences(data, seq_length=10)
    return X, y

# Test create_sequences function
def test_create_sequences(sample_data):
    seq_length = 10
    X, y = create_sequences(sample_data, seq_length)

    assert X.shape[0] == len(sample_data) - seq_length
    assert X.shape[1] == seq_length
    assert y.shape[0] == len(sample_data) - seq_length
    assert np.array_equal(X[0], sample_data[0:seq_length])
    assert y[0] == sample_data[seq_length]

# Test TimeSeriesDataset class
def test_timeseries_dataset(sample_sequences):
    X, y = sample_sequences
    dataset = TimeSeriesDataset(X, y)

    assert len(dataset) == len(X)
    assert np.array_equal(dataset[0][0], X[0])
    assert dataset[0][1] == y[0]

    # Test without labels
    dataset_no_y = TimeSeriesDataset(X)
    assert np.array_equal(dataset_no_y[0], X[0])

# Test LSTMModel class
@pytest.mark.parametrize("input_size, hidden_size, num_layers, output_size", [
    (1, 50, 1, 1),
    (1, 100, 2, 1),
])
def test_lstm_model(input_size, hidden_size, num_layers, output_size):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    # Create a dummy input tensor
    batch_size = 5
    seq_length = 60
    dummy_input = torch.randn(batch_size, seq_length, input_size)
    
    output = model(dummy_input)

    assert output.shape == torch.Size([batch_size, output_size])
    assert isinstance(model.lstm, torch.nn.LSTM)
    assert isinstance(model.fc, torch.nn.Linear)

# Test calculate_metrics function
def test_calculate_metrics():
    true_values = np.array([10, 20, 30, 40, 50])
    pred_values = np.array([11, 21, 28, 42, 48])

    mae, rmse, mape = calculate_metrics(true_values, pred_values)

    assert mae == pytest.approx(1.6)
    assert rmse == pytest.approx(1.6733200530681511)
    assert mape == pytest.approx(6.133333333333333)  # Corrected MAPE value
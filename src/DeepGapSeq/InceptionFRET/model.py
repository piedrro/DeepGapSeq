import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionModule(nn.Module):

    def __init__(self, in_channels, n_filters=40, bottleneck_channels=32, kernel_sizes=[9, 19, 39], activation_function=F.relu):
        super(InceptionModule, self).__init__()
        # Bottleneck layer is now a 1D Convolution over the last dimension
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1)

        self.conv_branches = nn.ModuleList()
        for kernel_size in kernel_sizes:
            padding = kernel_size // 2 if kernel_size % 2 == 1 else (kernel_size - 1) // 2
            self.conv_branches.append(nn.Conv1d(bottleneck_channels, n_filters, kernel_size=kernel_size, padding=padding))

        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool_conv = nn.Conv1d(bottleneck_channels, n_filters, kernel_size=1)

        self.batchnorm = nn.BatchNorm1d(n_filters * (len(kernel_sizes) + 1))
        self.activation_function = activation_function

    def forward(self, x):
        # Transpose to bring channels to the second dimension
        x = x.transpose(1, 2)

        x = self.bottleneck(x)

        conv_branches = [branch(x) for branch in self.conv_branches]

        # MaxPool branch
        maxpool_branch = self.maxpool(x)
        conv_1x1_branch = self.maxpool_conv(maxpool_branch)
        conv_branches.append(conv_1x1_branch)

        x = torch.cat(conv_branches, dim=1)
        x = self.batchnorm(x)
        x = self.activation_function(x)

        # Transpose back
        x = x.transpose(1, 2)

        return x

class InceptionFRET(nn.Module):
    def __init__(self, in_channels, n_classes = 2, depth=6, lstm_hidden_sizes = [128,64], dropout_rate=0.5, use_skip_connections=True, n_filters=40, bottleneck_channels=32, kernel_sizes=[9, 19, 39], activation_function=F.relu):

        super(InceptionFRET, self).__init__()
        self.use_skip_connections = use_skip_connections

        self.inception_modules = nn.ModuleList()
        self.skip_connections = nn.ModuleList() if use_skip_connections else None

        for i in range(depth):
            if use_skip_connections and i % 3 == 0 and i > 0:
                # Add skip connection
                self.skip_connections.append(nn.Conv1d(in_channels, n_filters * (len(kernel_sizes) + 1), kernel_size=1))

            module = InceptionModule(in_channels, n_filters, bottleneck_channels, kernel_sizes, activation_function)
            self.inception_modules.append(module)
            in_channels = n_filters * (len(kernel_sizes) + 1)  # Update in_channels for the next module

        # Bidirectional LSTM layers with dropout
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        lstm_input_size = in_channels
        for hidden_size in lstm_hidden_sizes:
            lstm_layer = nn.LSTM(input_size=lstm_input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            lstm_input_size = hidden_size * 2  # Multiply by 2 for bidirectional

        # The last hidden size * 2 (for bidirectional) will be the input feature size to the dense layer
        last_lstm_output_size = lstm_hidden_sizes[-1] * 2

        # Densely connected layer
        self.fc = nn.Linear(last_lstm_output_size, n_classes)

    def forward(self, x):
        skip_connection = None

        for i, module in enumerate(self.inception_modules):
            if self.use_skip_connections and i % 3 == 0 and i > 0:
                # Apply skip connection
                if skip_connection is not None:
                    x = x + self.skip_connections[i // 3 - 1](skip_connection)
                skip_connection = x
            x = module(x)

        # Pass through bidirectional LSTM layers with dropout
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm_layer(x)
            x = dropout_layer(x)

        # Pass through the dense layer
        x = self.fc(x)

        return x










# Parameters for the input data
batch_size = 10
in_channels = 2  # Number of input channels (e.g., features in your time series)
depth = 6
n_classes = 4  # Number of classes (output size)
lstm_hidden_sizes = [128, 64]  # Size of the hidden states in each LSTM layer
sequence_length = 100  # Length of the time series

input_data = torch.randn(batch_size, sequence_length, in_channels)
print("Input shape:", input_data.shape)


# Initialize the Inception module
model = InceptionFRET(in_channels, n_classes)

# Pass the input data through the Inception module
output_data = model(input_data)

print("Output shape:", output_data.shape)






#
#
#
#
# # Initialize the Inception module
# inception_module = InceptionModule(in_channels)
#
# # Pass the input data through the Inception module
# output_data = inception_module(input_data)
#
# print("Output shape:", output_data.shape)




# # Parameters for the input data
# batch_size = 10
# in_channels = 2  # Number of input channels (e.g., features in your time series)
# sequence_length = 100  # Length of the time series
# kernel_size = 40
#

#
# padding = kernel_size // 2
#
# conv = nn.Conv1d(2, 2, kernel_size=kernel_size, padding=40//2, stride=1)
# output = conv(input_data)
#
# print("Output shape:", output.shape)
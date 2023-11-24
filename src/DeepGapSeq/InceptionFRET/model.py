import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F







class lasiModule(nn.Module):
    
    def __init__(self, in_channels, block_filters=20, max_prime=23, res=None, pool=False, activation=F.relu):
        
        super(lasiModule, self).__init__()
        
        self.pool = pool
        self.res = res
        self.activation = activation
        self.max_prime= max_prime
        self.block_filters = block_filters
        self.in_channels = in_channels
        
        self.conv_branches = nn.ModuleList()
        
        self.poolconv = nn.Conv1d(in_channels, block_filters, kernel_size=1, padding="same")
        
        if self.pool:
            in_channels = block_filters
        
        for number in range(2, self.max_prime + 1):
            if all(number % i != 0 for i in range(2, number)):
                conv = nn.Conv1d(in_channels, block_filters, kernel_size=number, padding="same")
                self.conv_branches.append(conv)
        
        self.batchnorm = nn.BatchNorm1d(block_filters * len(self.conv_branches))
        self.branch_batchnorm = nn.BatchNorm1d(block_filters)
        self.activation_function = activation
        
    def forward(self, x):
        
        if self.pool:
            x = self.poolconv(x)
        
        conv_branches = [self.branch_batchnorm(branch(x)) for branch in self.conv_branches]
        
        x = torch.cat(conv_branches, dim=1)
        x = self.batchnorm(x)
        if self.res:
            x = self.activation_function(x)
            
        return x

class Conv1DBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation=F.relu):
        super(Conv1DBN, self).__init__()
        self.activation = activation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding = "same")
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class FinalCNN(nn.Module):
    def __init__(self, in_channels, block_filters=32, res=None, activation=F.relu):
        super(FinalCNN, self).__init__()
        self.res = res
        self.activation = activation

        # Create branches
        self.branch1 = Conv1DBN(in_channels, block_filters, 1, activation=activation)
        self.branch2 = Conv1DBN(in_channels, block_filters, 2, activation=activation)

    def forward(self, x):
        branch_output1 = self.branch1(x)
        branch_output2 = self.branch2(x)

        # Concatenate branches
        m = torch.cat((branch_output1, branch_output2), dim=1)

        # Apply residual connection
        if self.res is not None:
            m = m + self.res

        # Activation function
        x = self.activation(m)

        return x






class lasiModel(nn.Module):
    
    def __init__(self,
            in_channels = 2,
            n_classes = 2,
            model_type = "trace_classifier",
            max_prime = 23,
            resnet = True,
            activation_function=F.relu):
        
        super(lasiModel, self).__init__()
        
        self.resnet = resnet
        self.conv_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        if resnet:
            n_filters = 32
            lstm_hidden_layers = [128,32]
            dropout_list = [0.2,0.5]
        elif model_type == "trace_classifier":
            n_filters = 32
            lstm_hidden_layers = [128,64]
            dropout_list = [0.1,0.5]
        else:
            n_filters = 64
            lstm_hidden_layers = [128,128,128]
            dropout_list = [0.5,0.5]
        

        
        if resnet:
            
            for i in range(4):
                
                if i % 2 != 0:
                    res = True
                else:
                    res = False
                    
                kernel_sizes = self.get_kernel_sizes(max_prime)
                    
                layer = lasiModule(in_channels, n_filters, max_prime, res)
                in_channels = len(kernel_sizes)*n_filters

                self.conv_layers.append(layer)
                
        elif model_type == "trace_classifier": 
            
            pass
            
            
                
        
        lstm_input_size = in_channels
        for hidden_size, dropout_size in zip(lstm_hidden_layers, dropout_list):
            lstm_layer = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, batch_first=True, bidirectional=True)
            dropout_layer = nn.Dropout(dropout_size)
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(dropout_layer)
            lstm_input_size = hidden_size * 2  # Multiply by 2 for bidirectional
        
        
        self.final_dropout = nn.Dropout(0.5)
        
        # Densely connected layer
        last_lstm_output_size = lstm_hidden_layers[-1] * 2
        self.fc = nn.Linear(last_lstm_output_size, n_classes)
        
    def get_kernel_sizes(self, max_prime):
        
        kernel_sizes = []
        
        for number in range(2, max_prime + 1):
            if all(number % i != 0 for i in range(2, number)):
                kernel_sizes.append(number)
                
        return kernel_sizes
    
        
    def forward(self, x):
        
        for layer in self.conv_layers:
            # print(layer)
            x = layer(x)
        
        # Ensure x is in the shape [batch_size, sequence_length, features] for LSTM
        x = x.transpose(1, 2)
        
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm_layer(x)
            x = dropout_layer(x)
        
        x = self.final_dropout(x)
        
        x = self.fc(x)
        
        # return x to input shape [batch_size, features, sequence_length] for LSTM
        x = x.transpose(1, 2)
        
        return x
    
    
    
# # Parameters for the input data
# batch_size = 3
# in_channels = 2  # Number of input channels (e.g., features in your time series)
# depth = 6
# n_classes = 2  # Number of classes (output size)
# sequence_length = 500

# input_data = torch.randn(batch_size, in_channels, sequence_length)
# print("Input shape:", input_data.shape)

# model = FinalCNN(in_channels=2)

# output_data = model(input_data)

# print("Output shape:", output_data.shape)











class InceptionFRET(nn.Module):

    def __init__(self,
            in_channels,
            n_classes = 2,
            depth=6,
            lstm_hidden_sizes = [128,128,128],
            dropout_rate=0.5,
            use_skip_connections=True,
            n_filters=40,
            bottleneck_channels=32,
            kernel_sizes=[9, 19, 39, 99],
            activation_function=F.relu):

        super(InceptionFRET, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.n_classes = n_classes
        
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
                if skip_connection is not None:
                    x = x + self.skip_connections[i // 3 - 1](skip_connection)
                skip_connection = x
            x = module(x)

        # global_avg_pool = ChannelwiseGlobalAvgPool()
        # x = global_avg_pool(x)  # Output shape: [batch_size, 1, sequence_length]

        # Ensure x is in the shape [batch_size, sequence_length, features] for LSTM
        x = x.transpose(1, 2)

        # Pass through bidirectional LSTM layers with dropout
        for lstm_layer, dropout_layer in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm_layer(x)
            x = dropout_layer(x)

        x = self.fc(x)

        # # Ensure output is in the shape [batch_size, channels, sequence_length]
        x = x.transpose(1, 2)

        return x


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
        x = self.bottleneck(x)

        conv_branches = [branch(x) for branch in self.conv_branches]

        # MaxPool branch
        maxpool_branch = self.maxpool(x)
        conv_1x1_branch = self.maxpool_conv(maxpool_branch)
        conv_branches.append(conv_1x1_branch)

        x = torch.cat(conv_branches, dim=1)
        x = self.batchnorm(x)
        x = self.activation_function(x)

        return x



class ChannelwiseGlobalAvgPool(nn.Module):
    def forward(self, x):
        # x shape: [batch_size, channels, sequence_length]
        return torch.mean(x, dim=1, keepdim=True)  # Averages across channels


class InceptionTime(nn.Module):

    def __init__(self, in_channels, n_classes=2, depth=6, use_skip_connections=True, n_filters=40, bottleneck_channels=32, kernel_sizes=[9, 19, 29, 39, 49], activation_function=F.relu):
        super(InceptionTime, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.n_classes = n_classes

        self.inception_modules = nn.ModuleList()
        self.skip_connections = nn.ModuleList() if use_skip_connections else None

        for i in range(depth):
            if use_skip_connections and i % 1 == 0 and i > 0:
                # Add skip connection
                self.skip_connections.append(nn.Conv1d(in_channels, n_filters * (len(kernel_sizes) + 1), kernel_size=1))

            module = InceptionModule(in_channels, n_filters, bottleneck_channels, kernel_sizes, activation_function)
            self.inception_modules.append(module)
            in_channels = n_filters * (len(kernel_sizes) + 1)  # Update in_channels for the next module

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Calculate the correct number of input features for the dense layer
        num_features = n_filters * (len(kernel_sizes) + 1)

        # Initialize the dense layer with the correct number of input features
        self.fc = nn.Linear(num_features, n_classes)

    def forward(self, x):
        skip_connection = None

        for i, module in enumerate(self.inception_modules):
            if self.use_skip_connections and i % 3 == 0 and i > 0:
                if skip_connection is not None:
                    x = x + self.skip_connections[i // 3 - 1](skip_connection)
                skip_connection = x
            x = module(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)  # [batch_size, channels, 1]
        x = x.view(x.size(0), -1)  # Flatten [batch_size, channels]

        # Pass through the dense layer
        x = self.fc(x)

        return x








# # Parameters for the input data
# batch_size = 3
# in_channels = 2  # Number of input channels (e.g., features in your time series)
# depth = 6
# n_classes = 2  # Number of classes (output size)
# lstm_hidden_sizes = [128, 64]  # Size of the hidden states in each LSTM layer
# sequence_length = 500  # Length of the time series

# input_data = torch.randn(batch_size, in_channels, sequence_length)
# print("Input shape:", input_data.shape)

# model = InceptionFRET(in_channels, n_classes)
# output_data = model(input_data)

# print("Output shape:", output_data.shape)





# # Initialize the Inception module
# model = InceptionFRET(in_channels, n_classes)
#
# # Pass the input data through the Inception module
# output_data = model(input_data)
#
# print("Output shape:", output_data.shape)
#


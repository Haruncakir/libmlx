import torch
import torch.nn as nn

# First, define the VisNet model class (needed to load the complete model)
class VisNet(nn.Module):
    def __init__(self):
        super(VisNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the complete model
complete_model = torch.load("complete_visnet_model.pth", weights_only=False)

# Print model architecture
print("=== Model Architecture ===")
print(complete_model)

# Print model's state_dict
print("\n=== Model State Dictionary ===")
for param_name, param in complete_model.state_dict().items():
    print(f"{param_name}: {param.shape}")

# Examine specific layer details and parameters
print("\n=== Detailed Layer Examination ===")
for name, module in complete_model.named_modules():
    if name:  # Skip the empty name (root module)
        print(f"\nLayer: {name}")
        print(f"  Type: {type(module).__name__}")
        
        # Get parameters for this module
        module_params = {k: v for k, v in complete_model.state_dict().items() if k.startswith(name)}
        if module_params:
            print("  Parameters:")
            for param_name, param in module_params.items():
                print(f"    {param_name}: {param.shape}")
                
                # Print a small sample of values for each parameter
                flat_param = param.flatten()
                sample_size = min(5, flat_param.numel())
                print(f"      Sample values: {flat_param[:sample_size].tolist()}")

# Print total parameters count
total_params = sum(p.numel() for p in complete_model.parameters())
print(f"\nTotal Parameters: {total_params}")

# Check if model is in training or evaluation mode
print(f"Model is in {'training' if complete_model.training else 'evaluation'} mode")
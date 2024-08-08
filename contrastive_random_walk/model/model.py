import torch
import torch.nn as nn
import torchvision.models as models
import einops

class ResNetPatchEncoder(nn.Module):
    def __init__(self, resnet_type='resnet18', output_dim=128):
        super(ResNetPatchEncoder, self).__init__()
        
        # Load a pre-trained ResNet model
        resnet_model = getattr(models, resnet_type)(pretrained=True)
        
        # Remove the fully connected layer (classifier) from the ResNet model
        self.resnet_encoder = nn.Sequential(*list(resnet_model.children())[:-2]) # output size: (B*N, C1, H1, W1)
        
        # Get the output dimensions after passing through ResNet
        # TODO: This is a hacky way to get the output dimensions. Is there a better way?
        dummy_input = torch.zeros(1, 3, 224, 224)
        dummy_output = self.resnet_encoder(dummy_input)
        self.feature_size = dummy_output.shape[1] * dummy_output.shape[2] * dummy_output.shape[3] # C1 * H1 * W1
        
        # Linear layer to project the flattened ResNet features to the desired output dimensions (D)
        self.fc = nn.Linear(self.feature_size, output_dim)

    def forward(self, x):
        # x shape: (B, N, H, W, C)
        
        B, N, H, W, C = x.shape

        assert C == 3, "Input tensor must have 3 channels"

        # permute to (B, N, C, H, W)
        x = einops.rearrange(x, 'b n h w c -> b n c h w')
        
        # Reshape x to combine batch and node dimensions for efficient processing (use einops)
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        
        # Pass through ResNet encoder
        x = self.resnet_encoder(x) # shape: (B*N, C1, H1, W1)
        
        # Flatten the output of the ResNet encoder (use einops)
        x = einops.rearrange(x, 'b c h w -> b (c h w)')
        
        # Pass through the linear projection layer
        x = self.fc(x) # shape: (B*N, D)

        # Apply L2 normalization so that the output features have unit norm (the D dimension) has unit norm
        x = nn.functional.normalize(x, p=2, dim=-1)
        
        # Reshape back to include the batch and node dimensions (use einops)
        x = einops.rearrange(x, '(b n) d -> b n d', b=B, n=N)
        
        return x

# Example usage:
encoder = ResNetPatchEncoder(resnet_type='resnet18', output_dim=128)
input_tensor = torch.randn(16, 10, 224, 224, 3) # (B, N, H, W, C)
output = encoder(input_tensor) # output shape: (B, N, D)

print(output.shape)

# assert the unit norm
assert torch.isclose(output.norm(dim=-1), torch.ones_like(output.norm(dim=-1))).all()
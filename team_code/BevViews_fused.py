import torch
import torch.nn as nn

class CrossViewTransformer(nn.Module):
    """
    Fuses features from multiple camera views into a Bird's-Eye View (BEV) representation
    using cross-attention.
    """
    def __init__(self, in_channels, bev_size, num_heads=8, head_dim=32): 
        """
        Args:
            in_channels (int): Number of channels in the input multi-view features (C).
            bev_size (tuple): Target BEV grid size (bev_h, bev_w).
            num_heads (int): Number of attention heads.
            head_dim (int): Dimension of each attention head. embed_dim = num_heads * head_dim.
        """
        super().__init__()
        self.bev_h, self.bev_w = bev_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Learnable query representing the BEV grid structure
        # Shape: (1, embed_dim, bev_h, bev_w)
        self.bev_query = nn.Parameter(torch.randn(1, in_channels, self.bev_h, self.bev_w))

        # MultiheadAttention layer configured for cross-attention
        # It expects query, key, value inputs.
        # We need batch_first=True because our tensors are shaped (Batch, SeqLen, EmbedDim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_channels, 
            num_heads=self.num_heads, 
            batch_first=True  # VERY IMPORTANT!
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor containing features from 6 views, concatenated along batch dim.
                              Expected shape: (B*6, C, H, W), where C = embed_dim.

        Returns:
            torch.Tensor: Output BEV feature map. Shape: (B, C, bev_h, bev_w).
        """
        # 1. Extract Dimensions and Calculate Batch Size
        B6, C_in, H, W = x.shape
        B = B6 // 6
        
        # 2. Prepare BEV Query (Q)
        # Repeat the learnable query for each batch item
        # Shape: (B, C, bev_h, bev_w)
        bev_q = self.bev_query.repeat(B, 1, 1, 1) 
        # Flatten spatial dimensions (h*w) and permute for attention layer
        # Shape: (B, N_q = bev_h * bev_w, C)
        bev_q = bev_q.flatten(2).permute(0, 2, 1) 

        # 3. Prepare Multi-View Features (K, V)
        # Reshape to separate batch and view dimensions
        # Shape: (B, 6, C, H, W)
        x = x.view(B, 6, C_in, H, W)
        # Permute to group batch, views, spatial dims before flattening
        # Shape: (B, 6, H, W, C) 
        x = x.permute(0, 1, 3, 4, 2) 
        # Flatten the view, H, W dimensions into a single sequence dimension (N_kv)
        # Shape: (B, N_kv = 6 * H * W, C)
        x_flat = x.reshape(B, 6 * H * W, C_in)

        # 4. Perform Cross-Attention
        # Query: BEV grid locations (bev_q)
        # Key/Value: Flattened multi-view features (x_flat)
        # Attention output 'out' will have the same shape as the query
        # Shape: (B, N_q = bev_h * bev_w, C)
        out, _ = self.cross_attn(
            query=bev_q,
            key=x_flat,
            value=x_flat
        )

        # 5. Reshape Output to BEV Grid
        # Permute to bring channel dimension (C) forward
        # Shape: (B, C, N_q = bev_h * bev_w)
        out = out.permute(0, 2, 1)
        # Reshape the sequence dimension (N_q) back into spatial BEV dimensions (bev_h, bev_w)
        # Shape: (B, C, bev_h, bev_w)
        bev_features = out.reshape(B, C_in, self.bev_h, self.bev_w)
        
        return bev_features

# --- Example Usage (for demonstration) ---
if __name__ == '__main__':
    # Example Parameters (adjust based on your actual model)
    batch_size = 2
    num_views = 6
    img_feat_channels = 256 # Should match num_heads * head_dim
    img_feat_h = 16
    img_feat_w = 32
    bev_h_target = 100
    bev_w_target = 100
    num_attention_heads = 8
    attention_head_dim = 32 # num_heads * head_dim = 8 * 32 = 256 = img_feat_channels

    # Create dummy input: 6 views concatenated along batch dim
    dummy_input = torch.randn(batch_size * num_views, img_feat_channels, img_feat_h, img_feat_w)

    # Instantiate the transformer
    transformer = CrossViewTransformer(
        in_channels=img_feat_channels, 
        bev_size=(bev_h_target, bev_w_target), 
        num_heads=num_attention_heads,
        head_dim=attention_head_dim 
    )

    # Pass input through the transformer
    try:
        bev_output = transformer(dummy_input)
        print("Input shape:", dummy_input.shape)
        print("BEV Output shape:", bev_output.shape) # Expected: [B, C, bev_h, bev_w]
        print("Expected output shape:", (batch_size, img_feat_channels, bev_h_target, bev_w_target))
        assert bev_output.shape == (batch_size, img_feat_channels, bev_h_target, bev_w_target)
        print("Success!")
    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
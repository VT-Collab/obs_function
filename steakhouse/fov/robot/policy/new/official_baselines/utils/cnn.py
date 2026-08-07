
"""
CNN layer
ref: https://github.com/sjtu-marl/ZSC-Eval/tree/master

Params:
    args       -- config; uses args.hidden_size 
    obs_shape  -- (C, width, height), channel-first, as built by utils/features.py
                  (C == features.N_LAYERS == 23)
    kernel_size / stride / padding -- conv geometry; k=3,s=1,p=1 keeps W,H unchanged
    output_size -- width of the emitted feature vector; downstream modules
                   (RNNLayer, actor/critic heads) read self.output_size

input        (N, 23, W, H)      stack of 23 grids

Conv2d(23→32) + ReLU            32 grids of learned local patterns
Conv2d(32→64) + ReLU            64 grids, more abstract
Conv2d(64→32) + ReLU            32 grids, more abstract still

Flatten                         (N, 32*W*H)   one long line
Linear(32*W*H → 64) + ReLU + LayerNorm
Linear(64 → output_size) + ReLU + LayerNorm

output       (N, output_size)   fixed-size feature vector
"""


import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)
    
class CNNBase(nn.Module):
    def __init__(self, args, obs_shape, kernel_size=3, stride=1, padding=1, output_size=None):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        #width of the feature vector this module emits, set independently of
        #hidden_size -- downstream modules (RNNLayer, actor/critic) read this
        #fall back to hidden_size when the caller leaves it unset (None or 0)
        self.output_size = output_size if output_size else self.hidden_size

        self.cnn = self._build_cnn_model(
            obs_shape=obs_shape,
            kernel_size=self.kernel_size,
            stride = self.stride,
            padding = self.padding,
            hidden_size = self.hidden_size,
            output_size = self.output_size
        )

        #orthogonal init on conv/linear weights, zero bias
        #LayerNorm is skipped on purpose -- it keeps its default weight=1, bias=0
        #makes sure initialization of the weights are orthogonal matrices, which never changes the vector's length
        for m in self.cnn.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(m.bias, 0)


    def _build_cnn_model(
        self, 
        obs_shape,
        kernel_size,
        stride,
        padding,
        hidden_size,
        output_size,
    ):
        input_channel = obs_shape[0]
        w = obs_shape[1]
        h = obs_shape[2]
        
        #compute how the geometry changed as it pass through each Conv2d
        for _ in range(3): #3 Conv2d's
            w = (w + 2*padding - kernel_size) // stride + 1
            h = (h + 2*padding - kernel_size) // stride + 1
            
        
        return nn.Sequential(
            
            #First layer example: Conv2d(C,  32, k=3, s=1, p=1) → ReLU
            nn.Conv2d(
                in_channels=input_channel,
                out_channels= hidden_size//2, #// means floor division
                kernel_size=kernel_size,
                stride = stride,
                padding = padding
                ),
            
            nn.ReLU(),
            
            #Second layer example: Conv2d(32, 64, k=3, s=1, p=1) → ReLU
            nn.Conv2d(
                in_channels=hidden_size//2,
                out_channels=hidden_size,
                kernel_size=kernel_size,
                stride = stride,
                padding = padding
                ),
            
            nn.ReLU(),
            
            #Third layer example: Conv2d(64, 32, k=3, s=1, p=1) → ReLU
            nn.Conv2d(
                in_channels=hidden_size,
                out_channels=hidden_size//2,
                kernel_size=kernel_size,
                stride = stride,
                padding = padding
                ),
            
            nn.ReLU(),
            
            Flatten(),
            
            #Linear(32*W*H → 64) + ReLU + LayerNorm
            nn.Linear((hidden_size // 2)*w*h, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=hidden_size),

            #Linear(64 → output_size)     + ReLU + LayerNorm
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=output_size)
            
        )
        
    def forward(self, x):
        x = self.cnn(x)
        return x

    
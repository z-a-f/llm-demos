import math

import torch
from torch import nn


######################################################
## ViT Embedding

class PatchProjection(nn.Module):
    '''Converts the image into patches and then project them into a vector space.

    This is the fusion of "patching" and "linear projection of flattened patches"

    Args:
        config (dict): dict containing the following keys
            image_size (int): Size of the image. We will assume a square image (W == H)
            patch_size (int): Size of each patch to split the image into
            num_channels (int): Number of channels in the input image (Cin).
            hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.
    '''

    def __init__(self, config):
        super().__init__()
        self.image_size = config['image_size']
        self.patch_size = config['patch_size']
        self.num_channels = config['num_channels']
        self.hidden_size = config['hidden_size']
        # We will assume that the image size is the same for the width and height. Same for the patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size,
                                    kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        '''Applies the projection and transformation to generate the patch representations.

        Note:
            Cout == hidden_size
            Hout * Wout == num_patches
        '''
        x = self.projection(x)  # (N, Cin, Hin, Win) -> (N, Cout, Hout, Wout)
        x = x.flatten(2)        # (N, Cout, Hout, Wout) -> (N, Cout, Hout*Wout)
        x = x.transpose(1, 2)   # (N, Cout, Hout*Wout) -> (N, Hout*Wout, Cout)
        return x


class VisionEmbedding(nn.Module):
    '''Combines the patch projections with the class token and position embeddings.

    Args:
        config (dict): Configuration with the following keys
            image_size (int): Size of the image. We will assume a square image (W == H)
            patch_size (int): Size of each patch to split the image into
            num_channels (int): Number of channels in the input image (Cin).
            hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.

            hidden_dropout (float): Dropout probability for the output
            position_method (str): Method of generating the positional embedding.
                'random': Randomly initialized positional vector
                'range': Initialize positional vector as a list of numbers in the range [0, 1]
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_projections = PatchProjection(config)

        self.cls = nn.Parameter(torch.randn(1, 1, config['hidden_size']))

        if config['position_method'] == 'random':
            pos_tensor = torch.randn(1, self.patch_projections.num_patches + 1, config['hidden_size'])
        elif config['position_method'] == 'range':
            pos_tensor = torch.linspace(0, 1.0, self.patch_projections.num_patches + 1)   # (num_patches+1,)
            pos_tensor = pos_tensor.view(1, self.patch_projections.num_patches + 1, 1)  # (num_patches+1,) -> (1, num_patches+1, 1)
            pos_tensor = pos_tensor.expand(-1, self.patch_projections.num_patches + 1, config['hidden_size'])
        self.position_embeddings = nn.Parameter(pos_tensor)
        self.dropout = nn.Dropout(config['hidden_dropout'])

    def forward(self, x):
        x = self.patch_projections(x)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls.expand(batch_size, -1, -1)  # We have to expand it, as concatenation is not broadcastable (https://stackoverflow.com/a/50425863)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    
######################################################
## Attention

class AttentionHead(nn.Module):
    '''A single attention head.

    Unlike the torch implementation, we will split this into 3 separate linear layers (qkv)

    Args:
        hidden_size (int): Input size for QKV (Cin). The input shape is thus (N, L, Cin), where L is the sequence length
        head_size (int): Intermediate size of this head (Cout). The intermediate shape is this (N, L, Cout)

    Notes:
        The input of shape (N, L, Cin) is projected onto 3x (N, L, Cout) to generate the QKV representations
        After that QKV is used as per https://arxiv.org/abs/1706.03762

    '''
    def __init__(self, hidden_size, head_size, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_size = head_size
        # Create the query, key, and value projection layers
        self.query = nn.Linear(hidden_size, head_size, bias=bias)
        self.key = nn.Linear(hidden_size, head_size, bias=bias)
        self.value = nn.Linear(hidden_size, head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Calculate the attention scores: softmax(Q*K.T/sqrt(head_size))*V
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value)
        return (attention_output, attention_probs)

class MultiHeadAttention(nn.Module):
    '''Multi-head attention module.

    This module is used in the transformer block, and should mimic the `nn.MultiheadAttention`

    Args:
        config (dict): Configuration with keys
            hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.
            num_attention_heads (int): Number of attention heads with MHA
            qkv_bias (bool): Flag to use bias in the QKV projections
            attention_probs_dropout (float): Dropout probability within each attention head
            hidden_dropout (float): Dropout probability for the current block's output
    '''
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_attention_heads = config['num_attention_heads']
        assert self.hidden_size % self.num_attention_heads == 0, f"Let's keep the hidden dimension divisible by number of heads"
        # The attention head size is the hidden size divided by the number of attention heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.qkv_bias = config['qkv_bias']

        # Create a list of attention heads
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config['attention_probs_dropout'],
                self.qkv_bias
            )
            self.heads.append(head)

        # Create a linear layer to project the attention output back to the hidden size
        # In most cases, all_head_size and hidden_size are the same
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config['hidden_dropout'])

    def forward(self, x, output_attentions=False):
        # Calculate the attention output for each attention head
        attention_outputs = [head(x) for head in self.heads]
        # Concatenate the attention outputs from each attention head
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        # Project the concatenated attention output back to the hidden size
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        # Return the attention output and the attention probabilities (optional)
        attention_probs = None
        if output_attentions:
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
        return (attention_output, attention_probs)

######################################################
## MLP
    
class NewGELUActivation(nn.Module):
    '''Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    From https://github.com/huggingface/transformers/blob/415e9a0980b00ef230d850bff7ecf0021c52640d/src/transformers/activations.py#L50)
    '''

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class MLP(nn.Module):
    '''Multilayer perceptron.

    Args:
        config (dict): Configuration with keys
            hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.
            intermediate_size (int): Number of intermediate channels between two linear layers
            hidden_dropout (float): Dropout probability for the output
    '''
    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config['intermediate_size'], config['hidden_size'])
        self.dropout = nn.Dropout(config['hidden_dropout'])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

######################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        '''A single transformer layer

        Args:
            config (dict): Configuration dict. The keys are the same as the multihead attention and MLP
                hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.

        '''
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm_1 = nn.LayerNorm(config['hidden_size'])
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(config['hidden_size'])

    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = \
            self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        # Skip connection
        x = x + attention_output
        # Feed-forward network
        mlp_output = self.mlp(self.layernorm_2(x))
        # Skip connection
        x = x + mlp_output
        # Return the transformer block's output and the attention probabilities (optional)
        if output_attentions:
            return (x, attention_probs)
        else:
            return (x, None)

######################################################
## Encoder Model

class ViTEncoder(nn.Module):
    r'''Encoder wrapper around multiple transformer blocks

    Args:
        config (dict): Configuration with keys the same as transformer block
            num_hidden_layers (int): Number of transformer block to use for encoding
    '''
    def __init__(self, config):
        super().__init__()
        # Create a list of transformer blocks
        self.blocks = nn.ModuleList([])
        for _ in range(config['num_hidden_layers']):
            block = TransformerBlock(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        return (x, all_attentions)

class ViTModel(nn.Module):
    '''ViT classification model

    Args:
        config (dict): Configuration for the model. Keys should be the same as VisionEmbedding and VitEncoder
            hidden_size (int): The number of channels that the patches will be converted into (Cout).
                               This is the size of the projection space.
            num_classes (int): Number of class outputs
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_classes = config['num_classes']

        # Embeddings followed by the encoder
        self.embedding = VisionEmbedding(config)
        self.encoder = ViTEncoder(config)

        # This is the classification head
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # TODO: The embeddings should be initialized to a smaller range

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0])  # Take only the [CLS] token
        return (logits, all_attentions)
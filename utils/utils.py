import collections
import numpy as np
import torch

@torch.no_grad()
def num_parameters(model, as_table=False):
    '''Computes the number of parameters in the ViTModel

    Note that this will only work for the ViT model described in this notebook
    '''
    from .models import TransformerBlock  # Avoid circular import
    stats = {'total': {}, 'trainable': {}}  # We will ignore the buffers

    # Make a list of modules to traverse
    stats['total']['embedding'] = sum([param.numel() for param in model.embedding.parameters()])
    stats['trainable']['embedding'] = sum([param.numel() if param.requires_grad else 0 for param in model.embedding.parameters()])

    nodes = collections.deque([('encoder', model.encoder)])  # depth first
    while nodes:
        prefix, node = nodes.popleft()
        if isinstance(node, TransformerBlock):
            block_stats = {'total': {'attention': 0, 'mlp': 0}, 'trainable': {'attention': 0, 'mlp': 0}}

            block_stats['total']['attention'] = sum([param.numel() for param in node.attention.parameters()])
            block_stats['trainable']['attention'] = sum([param.numel() if param.requires_grad else 0 for param in node.attention.parameters()])
            block_stats['total']['mlp'] = sum([param.numel() for param in node.mlp.parameters()])
            block_stats['trainable']['mlp'] = sum([param.numel() if param.requires_grad else 0 for param in node.mlp.parameters()])

            stats['total'][prefix] = block_stats['total']
            stats['trainable'][prefix] = block_stats['trainable']
        else:
            children = [(prefix + '.' + name, child) for name, child in node.named_children()][::-1]
            nodes.extendleft(children)

    stats['total']['classifier'] = sum([param.numel() for param in model.classifier.parameters()])
    stats['trainable']['classifier'] = sum([param.numel() if param.requires_grad else 0 for param in model.classifier.parameters()])

    if as_table:
        columns = ['Layer', 'Total', 'Trainable', 'Trainable (%)']
        max_widths = [len(col) for col in columns]

        rows = []
        total = [0, 0]
        for name in stats['total'].keys():
            if isinstance(stats['total'][name], dict):
                rows.append([name, '', '', ''])
                max_widths[0] = max(max_widths[0], len(rows[-1][0]))
                for key in stats['total'][name].keys():
                    rows.append([f'| {key}', f"{stats['total'][name][key]:,}", f"{stats['trainable'][name][key]:,}",
                                 f'{stats["trainable"][name][key] / stats["total"][name][key]:.2%}'])
                    total[0] += stats["trainable"][name][key]
                    total[1] += stats["total"][name][key]
                    max_widths = [max(max_widths[idx], len(rows[-1][idx])) for idx in range(len(columns))]
            else:
                rows.append([name, f"{stats['total'][name]:,}", f"{stats['trainable'][name]:,}",
                             f'{stats["trainable"][name] / stats["total"][name]:.2%}'])
                total[0] += stats["trainable"][name]
                total[1] += stats["total"][name]
                max_widths = [max(max_widths[idx], len(rows[-1][idx])) for idx in range(len(columns))]
        totals_row = ['Total', f'{total[1]:,}', f'{total[0]:,}', f'{total[0]/total[1]:.2%}']

        max_widths[1] = max(max_widths[1], len(totals_row[1]))
        max_widths[2] = max(max_widths[2], len(totals_row[2]))

        table = [' | '.join([f'{columns[idx]:^{max_widths[idx]}}' for idx in range(len(columns))])]
        table.append(' | '.join([f'{"":-<{max_widths[idx]}}' for idx in range(len(columns))]))
        for row in rows:
            table.append(' | '.join([f'{row[0]:<{max_widths[0]}}'] + [f'{row[idx]:>{max_widths[idx]}}' for idx in range(1, len(columns))]))
        table.append(' | '.join([f'{"":-<{max_widths[idx]}}' for idx in range(len(columns))]))
        table.append(' | '.join([f'{totals_row[idx]:>{max_widths[idx]}}' for idx in range(len(columns))]))

        return '\n'.join(table)

    return stats

def compute_mean_std(dataset):
    r'''Use Welford's method to get the mean and std

    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

    We use Welford instead of standard, as the sum of all values might be extremely large
    '''
    single_sample = np.array(dataset[0][0])
    num_channels = 1 if single_sample.ndim < 3 else single_sample.shape[2]
    mean = np.zeros((num_channels, ))
    M2 = np.zeros((num_channels, ))
    count = 0

    for image, _ in dataset:
        image = np.array(image) / 255.0
        image = image.reshape(*image.shape[:2], num_channels)
        count += (image.shape[0] * image.shape[1])
        delta = image - mean
        mean += np.sum(delta, (0, 1)) / count
        M2 += np.sum(delta * (image - mean), (0, 1))

    variance = M2 / count
    std_dev = np.sqrt(variance)

    return mean, std_dev

def freeze_layers(model, layer_list, unless=None):
    '''Freezes the layers in the layer list.
    
    Args:
        model: PyTorch model
        layer_list: List of layer names to freeze
        unless: List of layer or parameter names to not freeze

    Note:
        - 'layer_list' is recursive, unless one of the parameter names is in 'unless'
    '''
    unless = unless or []
    for layer_name in layer_list:
        layer = model.get_submodule(layer_name)
        for param_name, param in layer.named_parameters():
            if param_name.split('.')[-1] in unless:
                continue
            if layer_name + '.' + param_name in unless:
                continue
            param.requires_grad = False
    return model

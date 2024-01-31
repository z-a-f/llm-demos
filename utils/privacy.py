import torch

import numpy as np

@torch.no_grad()
def simple_mia(model, train_subset, val_subset, criterion,
               n_splits=5, max_samples_per_subset=100_000):
    r'''Simple MIA based on per-sample loss distribution.

    Args:
        model: PyTorch model
        train_subset: Dataloader that the model was trained on
        val_subset: Dataloader that the model never saw during training
        criterion: Loss function. IMPORTANT: Make sure the 'reduction' is set to "none"
        n_splits: Number of splits to use for cross-validation

    Returns:
        1. Cross-validation score for a succesful attack
        2. 
    '''
    from sklearn import linear_model, model_selection
    assert getattr(criterion, 'reduction', 'none') == 'none'

    # Step 1. Get the member and non-member losses
    all_losses = []
    all_members = []
    with torch.no_grad():
        # Lightning doesn't like when others mess with its internal parameters
        # -- save the old to restore later
        _old_training = model.training
        model.eval()
        for member, dataloader in zip([1, 0], [train_subset, val_subset]):
            losses = []
            for x, y in dataloader:
                y_hat = model(x)
                if isinstance(y_hat, tuple):
                    y_hat = y_hat[0]  # Ignore the auxillary output
                loss = criterion(y_hat, y).cpu().detach().numpy().tolist()
                losses.extend(loss)
            losses = losses[:max_samples_per_subset]
            all_losses.extend(losses)
            all_members.extend([member] * len(losses))
        model.training = _old_training
    all_losses = np.array(all_losses).reshape(-1, 1)
    all_members = np.array(all_members)

    # Step 2. Create an attack model
    attack_model = linear_model.LogisticRegression(class_weight='balanced')
    # Step 3. Create CV splits
    cv = model_selection.StratifiedShuffleSplit(n_splits=n_splits)
    # Step 4. Create CV scores -- replace with 'cross_validate' to produce a model instead of scores
    mia_scores = model_selection.cross_val_score(
        attack_model, all_losses, all_members, cv=cv, scoring='accuracy'
    )
    return mia_scores
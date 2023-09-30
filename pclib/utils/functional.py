def vfe(state, batch_reduction='mean', layer_reduction='sum'):
    if layer_reduction == 'sum':
        vfe = sum([state_i[1].square().sum() for state_i in state])
    elif layer_reduction =='mean':
        vfe = sum([state_i[1].square().mean(dim=1).sum() for state_i in state])
    if batch_reduction == 'mean':
        vfe /= state[0][0].shape[0]
    return vfe
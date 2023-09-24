def vfe(state, reduction='sum'):
    if reduction == 'sum':
        vfe = sum(state_i[1].square().sum(dim=1).mean() for state_i in state)
    elif reduction =='mean':
        vfe = sum(state_i[1].square().mean() for state_i in state)
    return vfe
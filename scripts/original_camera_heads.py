"""Read four learned original controls and the optional PPO value prediction."""


def policy_heads(output, batch_size):
    if len(output) not in (4, 5):
        raise ValueError("Expected four action heads and an optional critic")
    heads = output[:4]
    for head, classes in zip(heads, (9, 2, 2, 2)):
        if tuple(head.shape) != (batch_size, classes):
            raise ValueError("Invalid original policy action distribution shape")
    if len(output) == 4:
        return heads, heads[0].new_zeros(batch_size)
    value = output[4].reshape(-1)
    if value.numel() != batch_size:
        raise ValueError("Expected one PPO value for every physical observation")
    return heads, value

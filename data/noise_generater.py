import numpy as np
"""
Generate noisy labels
"""
def generate_corrupted_labels(label, args):

    if args.noise_type == 'unif':
        C = uniform_mix_C(args.noise_level, args.num_classes)
    elif args.noise_type == 'flip':
        C = flip_labels_C(args.noise_level, args.num_classes)
    elif args.noise_type == 'flip2':
        C = flip_labels_C_two(args.noise_level, args.num_classes)
    else:
        assert False, "Invalid corruption type '{}' given. Must be in {'unif', 'flip', 'hierarchical'}".format(
            args.noise_type)

    for i in range(len(label)):
        label[i] = np.random.choice(args.num_classes, p=C[label[i]])
    return label


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
           (1 - mixing_ratio) * np.eye(num_classes)

def flip_labels_C(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
    return C

def flip_labels_C_two(corruption_prob, num_classes, seed=1):
    '''
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    '''
    np.random.seed(seed)
    C = np.eye(num_classes) * (1 - corruption_prob)
    row_indices = np.arange(num_classes)
    for i in range(num_classes):
        C[i][np.random.choice(row_indices[row_indices != i], 2, replace=False)] = corruption_prob / 2
    return C
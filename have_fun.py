import numpy as np 


def positional_embedding(d_model, num_positions):
    num_positions = num_positions
    d_model = d_model
    pos = np.arange(num_positions)[:, np.newaxis]
    ds = np.arange(d_model // 2)[np.newaxis, :]
    sin_branch = np.sin(pos / 10000 ** (4 * ds / d_model))
    cos_branch = np.cos(pos / 10000 ** (4 * ds / d_model))
    
    all_branch = np.stack([sin_branch, cos_branch], axis=-1)
    all_branch = all_branch.reshape(num_positions, -1)
    return all_branch
    
    



if __name__ == "__main__":
    pos_embed = positional_embedding(d_model=512, num_positions=1000)
    pos_embed_ = positional_embedding(d_model=512, num_positions=1000)
    
    flow_matrix = np.matmul(pos_embed, pos_embed_.T)
    
    
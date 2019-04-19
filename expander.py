import torch
from torch import nn

class MatrixExpander(nn.Module):
    def __init__(self, base_shape, mp):
        super(MatrixExpander, self).__init__()

        Ar, Ac              = base_shape
        Br, Bc              = mp, mp

        Fr, Fc              = Ar * Br, Ac * Bc
    
        left_mat_shape      = (Fr, Ar)
        right_mat_shape     = (Ac, Fc)
    
        ratio_left          = Ar
        ratio_right         = Ac
    
        left_mat            = torch.FloatTensor(*left_mat_shape).fill_(0)
        right_mat           = torch.FloatTensor(*right_mat_shape).fill_(0)
    
        for i in range(ratio_left):
            sr              = i * Br
            er              = sr + Br
    
            left_mat[sr:er, i]  = 1
        
        for j in range(ratio_right):
            sc              = j * Bc
            ec              = sc + Bc
            right_mat[j, sc:ec] = 1

        left_mat            = left_mat.unsqueeze(0).unsqueeze(0)
        right_mat           = right_mat.unsqueeze(0).unsqueeze(0)

        self.register_buffer('left_mat', left_mat)
        self.register_buffer('right_mat', right_mat)

        self.A_expander     = lambda A: torch.matmul(self.left_mat, torch.matmul(A, self.right_mat))

    def forward(self, A):
        A_expanded          = torch.matmul(self.left_mat, torch.matmul(A, self.right_mat))
        return A_expanded


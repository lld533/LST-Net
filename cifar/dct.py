import math
import torch

def dct_init(*args):
    if len(args) == 0:
        k1 = k2 = 3 #default [3,3]
    elif len(args) == 1:
        if isinstance(*args, int):
            v = args[0]
            k1 = k2 = v
        elif isinstance(*args, tuple) or isinstance(*args, list):
            if len(*args) == 1:
                v = args[0]
                k1 = k2 = v[0]
            elif len(*args) > 1:
                v = args[0]
                k1, k2 = v[0], v[1]
            else:
                assert(len(*args) != 0)
    elif len(args) > 1:
        k1, k2 = args[0], args[1]
    else:
        assert(len(args) != 0)

    assert(k1>0 and k2>0)
    
    flag = k1 >= k2
    if k1 < k2:
        k1, k2 = k2, k1

    x, y = [m.type(torch.float32) for m in torch.meshgrid([torch.arange(1,k2), 
                                                           1+2*torch.arange(0,k1)])]
    result = torch.cos(math.pi * x * y / float(2 * k1)) * math.sqrt(2.0 / float(k1))
    component = torch.ones(1, k1) * math.sqrt(1.0 / float(k1))

    result = torch.cat([component, result], axis=0)

    if flag:
        result = result.t()

    return result

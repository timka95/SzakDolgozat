# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use


import torch.nn as nn

# from Class_losses.sampler_new import *
# from Class_losses.reliability_loss_new import *


class MultiLoss_new (nn.Module):
    """ Combines several loss functions for convenience.
    *args: [loss weight (float), loss creator, ... ]
    
    Example:
        loss = MultiLoss( 1, MyFirstLoss(), 0.5, MySecondLoss() )
    """
    def __init__(self, *args):
        nn.Module.__init__(self)
        assert len(args) % 2 == 0, 'args must be a list of (float, loss)'
        self.weights = []
        self.losses = nn.ModuleList()
        for i in range(len(args)//2):
            weight = float(args[2*i+0])
            loss = args[2*i+1]
            assert isinstance(loss, nn.Module), "%s is not a loss!" % loss
            self.weights.append(weight)
            self.losses.append(loss)

    def forward_all(self, select=None, **variables):
        # LOSSES
        # ModuleList(
        #     (0): MSELoss_new()
        # )
        assert not select or all(1<=n<=len(self.losses) for n in select)
        d = dict()
        h = dict()
        cum_loss = 0
        hough_cum_loss = 0
        # WEIGHTS:[1.0]
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses),1):
            if select is not None and num not in select: continue
            # This goes into MSE_loss_new But how and why idk xd
            l = loss_func(**{k:v for k,v in variables.items()})
            #print("VARIBLES:", variables)
            if isinstance(l, tuple):
                hough_l = l[1]
                l=l[0]
                
                #assert len(l) == 2 and isinstance(l[1], dict)
            l = l, {loss_func.name:l}

            cum_loss = cum_loss + weight * l[0]
            for key,val in l[1].items():
                d['loss_'+key] = float(weight * val)


            #HOUGH
            hough_l = hough_l, {loss_func.name: hough_l}

            hough_cum_loss = hough_cum_loss + weight * hough_l[0]
            for key, val in hough_l[1].items():
                h['loss_' + key] = float(weight * val)
        h['loss'] = float(hough_cum_loss)

        return cum_loss, d, hough_cum_loss, h







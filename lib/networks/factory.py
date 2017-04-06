# --------------------------------------------------------
# Scene Graph Generation by Iterative Message Passing
# Licensed under The MIT License [see LICENSE for details]
# Written by Danfei Xu
# --------------------------------------------------------

"""Factory method for easily getting networks by name."""

__sets = {}

from models import *

__sets['vrd'] = vrd # VRD baseline
__sets['dual_graph_vrd_avgpool'] = dual_graph_vrd_avgpool  # avg pooling baseline
__sets['dual_graph_vrd_maxpool'] = dual_graph_vrd_maxpool  # max pooling baseline
__sets['dual_graph_vrd_final'] = dual_graph_vrd_final  # final model

def get_network(name):
    """Get a network by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown network: {}'.format(name))
    return __sets[name]

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()

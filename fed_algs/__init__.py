from .local_train import *
from .global_aggregate import *
from .fedrep_per import *
from .pfedme import *

alg2local = {
    'central': local_train_central,
    'local': local_train_central,
    # gen
    'fedavg': local_train_fedavg,
    'fedavgm': local_train_fedavg,
    'fedprox': local_train_fedprox,
    'fednova': local_train_fedavg,
    'scaffold': local_train_scaffold,
    'fedoptim': local_train_fedavg,
    # 'feddyn': local_train_feddyn,
    
    # per
    'fedrep': local_train_fedrep,
    'fedper': local_train_fedper,
    'ditto': local_train_ditto,
    'pfedme': local_train_pfedme,
    # 'perfed': local_train_perfed,

}

alg2global = {
    'central': global_aggregate_central,
    'local': global_aggregate_central,
    # gen
    'fedavg': global_aggregate_fedavg,
    'fedavgm': global_aggregate_fedavg,
    'fedprox': global_aggregate_fedavg,
    'fednova': global_aggregate_fednova,
    'scaffold': global_aggregate_fedavg,
    'fedoptim': global_aggregate_fedoptim,
    # 'feddyn': global_aggregate_fedavg,
    # per
    'fedrep': global_aggregate_fedper,
    'fedper': global_aggregate_fedper,
    'ditto': global_aggregate_fedavg,
    # 'perfed': global_aggregate_fedavg,
    'pfedme': global_aggregate_pfedme,

}
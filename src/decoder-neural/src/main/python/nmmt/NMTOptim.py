import logging
import os
import copy
import torch

from nmmt.internal_utils import opts_object, log_timed_action
from onmt import Optim

class ModelFileNotFoundException(BaseException):
    def __init__(self, path):
        self.message = "Model file not found: %s" % path


class NMTOptim(object):

    class Metadata(object):

        def __init__(self):
            self.method = 'sgd'
            self.lr = 1.
            self.lr_decay = 0.9
            self.lr_decay_start = False
            self.max_grad_norm = 5

        def __str__(self):
            return str(self.__dict__)

        def __repr__(self):
            return str(self.__dict__)

        def load_from_dict(self, d):
            for key in self.__dict__:
                if key in d:
                    self.__dict__[key] = d[key]

    @staticmethod
    def new_instance(metadata=None, model=None):
        if model is None:
            raise Exception('model cannot be None')

        if metadata is None:
            metadata = NMTOptim.Metadata()

        optimizer = NMTOptim(metadata)
        optimizer.set_parameters(model.parameters())
        return optimizer

    @staticmethod
    def load_from_file(path, model=None):
        if model is None:
            raise Exception('model cannot be None')

        metadata = NMTOptim.Metadata()
        checkpoint = torch.load(path)
        if checkpoint['metadata'] is not None:
            metadata.load_from_dict(checkpoint['metadata'])

        optimizer = NMTOptim(metadata)
        optimizer.set_parameters(model.parameters())
        optimizer.optimizer.load_state_dict(checkpoint['optimizer'])
        return optimizer

    def __init__(self, metadata=None):
        self._logger = logging.getLogger('nmmt.NMTOptim')
        self._log_level = logging.INFO
        self.metadata = copy.deepcopy(metadata) if metadata is not None else NMTOptim.Metadata()

        self.optimizer = Optim(self.metadata.method,
                               self.metadata.lr,
                               max_grad_norm=self.metadata.max_grad_norm,
                               lr_decay=self.metadata.lr_decay,
                               lr_decay_start=self.metadata.lr_decay_start)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def set_parameters(self, pars):
        self.optimizer.set_parameters(pars)

    def save_to_file(self, path):
        checkpoint = {
            'metadata': self.metadata.__dict__,
            'optimizer': self.optimizer.get_state_dict()
        }
        torch.save(checkpoint, path)
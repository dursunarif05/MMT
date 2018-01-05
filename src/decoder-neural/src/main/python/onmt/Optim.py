import torch.optim as optim
from torch.nn.utils import clip_grad_norm

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm,
                 lr_decay=1, lr_decay_start=False):

        self.last_ppl = None
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_start = lr_decay_start
        self.optimizer = None

    def step(self):
        "Compute gradients norm."
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def updateLearningRate(self):
        """
        Decay learning rate
        if perplexity on validation does not improve
        or if we hit the start_decay_at limit.
        """

        if self.lr_decay_start:
            self.lr = self.lr * self.lr_decay

        self.optimizer.param_groups[0]['lr'] = self.lr

    def get_state_dict(self):
        state_dict = {k: v for k, v in self.__dict__.iteritems() if 'params' not in k}
        state_dict['optimizer'] = self.optimizer.state_dict() if self.optimizer is not None else {}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update({k: v for k, v in state_dict.iteritems() if 'optimizer' not in k})
        self.optimizer.load_state_dict(state_dict['optimizer'])

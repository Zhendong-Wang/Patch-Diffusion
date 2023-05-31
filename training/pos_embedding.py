import torch
from torch_utils import persistence

@persistence.persistent_class
class Pos_Embedding(torch.nn.Module):
    def __init__(self, num_freqs=16, input_dim=2, log_sampling=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_freq = num_freqs - 1
        self.input_dim = input_dim
        self.log_sampling = log_sampling

        if self.log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., self.max_freq, steps=self.num_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** self.max_freq, steps=self.num_freqs)

        # embed_fns = []
        # out_dim = 0
        # d = input_dim
        # for freq in freq_bands:
        #     for p_fn in self.periodic_fns:
        #         embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
        #         out_dim += d
        #
        # self.embed_fns = embed_fns
        self.out_dim = int(input_dim * self.num_freqs * 2) # 2 is for [sin, cos] function list

    def forward(self, x):
        # concatenate in the channel dim
        # assert x.shape[0] == self.input_dim
        output = []
        for freq in self.freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                output.append(p_fn(x * freq))

        return torch.cat(output, 1)
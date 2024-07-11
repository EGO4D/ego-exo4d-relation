from inference.memory_manager import MemoryManager
from model.network import XMem
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network: XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config["mem_every"]
        self.deep_update_every = config["deep_update_every"]
        self.enable_long_term = config["enable_long_term"]

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = self.deep_update_every < 0

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config["mem_every"]
        self.deep_update_every = config["deep_update_every"]
        self.enable_long_term = config["enable_long_term"]

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = self.deep_update_every < 0
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, image, ego_image, mask, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0)  # add the batch dimension
        ego_image, _ = pad_divide_by(ego_image, 16)
        ego_image = ego_image.unsqueeze(0)  # add the batch dimension
        mask, _ = pad_divide_by(mask, 16)

        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)
        need_segment = self.curr_ti > 0
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame)
            or (  # synchronized
                not self.deep_update_sync
                and self.curr_ti - self.last_deep_update_ti >= self.deep_update_every
            )  # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (
            not end
        )

        encoded_results, mx, my, out_cls = self.network.encode_key(
            ego_image, mask, image
        )

        key = encoded_results[0]["key"]
        shrinkage = encoded_results[0]["shrinkage"] if is_mem_frame else None
        selection = encoded_results[0]["selection"]
        f16 = encoded_results[0]["f16"]
        f8 = encoded_results[0]["f8"]
        f4 = encoded_results[0]["f4"]

        ego_key = encoded_results[1]["key"]
        ego_shrinkage = encoded_results[1]["shrinkage"]
        ego_selection = encoded_results[1]["selection"]
        ego_f16 = encoded_results[1]["f16"]
        ego_f8 = encoded_results[1]["f8"]
        ego_f4 = encoded_results[1]["f4"]

        multi_scale_features = (f16, f8, f4)

        pred_prob_with_bg = aggregate(mask, dim=0)
        # also create new hidden states
        self.memory.create_hidden_state(len(self.all_labels), key)

        value, hidden = self.network.encode_value(
            ego_image,
            ego_f16,
            self.memory.get_hidden(),
            pred_prob_with_bg[1:].unsqueeze(0),
        )
        self.memory.add_memory(
            ego_key, ego_shrinkage, value, self.all_labels, selection=ego_selection
        )
        self.memory.set_hidden(hidden)

        # segment the current frame is needed
        memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
        hidden, _, pred_prob_with_bg = self.network.segment(
            multi_scale_features,
            memory_readout,
            self.memory.get_hidden(),
            h_out=is_normal_update,
            strip_bg=False,
        )
        # remove batch dim
        pred_prob_with_bg = pred_prob_with_bg[0]
        pred_prob_no_bg = pred_prob_with_bg[1:]
        if is_normal_update:
            self.memory.set_hidden(hidden)

        # save as memory if needed
        if is_mem_frame:
            value, hidden = self.network.encode_value(
                image,
                f16,
                self.memory.get_hidden(),
                pred_prob_no_bg.unsqueeze(0),
                is_deep_update=is_deep_update,
            )
            self.memory.add_memory(
                key,
                shrinkage,
                value,
                self.all_labels,
                selection=selection if self.enable_long_term else None,
            )
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti

        return unpad(pred_prob_with_bg, self.pad), out_cls

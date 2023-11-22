import torch
from collections import OrderedDict
from torch.nn.utils import clip_grad_norm_

from mixofshow.data.prompt_dataset import PromptDataset
from mixofshow.utils.registry import MODEL_REGISTRY
from .finetune_model import FinetuneModel


@MODEL_REGISTRY.register()
class LorsaModel(FinetuneModel):

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.get_bare_model(self.net_g).unet.train()
        self.get_bare_model(self.net_g).text_encoder.train()

        self.optimizer_g.zero_grad()
        loss, l1, non_zeros = self.net_g(self.images, self.prompts, self.masks)
        loss_dict['loss'] = loss
        loss_dict['loss_l1'] = l1
        loss += self.opt["network_g"]["finetune_cfg"]["lambda"]*l1
        loss_dict["nonzeros_percent"] = non_zeros[0]/non_zeros[1]
        loss_dict["non_zeros"] = non_zeros[0]
        loss.backward()

        grads_text_encoder = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight.grad
        if grads_text_encoder is not None:
            index_no_updates = torch.arange(len(self.get_bare_model(self.net_g).tokenizer)) != -1
            for token_id in self.get_bare_model(self.net_g).new_concept_token_id:
                index_no_updates[token_id] = False
            grads_text_encoder.data[index_no_updates, :] = grads_text_encoder.data[index_no_updates, :].fill_(0)

        if self.opt['train'].get('max_grad_norm'):
            clip_grad_norm_(self.net_g.parameters(), max_norm=self.opt['train']['max_grad_norm'])

        self.optimizer_g.step()
        self.net_g.module.set_zeros()

        token_embeds = self.get_bare_model(self.net_g).text_encoder.get_input_embeddings().weight
        for token_id in self.get_bare_model(self.net_g).new_concept_token_id:
            loss_dict[f'Norm_{token_id}'] = token_embeds[token_id].norm()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if isinstance(dataloader.dataset, (PromptDataset)):
            if self.opt['val'].get('vis_embedding', True):
                # sample embedding
                for k, v in self.get_bare_model(self.net_g).text_encoder_lora.named_buffers():
                    if 'alpha' in k:
                        v.fill_(0)
                for k, v in self.get_bare_model(self.net_g).unet_lora.named_buffers():
                    if 'alpha' in k:
                        v.fill_(0)
                self.opt['val']['sample']['use_negative_prompt'] = True
                self.visual_validation(dataloader, f'{current_iter}_embedding_negprompt', tb_logger, save_img)
                for k, v in self.get_bare_model(self.net_g).text_encoder_lora.named_buffers():
                    if 'alpha' in k:
                        v.fill_(1)
                for k, v in self.get_bare_model(self.net_g).unet_lora.named_buffers():
                    if 'alpha' in k:
                        v.fill_(1)

            # sample negprompt
            self.opt['val']['sample']['use_negative_prompt'] = True
            self.visual_validation(dataloader, f'{current_iter}_negprompt', tb_logger, save_img)
        else:
            self.loss_validation(dataloader, current_iter, tb_logger)

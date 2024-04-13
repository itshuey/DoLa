import logging
from typing import List, Tuple, Optional, Dict
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedTokenizer, PreTrainedModel
from transformers.generation.stopping_criteria import StoppingCriteriaList, T5StoppingCriteria

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DoLa:
    def __init__(self, model_name: str, device: str = 'cuda', num_gpus: int = 1, max_gpu_memory: int = 27):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.num_gpus = num_gpus
        self.max_gpu_memory = max_gpu_memory
        self.tokenizer, self.model = self._load_model()
        self.stopping_criteria = StoppingCriteriaList()

    def _setup_device(self, device: str) -> str:
        """Configure the device for model deployment."""
        if 'cuda' in device and torch.cuda.is_available():
            return device
        return 'cpu'
    
    def _load_model(self) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Load tokenizer and model with appropriate device and memory settings."""
        kwargs = {'torch_dtype': torch.float16}
        if self.device.startswith('cuda') and self.num_gpus > 1:
            kwargs['device_map'] = 'auto'
            kwargs['max_memory'] = {i: f'{self.max_gpu_memory}GiB' for i in range(self.num_gpus)}
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model_class = AutoModelForSeq2SeqLM if 't5' in self.model_name else AutoModelForCausalLM
            model = model_class.from_pretrained(self.model_name, **kwargs)
            model.to(self.device)
            return tokenizer, model
        except Exception as e:
            logging.error(f"Failed to load the model or tokenizer: {e}")
            raise


    def set_stop_words(self, stop_words: List[str]):
        """Add stop words to the generation stopping criteria."""
        for word in stop_words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            self.stopping_criteria.add(T5StoppingCriteria(ids))
            logging.info(f"Added stop word: {word} with IDs: {ids}")

    def generate(self, input_text: str, max_new_tokens: int = 256, **kwargs) -> str:
        """Generate text based on the input using configured model."""
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        max_length = input_ids.shape[-1] + max_new_tokens
        outputs = self.model.generate(input_ids, max_length=max_length, stopping_criteria=self.stopping_criteria, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1) -> torch.BoolTensor:
        """Calculate a filter for selecting the top relative scores."""
        scores_log_softmax = scores.log_softmax(dim=-1)
        sorted_logits, _ = torch.sort(scores_log_softmax, descending=True)
        threshold = sorted_logits[..., min_tokens_to_keep-1] + torch.log(torch.tensor([relative_top]))
        return scores_log_softmax < threshold.unsqueeze(-1)

    def print_detailed_layer_stats(self, js_divergences, logits_by_layer):
        k = 5 # The number of results to show for each layer
        for token_idx in range(len(js_divergences)):
            # Print the JS Divergences for each layer
            print("JS DIVERGENCES:")
            print(js_divergences[token_idx])

            print("\nTOKEN PREDICTIONS BY LAYER")
            top_k_layer_logits = [torch.topk(l,k,sorted=True) for l in logits_by_layer[token_idx]]
            for top_k_logits in top_k_layer_logits:
                layer_str = ""
                for i, enc_token in enumerate(top_k_logits.indices[0]):
                    token = self.tokenizer.decode(enc_token, skip_special_tokens=True)
                    layer_str += (f'Token "{token}": {top_k_logits.values[0][i]}, ')
                print(layer_str[:-2])
                tokens_to_track = top_k_logits.indices[0]
            print("\nTRACKING TOKEN INDICES")
            for layer_logits in logits_by_layer[token_idx]:
                position_str = ""
                sorted_logits, sorted_tokens = torch.sort(layer_logits[0], descending=True)
                sorted_positions = {int(t): int(sorted_tokens.tolist().index(t)) for t in tokens_to_track}
                for token_id, position in sorted_positions.items():
                    token = self.tokenizer.decode(token_id, skip_special_tokens=True)
                    position_str += f'Token "{token}" Position {position}, '
                print(position_str[:-2])
            print()

    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'dola-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dola':
                print('DOLA DOLA DOLA')
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

        # print(log_probs)
        return log_probs, (premature_layer_dist if mode == 'dola' else None)
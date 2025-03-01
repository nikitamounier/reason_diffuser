import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def make_step_rewards(logits, token_masks):
    '''
    Function for Qwen2.5-Math-PRM to compute step rewards
    '''
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


@ torch.no_grad()
def generate(model, prompt, prm_model, tokenizer, prm_tokenizer, num_candidates=5, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, l).
        prm_model: Preference Reward Model for scoring generated sequences.
        tokenizer: Tokenizer for the main model.
        prm_tokenizer: Tokenizer for the PRM model.
        num_candidates: Number of candidate sequences to generate per block.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        # Generate multiple candidate sequences for this block
        candidate_sequences = []
        
        for candidate_idx in range(num_candidates):
            # Create a copy of the current sequence for this candidate
            candidate_x = x.clone()
            
            block_mask_index = (candidate_x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            
            for i in range(steps):
                mask_index = (candidate_x == mask_id)
                if cfg_scale > 0.:
                    un_x = candidate_x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([candidate_x, un_x], dim=0)
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(candidate_x).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, candidate_x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                candidate_x[transfer_index] = x0[transfer_index]
            
            # After completing all steps for this candidate, add it to our list
            candidate_sequences.append(candidate_x.clone())
        
        # Score all candidate sequences using the PRM model
        # First decode all candidate sequences
        decoded_candidates = []
        for candidate_x in candidate_sequences:
            # Decode only the current block for each candidate
            block_text = tokenizer.decode(
                candidate_x[0, prompt.shape[1] + num_block * block_length:prompt.shape[1] + (num_block + 1) * block_length], 
                skip_special_tokens=True
            )
            # Split the text into steps (assuming double line breaks separate steps)
            steps_text = block_text.split("\n\n")
            # Join with the special token as required by PRM
            formatted_text = "<extra_0>".join(steps_text) + "<extra_0>"
            decoded_candidates.append(formatted_text)
        
        # Create system and query for PRM context
        query = tokenizer.decode(prompt[0], skip_special_tokens=True)
        system = "Please reason step by step, and put your final answer within \\boxed{}."
        
        # Prepare all candidates for batch scoring
        all_prm_input_ids = []
        for candidate_text in decoded_candidates:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
                {"role": "assistant", "content": candidate_text},
            ]
            conversation_str = prm_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            input_ids = prm_tokenizer.encode(
                conversation_str, 
                return_tensors="pt", 
            ).to(prm_model.device)
            
            all_prm_input_ids.append(input_ids)
        
        # Calculate scores for each candidate
        scores = []
        for input_ids in all_prm_input_ids:
            outputs = prm_model(input_ids=input_ids)
            
            step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
            token_masks = (input_ids == step_sep_id)
            step_reward = make_step_rewards(outputs[0], token_masks)
            
            # Use the average score of all steps as the candidate's score
            if step_reward[0]:  # Check if there are any scores
                avg_score = sum(step_reward[0]) / len(step_reward[0])
                scores.append(avg_score)
            else:
                scores.append(0.0)  # If no scores, assign 0
        
        # Select the candidate with the highest score
        if scores:  # Make sure we have scores
            best_candidate_idx = np.argmax(scores)
            x = candidate_sequences[best_candidate_idx]
        else:
            # If no valid scores, just take the first candidate
            x = candidate_sequences[0]

    return x


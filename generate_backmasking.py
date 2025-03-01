import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def make_step_rewards(logits, token_masks):
    """
    Function for Qwen2.5-Math-PRM to compute step rewards
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[
            :, 1
        ]  # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def calculate_backmasking_probs(block_scores, backmasking_alpha=5.0, min_prob=0.01):
    """
    Calculate backmasking probabilities for each block based on PRM scores.
    We use an exponential function to create a steeper curve that emphasizes poor scores more.
    
    Args:
        block_scores: List of PRM scores for each block
        backmasking_alpha: Controls the steepness of the exponential decay
        min_prob: Minimum probability of backmasking for any block
        
    Returns:
        Array of backmasking probabilities for each block
    """
    # Normalize scores to [0, 1]
    if not block_scores:
        return []
    
    # Convert to numpy array for easier manipulation
    scores = np.array(block_scores)
    
    # Apply exponential transformation
    probs = np.exp(-backmasking_alpha * scores)
    
    # Ensure minimum probability
    probs = np.maximum(probs, min_prob)
    
    # Normalize to [min_prob, 1]
    probs = min_prob + (1 - min_prob) * (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
    
    return probs


def get_backmasking_tokens(block_indices, block_probs, prompt_length, block_length, backmasking_intensity=0.5):
    """
    Determine which tokens to backmark based on block probabilities.
    
    Args:
        block_indices: Indices of each block
        block_probs: Probability of backmasking for each block
        prompt_length: Length of the prompt
        block_length: Length of each block
        backmasking_intensity: Overall intensity of backmasking
        
    Returns:
        Boolean mask where True indicates tokens to be masked
    """
    mask = torch.zeros_like(block_indices[0], dtype=torch.bool)
    
    for i, (indices, prob) in enumerate(zip(block_indices, block_probs)):
        # Calculate number of tokens to mask in this block
        block_size = indices.sum().item()
        num_to_mask = int(block_size * prob * backmasking_intensity)
        
        if num_to_mask > 0:
            # Get the block range
            start_idx = prompt_length + i * block_length
            end_idx = start_idx + block_length
            
            # Select random positions within the block to mask
            block_positions = torch.where(block_indices[i])[1]
            if len(block_positions) > 0:  # Ensure there are tokens to mask
                mask_positions = block_positions[torch.randperm(len(block_positions))[:num_to_mask]]
                mask[0, mask_positions] = True
    
    return mask


def compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer):
    """
    Compute the PRM score for a single block of text.
    
    Args:
        block_text: The text of the current block
        prompt_text: The original prompt text
        prm_model: The PRM model
        prm_tokenizer: The PRM tokenizer
        
    Returns:
        The PRM score for this block
    """
    # Split the text into steps (assuming double line breaks separate steps)
    steps_text = block_text.split("\n\n")
    
    # Join with the special token as required by PRM
    formatted_text = "<extra_0>".join(steps_text) + "<extra_0>"
    
    # Create the system prompt
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    
    # Create a conversation
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": formatted_text},
    ]
    
    conversation_str = prm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    input_ids = prm_tokenizer.encode(
        conversation_str,
        return_tensors="pt",
    ).to(prm_model.device)
    
    outputs = prm_model(input_ids=input_ids)
    
    # Find tokens that match the special token for step separation
    step_sep_id = prm_tokenizer.encode("<extra_0>")[0]
    token_masks = input_ids == step_sep_id
    
    # Compute rewards
    step_reward = make_step_rewards(outputs[0], token_masks)
    
    # Average score
    if step_reward[0]:  # Check if there are any scores
        avg_score = sum(step_reward[0]) / len(step_reward[0])
        return avg_score
    else:
        return 0.0  # If no scores, assign 0


def recompute_all_block_scores(x, prompt, prompt_text, tokenizer, prm_model, prm_tokenizer, block_length):
    """
    Recompute PRM scores for all blocks after backmasking and demasking.
    
    Args:
        x: The current token sequence
        prompt: The original prompt tokens
        prompt_text: The decoded prompt text
        tokenizer: The main tokenizer
        prm_model: The PRM model
        prm_tokenizer: The PRM tokenizer
        block_length: Length of each block
        
    Returns:
        List of updated PRM scores for each block
    """
    num_blocks = (x.shape[1] - prompt.shape[1]) // block_length
    updated_scores = []
    
    for block_idx in range(num_blocks):
        block_text = tokenizer.decode(
            x[
                0,
                prompt.shape[1]
                + block_idx * block_length : prompt.shape[1]
                + (block_idx + 1) * block_length,
            ],
            skip_special_tokens=True,
        )
        
        block_score = compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer)
        updated_scores.append(block_score)
    
    return updated_scores


@torch.no_grad()
def generate(
    model,
    prompt,
    prm_model,
    tokenizer,
    prm_tokenizer,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    backmasking_alpha=5.0,
    backmasking_intensity=0.5,
    global_demasking=True,
    backmasking_frequency=3,  # Apply backmasking every N blocks
    backmasking_threshold=0.4,  # Apply backmasking if any block scores below this threshold
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, l).
        prm_model: Preference Reward Model for scoring generated sequences.
        tokenizer: Tokenizer for the main model.
        prm_tokenizer: Tokenizer for the PRM model.
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        backmasking_alpha: Controls the steepness of the exponential decay for backmasking probabilities.
        backmasking_intensity: Overall intensity of backmasking (0-1).
        global_demasking: Whether to demask the entire sequence in one go after backmasking (True) or block by block (False).
        backmasking_frequency: Apply backmasking every N blocks (if set to 1, applies after every block).
        backmasking_threshold: Apply backmasking if any block scores below this threshold.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id
    prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    block_steps = steps // num_blocks
    
    # Store PRM scores for each block
    block_scores = []
    # Store block indices for backmasking
    block_indices = []

    for num_block in range(num_blocks):
        # Process one block at a time
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps)

        # Progressively unmask the current block
        for i in range(block_steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Only consider tokens in the current block
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(
                x0, dtype=torch.bool, device=x0.device
            )
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=num_transfer_tokens[j, i]
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

        # After completing this block, compute its PRM score
        current_block_text = tokenizer.decode(
            x[
                0,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ],
            skip_special_tokens=True,
        )
        
        # Compute score for this block
        block_score = compute_block_score(current_block_text, prompt_text, prm_model, prm_tokenizer)
        block_scores.append(block_score)
        
        # Store block indices for this block (used for backmasking later)
        current_block_indices = torch.zeros_like(x, dtype=torch.bool)
        current_block_indices[
            :,
            prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length,
        ] = True
        block_indices.append(current_block_indices)
        
        # Apply backmasking only when:
        # 1. We have enough blocks (more than 1)
        # 2. Either we've reached the backmasking frequency OR a block score is below threshold
        should_backmask = len(block_scores) > 1 and (
            (num_block + 1) % backmasking_frequency == 0 or  # Every N blocks
            block_scores[-1] < backmasking_threshold  # Latest block is below threshold
        )
        
        if should_backmask:
            print(f"Applying backmasking after block {num_block+1}. Block scores: {block_scores}")
            # Calculate backmasking probabilities for all blocks based on their scores
            backmasking_probs = calculate_backmasking_probs(block_scores, backmasking_alpha)
            
            # Determine which tokens to backmask
            mask = get_backmasking_tokens(
                block_indices, 
                backmasking_probs, 
                prompt.shape[1], 
                block_length, 
                backmasking_intensity
            )
            
            # Apply the mask
            x[mask] = mask_id
            
            # Now demask the sequence again
            if global_demasking:
                # Global demasking: demask the entire sequence at once
                mask_index = x == mask_id
                
                # Number of total masked tokens
                total_masked = mask_index.sum().item()
                global_steps = min(steps // 2, total_masked)  # Use fewer steps for efficiency
                
                if global_steps > 0:
                    # Calculate how many tokens to unmask per step
                    num_transfer_tokens = get_num_transfer_tokens(mask_index, global_steps)
                    
                    for i in range(global_steps):
                        # Update mask index
                        mask_index = x == mask_id
                        
                        if cfg_scale > 0.0:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = model(x).logits
                        
                        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                        x0 = torch.argmax(logits_with_noise, dim=-1)
                        
                        if remasking == "low_confidence":
                            p = F.softmax(logits.to(torch.float64), dim=-1)
                            x0_p = torch.squeeze(
                                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                            )
                        elif remasking == "random":
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                        else:
                            raise NotImplementedError(remasking)
                        
                        # Only consider tokens up to the current block
                        x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf
                        
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)
                        
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            _, select_index = torch.topk(
                                confidence[j], k=num_transfer_tokens[j, i]
                            )
                            transfer_index[j, select_index] = True
                        x[transfer_index] = x0[transfer_index]
                    
                    # After global demasking, recompute all block scores
                    block_scores = recompute_all_block_scores(
                        x, prompt, prompt_text, tokenizer, prm_model, prm_tokenizer, block_length
                    )
                    print(f"Updated block scores after demasking: {block_scores}")
            else:
                # Block-by-block demasking: demask each block separately
                for block_idx in range(num_block + 1):
                    block_mask_index = (
                        x[
                            :,
                            prompt.shape[1]
                            + block_idx * block_length : prompt.shape[1]
                            + (block_idx + 1) * block_length,
                        ]
                        == mask_id
                    )
                    
                    if block_mask_index.sum() > 0:  # If there are tokens to unmask in this block
                        block_steps_to_use = min(block_steps, block_mask_index.sum().item())
                        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, block_steps_to_use)
                        
                        for i in range(block_steps_to_use):
                            # Update mask index for the whole sequence
                            mask_index = x == mask_id
                            
                            if cfg_scale > 0.0:
                                un_x = x.clone()
                                un_x[prompt_index] = mask_id
                                x_ = torch.cat([x, un_x], dim=0)
                                logits = model(x_).logits
                                logits, un_logits = torch.chunk(logits, 2, dim=0)
                                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                            else:
                                logits = model(x).logits
                            
                            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                            x0 = torch.argmax(logits_with_noise, dim=-1)
                            
                            if remasking == "low_confidence":
                                p = F.softmax(logits.to(torch.float64), dim=-1)
                                x0_p = torch.squeeze(
                                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                                )
                            elif remasking == "random":
                                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                            else:
                                raise NotImplementedError(remasking)
                            
                            # Only consider tokens in the current block
                            block_start = prompt.shape[1] + block_idx * block_length
                            block_end = prompt.shape[1] + (block_idx + 1) * block_length
                            
                            # Set confidence to -inf for tokens outside the current block
                            block_confidence_mask = torch.zeros_like(x0, dtype=torch.bool)
                            block_confidence_mask[:, block_start:block_end] = True
                            
                            x0 = torch.where(mask_index, x0, x)
                            confidence = torch.where(mask_index & block_confidence_mask, x0_p, -np.inf)
                            
                            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                            for j in range(confidence.shape[0]):
                                _, select_index = torch.topk(
                                    confidence[j], k=num_transfer_tokens[j, i]
                                )
                                transfer_index[j, select_index] = True
                            x[transfer_index] = x0[transfer_index]
                
                # After block-by-block demasking, recompute all block scores
                block_scores = recompute_all_block_scores(
                    x, prompt, prompt_text, tokenizer, prm_model, prm_tokenizer, block_length
                )
                print(f"Updated block scores after demasking: {block_scores}")

    return x

import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM


import functools
print = functools.partial(print, flush=True)

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
    probs = min_prob + (1 - min_prob) * (probs - probs.min()) / (
        probs.max() - probs.min() + 1e-8
    )

    return probs


def get_backmasking_tokens(
    block_indices, block_probs, prompt_length, block_length, backmasking_intensity=0.5
):
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
                mask_positions = block_positions[
                    torch.randperm(len(block_positions))[:num_to_mask]
                ]
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
    # steps_text = block_text.split("\n\n")

    # Join with the special token as required by PRM
    formatted_text = "<extra_0>" + block_text + "<extra_0>"

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

    batch = prm_tokenizer(
        conversation_str,
        return_tensors="pt",
        padding=True,         # optional; safe to include
        truncation=True       # optional; good for long inputs
    )

    input_ids = batch["input_ids"].to(prm_model.device)

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


def recompute_all_block_scores(
    x, prompt, prompt_text, tokenizer, prm_model, prm_tokenizer, block_length
):
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

        block_score = compute_block_score(
            block_text, prompt_text, prm_model, prm_tokenizer
        )
        updated_scores.append(block_score)

    return updated_scores


def demask_steps(
    x,               
    ctx_end,          
    steps,          
    temperature,
    model,   
    cfg_scale,       
    remasking,      
    mask_id,        
    get_num_transfer
):
    """
    Repeatedly unmask `steps` times, each time picking top‑k masked positions by confidence.
    Modifies x in place and returns it.
    """
    for i in range(steps):
        # 1) figure out which positions are still masked
        mask_index = x == mask_id

        # 2) get model logits (with CFG if requested)
        if cfg_scale > 0:
            # … your cfg logic …
            un_x = x.clone()
            un_x[x != mask_id] = mask_id
            x_in = torch.cat([x, un_x], dim=0)
            logits = model(x_in).logits
            logits, un_logits = logits.chunk(2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x[:, :ctx_end]).logits

        # 3) add Gumbel noise & pick the argmax
        noisy = add_gumbel_noise(logits, temperature)
        x0    = noisy.argmax(dim=-1)

        # 4) compute “confidence” per mask
        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        else:  # random
            x0_p = torch.rand_like(x0, dtype=torch.float64)

        # 5) don’t unmask anything beyond ctx_end + current block
        x0_p[:, ctx_end:] = -float("inf")

        # 6) pick top‑k indices to transfer this step
        num_to_transfer = get_num_transfer(mask_index, steps)[0, i].item()
        _, to_unmask = x0_p[0].topk(num_to_transfer)
        x[0, to_unmask] = x0[0, to_unmask]

    return x


@torch.no_grad()
def generate(
    model,
    prompt,
    prm_model,
    tokenizer,
    prm_tokenizer,
    steps=128,
    gen_length=512,
    block_length=32,
    temperature=0.3,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    backmasking_alpha=5.0,
    backmasking_intensity=0.5,
    global_demasking=True,
    backmasking_frequency=3,
    backmasking_threshold=0.4,
    max_retry_attempts=5,
):
    print("===== Generation Started =====")
    print(f"Prompt (first 100 chars): {tokenizer.decode(prompt[0], skip_special_tokens=True)[:100]}")
    print(f"Steps: {steps}, Gen length: {gen_length}, Block length: {block_length}")
    print(f"Temperature: {temperature}, CFG scale: {cfg_scale}")
    print(f"Backmasking alpha: {backmasking_alpha}, intensity: {backmasking_intensity}")
    print(f"Global demasking: {global_demasking}, Frequency: {backmasking_frequency}, Threshold: {backmasking_threshold}")
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    K = backmasking_frequency
    prompt_index = x != mask_id
    prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    block_steps = steps // num_blocks

    block_scores = []
    block_indices = []

    for num_block in range(num_blocks):
        print(f"\n--- Processing Block {num_block+1}/{num_blocks} ---")
        best_block_score = 0.0
        best_block_tokens = None
        retry_count = 0
        meets_threshold = False

        while retry_count < max_retry_attempts and not meets_threshold and (num_block+1) % K == 0:
            if retry_count > 0:
                print(f"Retry {retry_count}/{max_retry_attempts} for block {num_block+1}")
                x[:, prompt.shape[1] + num_block*block_length : prompt.shape[1] + (num_block+1)*block_length] = mask_id

            ctx_end = prompt.shape[1] + (num_block+1)*block_length
            demask_steps(
                x,
                ctx_end=ctx_end,
                steps=block_steps,
                temperature=temperature,
                model=model,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
                get_num_transfer=get_num_transfer_tokens,
            )

            # compute PRM score
            block_text = tokenizer.decode(
                x[0, prompt.shape[1] + num_block*block_length : prompt.shape[1] + (num_block+1)*block_length],
                skip_special_tokens=True
            )
            print(f"Block {num_block+1} decoded text (first 50 chars): {block_text[:50]}...")
            block_score = compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer)
            print(f"Block {num_block+1} PRM Score: {block_score:.4f}")

            if block_score > best_block_score:
                best_block_score = block_score
                best_block_tokens = x[:, prompt.shape[1]+num_block*block_length : prompt.shape[1]+(num_block+1)*block_length].clone()

            if block_score >= backmasking_threshold:
                meets_threshold = True
                print(f"Block {num_block+1} passed threshold.\n")
            else:
                retry_count += 1

        if retry_count and not meets_threshold:
            print(f"Using best attempt for block {num_block+1} with score {best_block_score:.4f}")
            x[:, prompt.shape[1]+num_block*block_length : prompt.shape[1]+(num_block+1)*block_length] = best_block_tokens
            block_score = best_block_score

        block_scores.append(block_score)
        idx = torch.zeros_like(x, dtype=torch.bool)
        idx[:, prompt.shape[1]+num_block*block_length : prompt.shape[1]+(num_block+1)*block_length] = True
        block_indices.append(idx)

        if len(block_scores) >= K and min(block_scores[-K:]) < backmasking_threshold and (num_block+1) % K == 0:
            print(f"\n*** Backmasking triggered after block {num_block+1} ***")
            start = len(block_indices) - K
            recent_indices = block_indices[start:]
            recent_scores = calculate_backmasking_probs(block_scores, backmasking_alpha)[start:]
            mask = get_backmasking_tokens(
                recent_indices,
                recent_scores,
                prompt.shape[1],
                block_length,
                backmasking_intensity,
            )
            masked = mask.sum().item()
            x[mask] = mask_id
            print(f"Backmasked {masked} tokens.")

            ctx_end = prompt.shape[1] + (num_block+1)*block_length
            masked_count = (x == mask_id).sum().item()
            global_steps = min(steps//2, masked_count)
            print(f"Demasking {masked_count} masked tokens over {global_steps} steps.")

            if global_steps:
                if global_demasking:
                    print(f"Running global demasking...")
                    demask_steps(
                        x, ctx_end,
                        steps=global_steps,
                        temperature=temperature,
                        model=model,
                        cfg_scale=cfg_scale,
                        remasking=remasking,
                        mask_id=mask_id,
                        get_num_transfer=get_num_transfer_tokens,
                    )
                else:
                    print(f"Running block-by-block demasking...")
                    for b in range(num_block+1):
                        b_ctx = prompt.shape[1] + (b+1)*block_length
                        masked_in_block = get_num_transfer_tokens(x[:, :b_ctx]==mask_id, 1)[0,0].item()
                        if masked_in_block > 0:
                            print(f"Demasking block {b+1}, {masked_in_block} masked tokens...")
                            demask_steps(
                                x, b_ctx,
                                steps=masked_in_block,
                                temperature=temperature,
                                cfg_scale=cfg_scale,
                                model=model,
                                remasking=remasking,
                                mask_id=mask_id,
                                get_num_transfer=get_num_transfer_tokens,
                            )

    print("\n===== Generation Complete =====")
    print(f"Final block scores: {[f'{s:.3f}' for s in block_scores]}")
    decoded = tokenizer.decode(x[0, prompt.shape[1]:], skip_special_tokens=True)
    print(f"Generated output:\n{decoded[:500]}\n{'...' if len(decoded) > 500 else ''}")
    return x

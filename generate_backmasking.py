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


# --- Modified make_step_rewards to handle batches (from previous example) ---
def make_step_rewards_batched(logits, token_masks):
    """
    Function for Qwen2.5-Math-PRM to compute step rewards for a batch.
    Args:
        logits: Tensor of shape (batch_size, seq_len, num_labels)
        token_masks: Tensor of shape (batch_size, seq_len)
    Returns:
        List of lists, where each inner list contains scores for one sample in the batch.
    """
    probabilities = F.softmax(logits, dim=-1)
    # Ensure token_masks is bool or float for multiplication
    token_masks_float = token_masks.unsqueeze(-1).to(probabilities.dtype)
    probabilities = probabilities * token_masks_float # bs, seq_len, num_labels

    all_scores_res = []
    # Iterate through each sample in the batch
    for i in range(probabilities.size(0)):
        sample_probs = probabilities[i] # seq_len, num_labels
        sample_mask = token_masks[i]    # seq_len

        # Find non-zero probability entries corresponding to mask tokens
        # Assuming the relevant logits are at masked positions and have dim 2 (pos/neg)
        masked_probs = sample_probs[sample_mask] # num_masked_tokens, num_labels

        # Check if masked_probs is not empty and has the expected last dimension size (e.g., 2)
        if masked_probs.numel() > 0 and masked_probs.shape[-1] == 2:
             # Assuming positive score is at index 1
            positive_probs = masked_probs[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        elif masked_probs.numel() > 0 and masked_probs.shape[-1] != 2:
             # Fallback or warning if the shape is unexpected but not empty
             print(f"Warning: Unexpected shape for masked_probs: {masked_probs.shape}. Using sum as score.")
             # Example fallback: just sum probabilities if shape is wrong
             all_scores_res.append([masked_probs.sum().item()]) # Or handle as error
        else:
            # Handle cases with no masked tokens found
            all_scores_res.append([])
    return all_scores_res


# --- Helper to compute product score for a full sequence ---
def compute_full_sequence_product_score(
    sequence_tensor, # Shape (1, seq_len)
    prompt_len,
    num_completed_blocks,
    block_length,
    tokenizer,
    prompt_text,
    prm_model,
    prm_tokenizer,
    score_epsilon=1e-9 # Small value to avoid multiplying by zero
):
    """Computes the product of PRM scores for all completed blocks in a sequence."""
    product_score = 1.0
    for block_idx in range(num_completed_blocks):
        start_idx = prompt_len + block_idx * block_length
        end_idx = start_idx + block_length
        block_text = tokenizer.decode(
            sequence_tensor[0, start_idx:end_idx],
            skip_special_tokens=True
        )
        block_score = compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer)
        # Multiply by score, ensuring it's not zero or negative
        product_score *= max(block_score, score_epsilon)
    return product_score


@torch.no_grad()
def generate( # Renamed function
    model,
    prompt,
    prm_model,
    tokenizer,
    prm_tokenizer,
    # --- Generation Params ---
    steps=128,
    gen_length=512,
    block_length=32,
    temperature=0.6,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    # --- Parallel Block Candidate Params ---
    num_candidates_per_block=5, # k: Number of parallel candidates per block
    # --- Backmasking Params ---
    backmasking_lookback=3,      # n: Lookback window for triggering backmasking
    backmasking_min_score_threshold=0.9, # Threshold for min score in lookback window
    backmasking_alpha=5.0,
    backmasking_intensity=0.5,
    # --- Parallel Demasking Params ---
    num_demasking_candidates=5, # d: Number of parallel demasking attempts
    # --- Technical Params ---
    prm_batch_size=None # Optional: Limit PRM batch size if it causes OOM
):
    """
    Generates text using parallel candidates per block and parallel demasking after
    backmasking triggered by low scores in a lookback window.

    Args:
        k (num_candidates_per_block): Generate k candidates per block in parallel.
        n (backmasking_lookback): Check min score over last n blocks.
        backmasking_min_score_threshold: Trigger backmasking if min score < threshold.
        d (num_demasking_candidates): Generate d candidates in parallel during demasking.
        (Other args similar to previous versions)
    """
    device = model.device
    print(f"Starting parallel generation & backmasking:")
    print(f"  - Block Candidates (k): {num_candidates_per_block}")
    print(f"  - Backmasking Lookback (n): {backmasking_lookback}, Threshold: {backmasking_min_score_threshold}")
    print(f"  - Demasking Candidates (d): {num_demasking_candidates}")
    print(f"  - Steps: {steps}, Gen Length: {gen_length}, Block Length: {block_length}")
    print(f"  - Temp: {temperature}, CFG Scale: {cfg_scale}, Remasking: {remasking}")

    # Initial sequence (batch size 1)
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]
    prompt_index_base = torch.zeros(1, prompt_len + gen_length, dtype=torch.bool, device=device)
    prompt_index_base[:, :prompt_len] = True
    prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)


    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    assert steps >= num_blocks, "Total steps must be at least the number of blocks"
    steps_per_block = max(1, steps // num_blocks) # Ensure at least 1 step per block

    print(f"Num Blocks: {num_blocks}, Steps per Block: {steps_per_block}")

    block_scores = [] # Store scores of the *selected* best block from candidates
    block_indices = [] # Store indices corresponding to the blocks in 'x'

    actual_prm_batch_size = prm_batch_size if prm_batch_size else max(num_candidates_per_block, num_demasking_candidates)

    # ========================
    # Main Generation Loop
    # ========================
    for num_block in range(num_blocks):
        print(f"\n{'='*50}")
        print(f"Processing Block {num_block+1}/{num_blocks}")
        print(f"{'='*50}")
        start_block_idx = prompt_len + num_block * block_length
        end_block_idx = start_block_idx + block_length

        # ----------------------------------------
        # 1. Parallel Candidate Generation for Current Block
        # ----------------------------------------
        print(f"Generating {num_candidates_per_block} candidates for block {num_block+1}...")
        # Expand the current best sequence 'x' into a batch for candidates
        candidate_batch_x = x.repeat(num_candidates_per_block, 1) # Shape: (k, seq_len)
        prompt_index = prompt_index_base.repeat(num_candidates_per_block, 1) # Shape: (k, seq_len)

        # Get mask for the *current block* across the batch
        block_mask_index_batch = torch.zeros_like(candidate_batch_x, dtype=torch.bool)
        block_mask_index_batch[:, start_block_idx:end_block_idx] = (
            candidate_batch_x[:, start_block_idx:end_block_idx] == mask_id
        )

        # Calculate tokens to transfer per step for the batch (based on current block mask)
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index_batch[:, start_block_idx:end_block_idx], # Pass only the block's mask portion
            steps_per_block
            ) # Shape: (k, steps_per_block)

        # Iteratively decode the current block for all k candidates in parallel
        for i in range(steps_per_block):
            mask_index = candidate_batch_x == mask_id # Full mask for confidence calc: (k, seq_len)

            # --- Batched Model Inference ---
            if cfg_scale > 0.0:
                un_x = candidate_batch_x.clone()
                un_x[prompt_index] = mask_id # Mask prompt tokens
                x_ = torch.cat([candidate_batch_x, un_x], dim=0) # Shape: (2k, seq_len)
                all_logits = model(x_).logits # Shape: (2k, seq_len, vocab_size)
                logits, un_logits = torch.chunk(all_logits, 2, dim=0) # Each: (k, seq_len, vocab_size)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits) # Apply CFG
            else:
                logits = model(candidate_batch_x).logits # Shape: (k, seq_len, vocab_size)

            # --- Batched Sampling & Confidence ---
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # Predicted tokens: (k, seq_len)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1) # (k, seq_len, vocab_size)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1) # (k, seq_len)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64) # (k, seq_len)
            else:
                raise NotImplementedError(remasking)

            # Ignore confidence outside the current block for selection
            confidence_block_mask = torch.full_like(x0_p, -np.inf)
            confidence_block_mask[:, start_block_idx:end_block_idx] = 0.0 # Allow confidence here
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence += confidence_block_mask # Zeros out confidence outside the block

            x0_applied = torch.where(mask_index, x0, candidate_batch_x)

            # --- Batched Top-K Selection (within current block) ---
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(num_candidates_per_block): # Loop over k candidates
                k_tokens = num_transfer_tokens[j, i].item()
                if k_tokens > 0:
                    # Use confidence calculated *within the current block*
                    confidence_in_block = confidence[j, start_block_idx:end_block_idx]
                    # Ensure we don't request more tokens than available/masked in the block
                    num_masked_in_block = (mask_index[j, start_block_idx:end_block_idx]).sum().item()
                    actual_k = min(k_tokens, confidence_in_block.numel(), num_masked_in_block)
                    if actual_k > 0:
                        _, select_indices_local = torch.topk(confidence_in_block, k=actual_k)
                        select_indices_global = select_indices_local + start_block_idx
                        transfer_index[j, select_indices_global] = True

            candidate_batch_x = torch.where(transfer_index, x0_applied, candidate_batch_x)
            # Debug print (optional)
            # print(f"  Block {num_block+1}, Step {i+1}: Unmasked {transfer_index.sum().item()} tokens across candidates")


        # ----------------------------------------
        # 2. Score Candidates and Select Best Block
        # ----------------------------------------
        print(f"Scoring {num_candidates_per_block} candidates for block {num_block+1}...")
        candidate_block_scores = []

        # Process candidates in mini-batches for PRM if needed (memory)
        for batch_start in range(0, num_candidates_per_block, actual_prm_batch_size):
            batch_end = min(batch_start + actual_prm_batch_size, num_candidates_per_block)
            current_bs = batch_end - batch_start
            current_candidate_sub_batch = candidate_batch_x[batch_start:batch_end] # (current_bs, seq_len)

            # Decode the *current block* for the mini-batch
            decoded_blocks = []
            for cand_idx in range(current_bs):
                block_text = tokenizer.decode(
                    current_candidate_sub_batch[cand_idx, start_block_idx:end_block_idx],
                    skip_special_tokens=True,
                )
                # PRM Formatting (adjust if needed for specific PRM)
                steps_text = block_text.split("\n\n")
                formatted_text = "<extra_0>".join(steps_text) + "<extra_0>"
                decoded_blocks.append(formatted_text)

            # Prepare PRM inputs for the mini-batch
            prm_input_list = []
            system = "Please reason step by step, and put your final answer within \\boxed{}." # Example system prompt

            for candidate_block_text in decoded_blocks:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt_text}, # Use decoded prompt
                    {"role": "assistant", "content": candidate_block_text},
                ]
                tokenized = prm_tokenizer(
                     prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
                     return_tensors=None, add_special_tokens=True
                     )
                prm_input_list.append(tokenized)

            # Pad the batch
            prm_inputs_padded = prm_tokenizer.pad(
                prm_input_list, padding=True, return_tensors="pt"
            ).to(prm_model.device)

            # --- Batched PRM Inference ---
            prm_outputs = prm_model(**prm_inputs_padded)

            # Find special token masks
            step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
            prm_token_masks = prm_inputs_padded['input_ids'] == step_sep_id

            # Calculate step rewards for the batch
            batch_step_rewards = make_step_rewards_batched(prm_outputs[0], prm_token_masks)

            # Calculate average score for each candidate in the mini-batch
            for step_reward_list in batch_step_rewards:
                if step_reward_list:
                    avg_score = sum(step_reward_list) / len(step_reward_list)
                    candidate_block_scores.append(avg_score)
                else:
                    candidate_block_scores.append(float('-inf')) # Assign low score if no rewards

        # --- Select Best Candidate Block ---
        if not candidate_block_scores or max(candidate_block_scores) == float('-inf'):
             print("Warning: No valid scores generated for any candidate block. Selecting candidate 0.")
             best_candidate_idx = 0
             selected_block_score = 0.0 # Assign a default score
        else:
            best_candidate_idx = np.argmax(candidate_block_scores)
            selected_block_score = candidate_block_scores[best_candidate_idx]
            print(f"Selected candidate {best_candidate_idx} for block {num_block+1} with score: {selected_block_score:.4f}")

        # Update the main sequence 'x' with the best candidate's *full sequence*
        x = candidate_batch_x[best_candidate_idx].unsqueeze(0) # Back to shape (1, seq_len)

        # Store the score and indices of the selected block
        block_scores.append(selected_block_score)
        current_block_indices = torch.zeros_like(x, dtype=torch.bool) # Shape (1, seq_len)
        current_block_indices[:, start_block_idx:end_block_idx] = True
        block_indices.append(current_block_indices)


        # ----------------------------------------
        # 3. Check for Backmasking Trigger
        # ----------------------------------------
        should_backmask = False
        if num_block + 1 >= backmasking_lookback:
            min_score_in_window = min(block_scores[-backmasking_lookback:])
            print(f"Min score in last {backmasking_lookback} blocks: {min_score_in_window:.4f}")
            if min_score_in_window < backmasking_min_score_threshold:
                should_backmask = True
                print(f"Triggering backmasking (min score {min_score_in_window:.4f} < {backmasking_min_score_threshold:.4f})")
            else:
                 print(f"No backmasking needed (min score {min_score_in_window:.4f} >= {backmasking_min_score_threshold:.4f})")

        # ----------------------------------------
        # 4. Apply Backmasking & Parallel Demasking (if triggered)
        # ----------------------------------------
        if should_backmask:
            print(f"\n{'*'*50}")
            print(f"Applying backmasking based on scores: {[f'{s:.4f}' for s in block_scores]}")
            print(f"{'*'*50}")

            # Calculate probs based on *all* current block scores
            backmasking_probs = calculate_backmasking_probs(block_scores, backmasking_alpha)
            print(f"Backmasking probabilities: {[f'{p:.4f}' for p in backmasking_probs]}")

            # Determine tokens to mask across relevant blocks
            # Pass only the block_indices generated so far
            mask = get_backmasking_tokens(
                block_indices, # List of masks, one per block
                backmasking_probs,
                prompt_len,
                block_length,
                backmasking_intensity
            ) # mask shape: (1, seq_len)

            # Apply the mask
            x_masked = x.clone()
            tokens_to_backmask = mask.sum().item()
            if tokens_to_backmask > 0:
                x_masked[mask] = mask_id
                print(f"Backmasked {tokens_to_backmask} tokens.")

                # --- Parallel Demasking ---
                print(f"\nStarting parallel demasking with {num_demasking_candidates} candidates...")
                # Expand the masked sequence into a batch for demasking candidates
                demasking_candidate_batch = x_masked.repeat(num_demasking_candidates, 1) # Shape: (d, seq_len)
                demasking_prompt_index = prompt_index_base.repeat(num_demasking_candidates, 1) # Shape: (d, seq_len)

                # Demasking loop (similar to generation, but on the whole masked sequence)
                demasking_mask_index = demasking_candidate_batch == mask_id
                total_masked = demasking_mask_index[0].sum().item() # Count masks in one sample
                # Use fewer steps for efficiency, proportional to block steps maybe?
                demasking_steps = max(1, min(steps_per_block * backmasking_lookback, total_masked)) # Heuristic for steps
                print(f"Total masked tokens: {total_masked}, using {demasking_steps} demasking steps.")

                if demasking_steps > 0 and total_masked > 0:
                     # Calculate transfer tokens for the full mask across the batch
                    num_demasking_transfer_tokens = get_num_transfer_tokens(
                        demasking_mask_index, # Pass the full mask index for the batch
                        demasking_steps
                    ) # Shape: (d, demasking_steps)

                    for i_de in range(demasking_steps):
                        current_demask_mask_index = demasking_candidate_batch == mask_id # (d, seq_len)

                        # --- Batched Model Inference ---
                        if cfg_scale > 0.0:
                            un_x_de = demasking_candidate_batch.clone()
                            un_x_de[demasking_prompt_index] = mask_id
                            x_de_ = torch.cat([demasking_candidate_batch, un_x_de], dim=0) # (2d, seq_len)
                            all_logits_de = model(x_de_).logits
                            logits_de, un_logits_de = torch.chunk(all_logits_de, 2, dim=0) # (d, ...)
                            logits_de = un_logits_de + (cfg_scale + 1) * (logits_de - un_logits_de)
                        else:
                            logits_de = model(demasking_candidate_batch).logits # (d, ...)

                        # --- Batched Sampling & Confidence ---
                        logits_de_noise = add_gumbel_noise(logits_de, temperature=temperature)
                        x0_de = torch.argmax(logits_de_noise, dim=-1) # (d, seq_len)

                        if remasking == "low_confidence":
                            p_de = F.softmax(logits_de.to(torch.float64), dim=-1)
                            x0_p_de = torch.gather(p_de, dim=-1, index=x0_de.unsqueeze(-1)).squeeze(-1)
                        elif remasking == "random":
                            x0_p_de = torch.rand_like(x0_de, dtype=torch.float64)
                        else: # Should not happen based on earlier checks
                           raise NotImplementedError(remasking)


                        # Consider confidence only for masked tokens *up to the current block*
                        confidence_demask_mask = torch.full_like(x0_p_de, -np.inf)
                        confidence_demask_mask[:, :end_block_idx] = 0.0 # Allow confidence up to end of current block

                        confidence_de = torch.where(current_demask_mask_index, x0_p_de, -np.inf)
                        confidence_de += confidence_demask_mask

                        x0_de_applied = torch.where(current_demask_mask_index, x0_de, demasking_candidate_batch)

                        # --- Batched Top-K Selection (Global) ---
                        transfer_index_de = torch.zeros_like(x0_de, dtype=torch.bool)
                        for j_de in range(num_demasking_candidates): # Loop over d candidates
                             k_de_tokens = num_demasking_transfer_tokens[j_de, i_de].item()
                             if k_de_tokens > 0:
                                 # Use confidence over the whole allowed region
                                 confidence_global = confidence_de[j_de]
                                 num_masked_global = current_demask_mask_index[j_de].sum().item()
                                 actual_k_de = min(k_de_tokens, confidence_global.numel(), num_masked_global)

                                 if actual_k_de > 0:
                                     # Select top k over the entire sequence (where confidence is not -inf)
                                     _, select_indices_de = torch.topk(confidence_global, k=actual_k_de)
                                     transfer_index_de[j_de, select_indices_de] = True


                        demasking_candidate_batch = torch.where(transfer_index_de, x0_de_applied, demasking_candidate_batch)
                        # print(f"  Demasking Step {i_de+1}: Unmasked {transfer_index_de.sum().item()} tokens across candidates")


                    # --- Score Full Sequences from Demasking Candidates ---
                    print(f"Scoring {num_demasking_candidates} full sequences after demasking...")
                    demasked_sequence_scores = []
                    num_completed_blocks = num_block + 1

                    for d_idx in range(num_demasking_candidates):
                        current_sequence = demasking_candidate_batch[d_idx].unsqueeze(0) # (1, seq_len)
                        # Compute product score for this full candidate sequence
                        product_score = compute_full_sequence_product_score(
                            current_sequence, prompt_len, num_completed_blocks, block_length,
                            tokenizer, prompt_text, prm_model, prm_tokenizer
                        )
                        demasked_sequence_scores.append(product_score)
                        print(f"  Demasked Candidate {d_idx} Product Score: {product_score:.6f}")


                    # --- Select Best Demasked Sequence ---
                    if not demasked_sequence_scores:
                        print("Warning: No scores generated for demasked sequences. Keeping original masked state (before demasking).")
                        # x remains x_masked (or could revert to x before masking) - debatable choice
                        # Let's revert to x before masking happened in this edge case
                        x = x # Revert to state before mask application
                    else:
                        best_demasked_idx = np.argmax(demasked_sequence_scores)
                        print(f"Selected demasked sequence candidate {best_demasked_idx} with product score: {demasked_sequence_scores[best_demasked_idx]:.6f}")
                        # Update x to the best fully demasked sequence
                        x = demasking_candidate_batch[best_demasked_idx].unsqueeze(0)

                        # --- !!! CRITICAL: Recompute block_scores for the chosen sequence !!! ---
                        print("Recomputing individual block scores for the selected demasked sequence...")
                        block_scores = recompute_all_block_scores(
                            x, prompt.to(device), prompt_text, tokenizer, prm_model, prm_tokenizer, block_length
                        )
                        print(f"Updated block scores after demasking: {[f'{s:.4f}' for s in block_scores]}")

                else: # No masks to demask or zero steps
                    print("Skipping demasking as no tokens were masked or demasking steps is zero.")
                    # x remains x_masked if tokens_to_backmask > 0, otherwise it's unchanged x
                    x = x_masked if tokens_to_backmask > 0 else x

            else: # No tokens were selected for backmasking
                 print("No tokens selected for backmasking based on probabilities.")

    # ========================
    # Final Output
    # ========================
    print(f"\n{'='*50}")
    print(f"Generation Complete!")
    print(f"Final block scores: {[f'{score:.4f}' for score in block_scores]}")
    print(f"{'='*50}")

    return x

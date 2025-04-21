import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM # Assuming AutoModelForCausalLM is needed for prm_model

import functools
print = functools.partial(print, flush=True)

# --- Keep these helper functions as they are ---
def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    logits = logits.to(torch.float64)
    # Add small epsilon to prevent log(0)
    noise = torch.rand_like(logits, dtype=torch.float64).clamp(min=1e-9)
    # Apply temperature safely
    if temperature == 0:
         # Deterministic case: choose the highest logit
         # Adding Gumbel noise with temp 0 is equivalent to argmax,
         # but to avoid division by zero, we handle it directly.
         # Let's return something proportional to logits.exp()
         # or handle it specifically where called. For simplicity here,
         # we add a tiny amount of noise conceptually, but the argmax logic handles it.
         # Return logits directly, let argmax handle temp=0 case.
         return logits # Or return logits.exp() - conceptually similar for argmax
    else:
        gumbel_noise = (-torch.log(noise)) ** temperature
        # Add epsilon to gumbel_noise denominator to prevent division by zero
        return logits.exp() / (gumbel_noise + 1e-9)


def get_num_transfer_tokens_schedule(mask_index, steps):
    """
    PRECOMPUTES the number of tokens to transfer at each step for a GIVEN mask.
    Returns a schedule tensor for the batch.

    Args:
        mask_index: Boolean tensor indicating masks (e.g., for a specific block or globally).
        steps: Number of steps to distribute the unmasking over.

    Returns:
        Tensor of shape (batch_size, steps) with the count for each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True) # shape (batch_size, 1)
    if steps == 0: # Avoid division by zero if steps is 0
         return torch.zeros(mask_num.size(0), 0, device=mask_index.device, dtype=torch.int64)
         
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens_schedule = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    # Distribute remainder tokens
    for i in range(mask_num.size(0)): # Iterate over batch
        if remainder[i] > 0:
            num_transfer_tokens_schedule[i, : remainder[i]] += 1

    # Ensure we don't try to transfer more tokens than available masks in total
    # This can happen if steps is very large compared to mask_num
    total_scheduled = num_transfer_tokens_schedule.sum(dim=1)
    over_schedule_mask = total_scheduled > mask_num.squeeze(1)
    if over_schedule_mask.any():
         print(f"Warning: Over-scheduling detected. Adjusting schedule.")
         # Simple correction: Reduce last steps, could be more sophisticated
         diff = (total_scheduled - mask_num.squeeze(1))[over_schedule_mask]
         # Iterate and reduce from the end? Or just recalculate?
         # For simplicity, let's just ensure it sums correctly if needed.
         # A common case is steps > mask_num -> schedule might be [1,1,1,...0,0,0]
         # The base/remainder logic should handle this correctly unless mask_num < steps.
         
         # Recalculate ensuring max 1 per step until mask_num is reached if steps >= mask_num
         for i in torch.where(over_schedule_mask)[0]:
             masks_to_assign = mask_num[i, 0].item()
             num_transfer_tokens_schedule[i, :] = 0
             num_transfer_tokens_schedule[i, :masks_to_assign] = 1


    return num_transfer_tokens_schedule


# --- PRM related functions (keep as is) ---
def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        non_zero_probs = sample[sample.sum(dim=-1) != 0] # Get rows where mask applies
        if non_zero_probs.numel() > 0:
             # Assuming PRM gives N labels, and we want the score for the "correct" label (often index 1)
             # This needs adjustment based on the actual PRM output structure
             # Assuming 2 labels [incorrect, correct] for simplicity:
             positive_probs = non_zero_probs.view(-1, 2)[:, 1]
             non_zero_elements_list = positive_probs.cpu().tolist()
             all_scores_res.append(non_zero_elements_list)
        else:
             all_scores_res.append([])
    return all_scores_res

def calculate_backmasking_probs(block_scores, backmasking_alpha=5.0, min_prob=0.01):
    if not block_scores: return []
    scores = np.array(block_scores)
    probs = np.exp(-backmasking_alpha * scores)
    probs = np.maximum(probs, min_prob)
    max_p, min_p = probs.max(), probs.min()
    if max_p == min_p: return np.full_like(probs, min_prob) # Avoid division by zero if all scores are same
    probs = min_prob + (1 - min_prob) * (probs - min_p) / (max_p - min_p)
    return probs

def get_backmasking_tokens(
    block_indices, # List of boolean masks, one per block
    block_probs,   # Array of probabilities, one per block
    prompt_length,
    block_length,
    backmasking_intensity=0.5,
    x_shape=None,  # Pass the shape of x to create the final mask
):
    if x_shape is None or not block_indices:
        return torch.zeros(x_shape, dtype=torch.bool) # Return empty mask if no shape or blocks

    final_mask = torch.zeros(x_shape, dtype=torch.bool, device=block_indices[0].device) # block_indices[0] provides device

    for i, (indices_mask, prob) in enumerate(zip(block_indices, block_probs)):
        # Calculate number of tokens to mask in this block
        # indices_mask should already be the correct shape (1, seq_len)
        block_token_indices = torch.where(indices_mask[0])[0] # Get indices of True values in this block mask
        block_size = len(block_token_indices)
        if block_size == 0: continue

        num_to_mask = int(block_size * prob * backmasking_intensity)

        if num_to_mask > 0:
            # Select random positions *within the block* to mask
            # Shuffle the indices within the block and take the first num_to_mask
            perm = torch.randperm(block_size, device=indices_mask.device)
            mask_positions_in_block = block_token_indices[perm[:num_to_mask]]
            final_mask[0, mask_positions_in_block] = True # Apply to the final mask

    return final_mask


def compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer):
    # Ensure inputs are valid
    if not block_text.strip(): return 0.0 # Handle empty blocks

    # Format input for PRM model
    formatted_text = "<extra_0>" + block_text + "<extra_0>"
    system = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": formatted_text},
    ]
    conversation_str = prm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Tokenize
    batch = prm_tokenizer(
        conversation_str,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prm_tokenizer.model_max_length # Ensure max_length is respected
    )
    input_ids = batch["input_ids"].to(prm_model.device)
    attention_mask = batch["attention_mask"].to(prm_model.device)

    # Find tokens that match the special token for step separation
    # Handle potential missing token or multiple tokens in encoding
    step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
    if isinstance(step_sep_id, list): step_sep_id = step_sep_id[0] # Take the first if it's a list
    
    token_masks = input_ids == step_sep_id

    # Compute rewards
    with torch.no_grad():
        outputs = prm_model(input_ids=input_ids, attention_mask=attention_mask)
        # Assuming PRM output structure provides logits or scores directly
        # Adjust based on actual PRM model output format
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits 
        
    step_reward = make_step_rewards(logits, token_masks)

    # Average score
    if step_reward and step_reward[0]: # Check if list and first element exist and are not empty
        avg_score = sum(step_reward[0]) / len(step_reward[0])
        return avg_score
    else:
        return 0.0

# --- Refactored Demasking Function ---
def demask_steps_refactored(
    x,                      # Current tokens (batch_size, seq_len)
    mask_schedule,          # Schedule: (batch_size, steps) -> num tokens per step
    limit_mask,             # Boolean mask: (batch_size, seq_len) -> region where unmasking is allowed
    model,
    temperature,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Refactored demasking loop matching the logic of the first generate function.
    Modifies x in place.

    Args:
        x: Token tensor to be modified.
        mask_schedule: Tensor indicating how many tokens to unmask at each step.
        limit_mask: A boolean mask defining the region where tokens CAN be unmasked.
                    Tokens outside this region will never be chosen, even if masked.
        model: The mask predictor model.
        temperature: Sampling temperature.
        cfg_scale: Classifier-Free Guidance scale.
        remasking: 'low_confidence' or 'random'.
        mask_id: ID of the mask token.
    """
    steps = mask_schedule.shape[1]
    prompt_index = x != mask_id # Initial prompt doesn't change

    for i in range(steps):
        num_to_transfer_this_step = mask_schedule[:, i] # Tokens for this step for each item in batch

        # If no tokens to transfer for any item in the batch this step, skip
        if torch.all(num_to_transfer_this_step == 0):
            continue

        # 1) Figure out which positions are *currently* masked AND *within the allowed limit*
        current_mask_index = (x == mask_id) & limit_mask

        # If no valid positions left to unmask, break early
        if not current_mask_index.any():
            print(f"Demasking step {i+1}/{steps}: No valid masked tokens remaining.")
            break

        # 2) Get model logits (use full x, model handles attention)
        if cfg_scale > 0.0:
            # Important: For CFG, the unconditional input should mask the prompt too IF that's how it was trained.
            # Assuming standard LLaDA usage where prompt is kept for conditional, masked for unconditional.
            un_x = torch.full_like(x, mask_id)
            # Keep the prompt in the conditional input
            cond_x = x.clone()
            # Mask the prompt in the unconditional input
            un_x[:, :prompt_index.shape[1]][prompt_index] = x[prompt_index] # Copy prompt to un_x
            un_x[prompt_index] = mask_id # Then mask it

            x_ = torch.cat([cond_x, un_x], dim=0)
            logits_full = model(x_).logits
            logits, un_logits = torch.chunk(logits_full, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(x).logits # Pass full x

        # 3) Add Gumbel noise & pick the argmax prediction (x0)
        # Handle temperature=0 explicitly during selection, not noise addition
        if temperature == 0:
             # Deterministic: take argmax of raw logits
             x0 = torch.argmax(logits, dim=-1)
        else:
             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
             x0 = torch.argmax(logits_with_noise, dim=-1)

        # 4) Compute "confidence" based on original logits
        if remasking == "low_confidence":
             # Use float64 for softmax stability if needed, match add_gumbel_noise
             p = F.softmax(logits.to(torch.float64), dim=-1)
             # Gather probabilities of the predicted tokens (x0)
             x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
        elif remasking == "random":
             x0_p = torch.rand_like(x0, dtype=torch.float64)
        else:
             raise NotImplementedError(remasking)

        # 5) Create the confidence score for *selection*, considering only valid candidates
        # Candidates are tokens that are: currently masked AND within the allowed limit_mask
        confidence_for_selection = torch.where(
            current_mask_index,
            x0_p,
            torch.tensor(-np.inf, dtype=x0_p.dtype, device=x.device) # Set non-candidates to -inf
        )

        # 6) Select top-k indices *from the valid candidates* to unmask this step
        # Need to handle batch size > 1 if applicable, assuming batch size 1 here based on code
        if x.shape[0] == 1:
             num_to_transfer = num_to_transfer_this_step[0].item()
             if num_to_transfer > 0:
                 # Ensure we don't try to select more than available valid masked tokens
                 available_masked = current_mask_index[0].sum().item()
                 k = min(num_to_transfer, available_masked)

                 if k > 0:
                     _, select_indices = torch.topk(confidence_for_selection[0], k=k)
                     # 7) Update x at the selected positions with the model's prediction (x0)
                     x[0, select_indices] = x0[0, select_indices]
        else:
             # Handle batch > 1
             for j in range(x.shape[0]):
                  num_to_transfer = num_to_transfer_this_step[j].item()
                  if num_to_transfer > 0:
                       available_masked = current_mask_index[j].sum().item()
                       k = min(num_to_transfer, available_masked)
                       if k > 0:
                            _, select_indices = torch.topk(confidence_for_selection[j], k=k)
                            x[j, select_indices] = x0[j, select_indices]


# --- Main Generate Function (Modified) ---
@torch.no_grad()
def generate_prm( # Renamed to avoid clash
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
    print("===== Generation Started (PRM Refactored) =====")
    print(f"Prompt (first 100 chars): {tokenizer.decode(prompt[0], skip_special_tokens=True)[:100]}...")
    print(f"Steps: {steps}, Gen length: {gen_length}, Block length: {block_length}")
    print(f"Temperature: {temperature}, CFG scale: {cfg_scale}")
    print(f"Backmasking alpha: {backmasking_alpha}, intensity: {backmasking_intensity}")
    print(f"Global demasking: {global_demasking}, Frequency: {backmasking_frequency}, Threshold: {backmasking_threshold}")

    device = model.device
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, : prompt.shape[1]] = prompt.clone()
    K = backmasking_frequency
    prompt_len = prompt.shape[1]
    prompt_text = tokenizer.decode(prompt[0], skip_special_tokens=True)

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    block_steps = steps // num_blocks

    block_scores = []
    block_masks_list = [] # Store the boolean mask defining each block's region

    for num_block in range(num_blocks):
        print(f"\n--- Processing Block {num_block+1}/{num_blocks} ---")
        block_start_idx = prompt_len + num_block * block_length
        block_end_idx = block_start_idx + block_length

        best_block_score = -np.inf # Initialize lower
        best_block_tokens = None
        retry_count = 0
        meets_threshold = False

        while retry_count < max_retry_attempts and not meets_threshold and (num_block + 1) % K == 0:
            # Reset block tokens only if retrying
            if retry_count > 0:
                print(f"Retry {retry_count}/{max_retry_attempts} for block {num_block+1}")
                x[:, block_start_idx : block_end_idx] = mask_id

            # 1. Define the mask for the current block
            current_block_mask_initial = torch.zeros_like(x, dtype=torch.bool)
            current_block_mask_initial[:, block_start_idx : block_end_idx] = (x[:, block_start_idx : block_end_idx] == mask_id)

            # 2. Get the schedule for unmasking tokens *within this block*
            mask_schedule_block = get_num_transfer_tokens_schedule(current_block_mask_initial, block_steps)

            # 3. Define the limit mask for demasking (only allow unmasking up to the end of this block)
            limit_mask_block = torch.zeros_like(x, dtype=torch.bool)
            limit_mask_block[:, :block_end_idx] = True # Allow unmasking anywhere up to current block end

            # 4. Demask using the refactored function
            demask_steps_refactored(
                x=x,
                mask_schedule=mask_schedule_block,
                limit_mask=limit_mask_block, # Limit unmasking to current block and previous
                model=model,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking=remasking,
                mask_id=mask_id,
            )

            # 5. Compute PRM score
            block_text = tokenizer.decode(
                x[0, block_start_idx : block_end_idx],
                skip_special_tokens=True
            )
            print(f"Block {num_block+1} decoded text (first 50 chars): {block_text[:50]}...")
            block_score = compute_block_score(block_text, prompt_text, prm_model, prm_tokenizer)
            print(f"Block {num_block+1} PRM Score: {block_score:.4f}")

            # Update best score/tokens for this block attempt
            if block_score > best_block_score:
                best_block_score = block_score
                best_block_tokens = x[:, block_start_idx : block_end_idx].clone()

            # Check threshold
            if block_score >= backmasking_threshold:
                meets_threshold = True
                print(f"Block {num_block+1} passed threshold.")
            else:
                retry_count += 1
                if retry_count == max_retry_attempts:
                     print(f"Max retries reached for block {num_block+1}.")

        # If threshold not met after retries, use the best attempt
        if not meets_threshold and best_block_tokens is not None:
            print(f"Using best attempt for block {num_block+1} with score {best_block_score:.4f}")
            x[:, block_start_idx : block_end_idx] = best_block_tokens
            block_score = best_block_score # Use the score of the best attempt
        elif not meets_threshold and best_block_tokens is None:
             print(f"Warning: Block {num_block+1} failed all retries and no best tokens saved. Score=0.")
             block_score = 0.0 # Assign score 0 if all attempts failed somehow


        # Store score and the mask defining this block's region
        block_scores.append(block_score)
        block_region_mask = torch.zeros_like(x, dtype=torch.bool)
        block_region_mask[:, block_start_idx:block_end_idx] = True
        block_masks_list.append(block_region_mask)

        # --- Backmasking Logic ---
        if len(block_scores) >= K and (num_block + 1) % K == 0:
            # Check if any of the last K blocks are below threshold
            if min(block_scores[-K:]) < backmasking_threshold:
                print(f"\n*** Backmasking triggered after block {num_block+1} ***")
                start_idx_backmask = len(block_masks_list) - K
                recent_block_masks = block_masks_list[start_idx_backmask:]
                recent_scores = block_scores[start_idx_backmask:] # Use raw scores for prob calc

                backmasking_probs = calculate_backmasking_probs(recent_scores, backmasking_alpha)

                # Get the final mask indicating which tokens to reset to mask_id
                backmasking_mask = get_backmasking_tokens(
                    recent_block_masks,
                    backmasking_probs,
                    prompt_len,
                    block_length,
                    backmasking_intensity,
                    x_shape=x.shape, # Pass shape
                )

                num_backmasked = backmasking_mask.sum().item()
                if num_backmasked > 0:
                    x[backmasking_mask] = mask_id
                    print(f"Backmasked {num_backmasked} tokens across blocks {start_idx_backmask+1}-{num_block+1}.")

                    # --- Demasking after Backmasking ---
                    currently_masked_tokens = (x == mask_id)
                    num_masked_total = currently_masked_tokens[:, prompt_len:].sum().item() # Count only in generated region
                    
                    # Limit demasking steps to avoid excessive computation, maybe half of original total steps? Or proportional to num_masked?
                    # Let's use a fraction of total steps, proportional to masked ratio, but capped.
                    demasking_steps_after_backmask = min(steps // 2, num_masked_total) # Cap steps

                    print(f"Total {num_masked_total} masked tokens after backmasking. Demasking over {demasking_steps_after_backmask} steps.")

                    if demasking_steps_after_backmask > 0:
                        if global_demasking:
                            print(f"Running global demasking...")
                            # Define the limit mask: allow unmasking anywhere *after prompt*
                            limit_mask_global = torch.zeros_like(x, dtype=torch.bool)
                            limit_mask_global[:, prompt_len:] = True

                            # Get schedule based on *all* currently masked tokens after prompt
                            mask_schedule_global = get_num_transfer_tokens_schedule(
                                currently_masked_tokens & limit_mask_global,
                                demasking_steps_after_backmask
                            )

                            demask_steps_refactored(
                                x=x,
                                mask_schedule=mask_schedule_global,
                                limit_mask=limit_mask_global,
                                model=model,
                                temperature=temperature,
                                cfg_scale=cfg_scale,
                                remasking=remasking,
                                mask_id=mask_id,
                            )
                        else:
                            # Block-by-block demasking (more complex to implement correctly with schedule)
                            # This requires careful handling of steps per block based on remaining masks
                            # For simplicity, the global approach is often preferred after backmasking.
                            # If block-by-block is strictly needed, it would involve iterating through
                            # blocks 0 to num_block, calculating schedule for masks within each, and running demask_steps.
                            print(f"Running block-by-block demasking (Simplified - using global logic for now)...")
                            # Re-using global logic as block-by-block is complex to schedule fairly here
                            limit_mask_global = torch.zeros_like(x, dtype=torch.bool)
                            limit_mask_global[:, prompt_len:] = True
                            mask_schedule_global = get_num_transfer_tokens_schedule(
                                currently_masked_tokens & limit_mask_global,
                                demasking_steps_after_backmask
                            )
                            demask_steps_refactored(
                                x=x, mask_schedule=mask_schedule_global, limit_mask=limit_mask_global,
                                model=model, temperature=temperature, cfg_scale=cfg_scale, remasking=remasking, mask_id=mask_id
                            )

                    # OPTIONAL: Recompute scores for affected blocks after demasking?
                    # Could add: block_scores = recompute_all_block_scores(...) here if needed

    print("\n===== Generation Complete =====")
    print(f"Final block scores: {[f'{s:.3f}' for s in block_scores]}")
    decoded = tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)
    print(f"Generated output:\n{decoded[:500]}\n{'...' if len(decoded) > 500 else ''}")
    return x
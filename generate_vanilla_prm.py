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
    token_masks_float = token_masks.unsqueeze(-1).to(probabilities.dtype)
    probabilities = probabilities * token_masks_float 

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample_probs = probabilities[i] 
        sample_mask = token_masks[i]    
        masked_probs = sample_probs[sample_mask] 
        if masked_probs.shape[0] > 0 and masked_probs.shape[-1] == 2:
            positive_probs = masked_probs[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        else:
            all_scores_res.append([])
    return all_scores_res


@torch.no_grad()
def generate( 
    model,
    prompt,
    prm_model,
    tokenizer,
    prm_tokenizer,
    num_candidates=5,
    steps=128,
    gen_length=512,
    block_length=32,
    temperature=0.6,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    prm_batch_size=None # Optional: Limit PRM batch size if it causes OOM
):
    """
    Generates text by creating candidates in parallel for each block and selecting the best.

    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, l).
        prm_model: Preference Reward Model for scoring generated sequences.
        tokenizer: Tokenizer for the main model.
        prm_tokenizer: Tokenizer for the PRM model.
        num_candidates: Number of candidate sequences to generate per block (in parallel).
        steps: Sampling steps *per block*.
        gen_length: Generated answer length.
        block_length: Block length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK].
        prm_batch_size: Max batch size for PRM scoring (if None, uses num_candidates).
    """
    device = model.device
    # Initial sequence (batch size 1)
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_len = prompt.shape[1]
    prompt_index_base = torch.zeros(1, prompt_len + gen_length, dtype=torch.bool, device=device)
    prompt_index_base[:, :prompt_len] = True

    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length

    # steps_total = steps # Keep original total steps if needed, but generation uses steps_per_block
    steps_per_block = max(1, steps // num_blocks) # Ensure at least 1 step per block


    print(f"Starting parallel generation: {num_candidates} candidates, {num_blocks} blocks, {steps_per_block} steps/block.")

    for num_block in range(num_blocks):
        print(f"--- Processing Block {num_block+1}/{num_blocks} ---")
        start_block_idx = prompt_len + num_block * block_length
        end_block_idx = start_block_idx + block_length

        # --- Parallel Candidate Generation ---
        # Expand the current best sequence 'x' into a batch for candidates
        candidate_batch_x = x.repeat(num_candidates, 1) # Shape: (num_candidates, seq_len)
        prompt_index = prompt_index_base.repeat(num_candidates, 1) # Shape: (num_candidates, seq_len)

        # Get mask for the *current block* across the batch
        block_mask_index_batch = torch.zeros_like(candidate_batch_x, dtype=torch.bool)
        block_mask_index_batch[:, start_block_idx:end_block_idx] = (
            candidate_batch_x[:, start_block_idx:end_block_idx] == mask_id
        )

        
        num_transfer_tokens = get_num_transfer_tokens(
            block_mask_index_batch[:, start_block_idx:end_block_idx], # Pass only the block's mask portion
            steps_per_block
            ) # Shape: (num_candidates, steps_per_block)

        # Iteratively decode the current block for all candidates in parallel
        for i in range(steps_per_block):
            mask_index = candidate_batch_x == mask_id # Full mask for confidence calc: (num_candidates, seq_len)

            # --- Batched Model Inference ---
            if cfg_scale > 0.0:
                un_x = candidate_batch_x.clone()
                un_x[prompt_index] = mask_id # Mask prompt tokens
                 # Batch becomes [candidates, candidates_unconditioned]
                x_ = torch.cat([candidate_batch_x, un_x], dim=0) # Shape: (2 * num_candidates, seq_len)
                all_logits = model(x_).logits # Shape: (2 * num_candidates, seq_len, vocab_size)
                logits, un_logits = torch.chunk(all_logits, 2, dim=0) # Each: (num_candidates, seq_len, vocab_size)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits) # Apply CFG
            else:
                logits = model(candidate_batch_x).logits # Shape: (num_candidates, seq_len, vocab_size)

            # --- Batched Sampling & Confidence ---
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # Predicted tokens: (num_candidates, seq_len)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1) # (num_candidates, seq_len, vocab_size)
                # Gather probabilities of the predicted tokens x0
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1) # (num_candidates, seq_len)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64) # (num_candidates, seq_len)
            else:
                raise NotImplementedError(remasking)

            # Ignore confidence outside the current block for selection
            # Create a mask to keep confidence only within the current block
            confidence_block_mask = torch.full_like(x0_p, -np.inf)
            confidence_block_mask[:, start_block_idx:end_block_idx] = 0.0 # Allow confidence here
            
            # Apply mask: confidence is original where mask_index is True AND within the block, else -inf
            confidence = torch.where(mask_index, x0_p, -np.inf)
            confidence += confidence_block_mask # Zeros out confidence outside the block

            # Where `mask_index` is true, replace with prediction `x0`, otherwise keep original `candidate_batch_x`
            x0_applied = torch.where(mask_index, x0, candidate_batch_x)

            # --- Batched Top-K Selection ---
            # Select tokens to unmask in this step based on confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            # Loop necessary because topk `k` varies per candidate per step
            for j in range(num_candidates):
                k = num_transfer_tokens[j, i].item() # Get k for this candidate/step
                if k > 0:
                     # Find top-k confident indices *only within the current block* for this candidate
                     # Note: Confidence outside the block is already -inf
                    _, select_indices = torch.topk(confidence[j, start_block_idx:end_block_idx], k=k)
                    # Adjust indices to be relative to the full sequence length
                    select_indices_global = select_indices + start_block_idx
                    transfer_index[j, select_indices_global] = True

            # Apply the transfers for all candidates
            candidate_batch_x = torch.where(transfer_index, x0_applied, candidate_batch_x)
            # Alternative (might be slightly clearer):
            # candidate_batch_x[transfer_index] = x0[transfer_index]

        # --- Parallel Candidate Scoring ---
        print(f"Scoring {num_candidates} candidates for block {num_block+1}...")
        all_scores = []
        actual_prm_batch_size = prm_batch_size if prm_batch_size else num_candidates

        # Process candidates in mini-batches for PRM if needed (memory)
        for batch_start in range(0, num_candidates, actual_prm_batch_size):
            batch_end = min(batch_start + actual_prm_batch_size, num_candidates)
            current_candidate_batch = candidate_batch_x[batch_start:batch_end] # (current_bs, seq_len)

            # Decode the *current block* for the mini-batch
            decoded_blocks = []
            for cand_idx in range(current_candidate_batch.size(0)):
                block_text = tokenizer.decode(
                    current_candidate_batch[cand_idx, start_block_idx:end_block_idx],
                    skip_special_tokens=True,
                )
                formatted_text = "<extra_0>" + block_text + "<extra_0>"
                decoded_blocks.append(formatted_text)

            # Prepare PRM inputs for the mini-batch
            prm_input_list = []
            query = tokenizer.decode(prompt[0], skip_special_tokens=True) # Decode prompt once
            system = "Please reason step by step, and put your final answer within \\boxed{}."

            for candidate_block_text in decoded_blocks:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": candidate_block_text},
                ]
                # Tokenize individually first to handle varying lengths before padding
                # Don't add tensors yet, let pad handle it
                tokenized = prm_tokenizer(
                     prm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False),
                     return_tensors=None, # Return list of ids first
                     add_special_tokens=True # Ensure BOS/EOS if needed by PRM tokenizer
                     )
                prm_input_list.append(tokenized)

            # Pad the batch of PRM inputs
            # The padder handles creating 'input_ids' and 'attention_mask'
            prm_inputs_padded = prm_tokenizer.pad(
                prm_input_list,
                padding=True,
                return_tensors="pt"
            ).to(prm_model.device)


            # --- Batched PRM Inference ---
            prm_outputs = prm_model(**prm_inputs_padded) # Pass dict directly

            # Find the special token masks in the padded batch
            step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>") # More robust way
            # Need to use input_ids from the padded batch now
            prm_token_masks = prm_inputs_padded['input_ids'] == step_sep_id # (current_bs, prm_seq_len)

            # Calculate step rewards for the batch
            # Use the adapted batched reward function
            batch_step_rewards = make_step_rewards_batched(prm_outputs[0], prm_token_masks)

            # Calculate average score for each candidate in the mini-batch
            for step_reward_list in batch_step_rewards:
                if step_reward_list:
                    avg_score = sum(step_reward_list) / len(step_reward_list)
                    all_scores.append(avg_score)
                else:
                    all_scores.append(float('-inf')) # Assign very low score if no rewards found

        # --- Select Best Candidate ---
        if not all_scores:
             print("Warning: No scores generated for any candidate. Selecting candidate 0.")
             best_candidate_idx = 0
        else:
            best_candidate_idx = np.argmax(all_scores)
            print(f"Selected candidate {best_candidate_idx} with score: {all_scores[best_candidate_idx]:.4f}")

        # Update the main sequence 'x' with the best candidate's full sequence
        # Note: candidate_batch_x contains the *full* sequences up to the current block
        x = candidate_batch_x[best_candidate_idx].unsqueeze(0) # Back to shape (1, seq_len)

    print("--- Generation Complete ---")
    return x

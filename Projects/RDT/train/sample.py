from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
def check_condition(expanded_state_norm):

    front_indices = torch.tensor([0,1,2,3,4,5], dtype=torch.long)        
    back_indices = torch.tensor([50,51,52,53,54,55], dtype=torch.long)   

    front_values = expanded_state_norm[..., front_indices]  
    back_values = expanded_state_norm[..., back_indices]     
    
    cond_front_zero = (front_values == 0).all(dim=-1)       
    cond_back_zero = (back_values == 0).all(dim=-1)         
    # print("cond_front_zero", cond_front_zero)
    # print("cond_back_zero", cond_back_zero)
    final_condition = cond_front_zero | cond_back_zero      
    
    return final_condition, cond_front_zero, cond_back_zero

@torch.no_grad()
def log_sample_res(
    text_encoder, vision_encoder, rdt, args, 
    accelerator, weight_dtype, dataset_id2name, dataloader, logger
):
    with torch.autocast(device_type='cuda',dtype=torch.float16):
        logger.info(
            f"Running sampling for {args.num_sample_batches} batches..."
        )

        rdt.eval()
        
        loss_for_log = defaultdict(float)
        loss_counter = defaultdict(int)
        for step, batch in enumerate(dataloader):
            if step >= args.num_sample_batches:
                break
            
            data_indices = batch["data_indices"]
            ctrl_freqs = batch["ctrl_freqs"]
            state_norm = batch["state_norm"].to(dtype=weight_dtype)
            images = batch["images"].to(dtype=weight_dtype)
            states = batch["states"].to(dtype=weight_dtype)
            # We only use the last state as input
            states = states[:, -1:, :]
            actions = batch["actions"].to(dtype=weight_dtype)
            state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                
            batch_size, _, C, H, W = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
            image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))
            
            lang_attn_mask = batch["lang_attn_mask"]
            text_embeds = batch["lang_embeds"].to(dtype=weight_dtype) \
                if args.precomp_lang_embed \
                else text_encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=lang_attn_mask
                )["last_hidden_state"].detach()
                
            pred_actions = rdt.predict_action(
                lang_tokens=text_embeds,
                lang_attn_mask=lang_attn_mask,
                img_tokens=image_embeds,
                state_tokens=states,
                action_mask=state_elem_mask.unsqueeze(1),
                ctrl_freqs=ctrl_freqs
            )
            
            num_steps = pred_actions.shape[1]
            # print("state_norm", state_norm.shape)
            expanded_state_elem_mask = state_elem_mask.unsqueeze(1).tile((1, num_steps, 1)).float()
            expanded_state_norm = state_norm.unsqueeze(1).tile((1, num_steps, 1)).float()
            # print("expanded_state_norm", expanded_state_norm)
            loss = F.mse_loss(pred_actions, actions, reduction='none').float()
            
            mse_loss_per_entry = ((loss * expanded_state_elem_mask).reshape((batch_size, -1)).sum(1)
                                / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1))
            final_condition, is_front_zero, is_back_zero = check_condition(state_norm)

            l2_loss_per_entry = loss.sqrt() / (expanded_state_norm + 1e-3)
            if final_condition.any():
                # print("Condition met, using l2_loss = loss.sqrt()")
                for i in range(is_front_zero.shape[0]):
                    if is_front_zero[i].any():
                        # print("Front zero condition met",i)
                        l2_loss_per_entry[i, :, [0,1,2,3,4,5]] = loss[i, :, [0,1,2,3,4,5]].sqrt()
                for j in range(is_back_zero.shape[0]):
                    if is_back_zero[j].any():
                        # print("Back zero condition met",j)
                        l2_loss_per_entry[j, :, [50,51,52,53,54,55]] = loss[j, :, [50,51,52,53,54,55]].sqrt()

            # print("l2_loss_per_entry", l2_loss_per_entry)
            l2_loss_per_entry = ((l2_loss_per_entry * expanded_state_elem_mask).reshape((batch_size, -1)).sum(1)
                            / expanded_state_elem_mask.reshape((batch_size, -1)).sum(1))

            dataset_indices, mse_losses, l2_losses = accelerator.gather_for_metrics(
                (torch.LongTensor(data_indices).to(device=pred_actions.device), 
                mse_loss_per_entry, l2_loss_per_entry),
            ) 
            dataset_indices = dataset_indices.tolist()
            if accelerator.is_main_process:
                for loss_suffix, losses in zip(["_sample_mse", "_sample_l2err"], [mse_losses, l2_losses]):
                    for dataset_idx, loss_tensor in zip(dataset_indices, losses):
                        loss_name = dataset_id2name[dataset_idx] + loss_suffix
                        loss_for_log[loss_name] += loss_tensor.item()
                        loss_counter[loss_name] += 1
            
            mse_loss = (loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
            mse_loss_scaler = accelerator.gather(mse_loss).mean().item()
            loss_for_log["overall_avg_sample_mse"] += mse_loss_scaler

            # import ipdb; ipdb.set_trace()
            # print(f"expanded_state_elem_mask", expanded_state_elem_mask)
            l2_loss = loss.sqrt() / (expanded_state_norm + 1e-3)
            
            #修改点
            if final_condition.any():
                # print("Condition met, using l2_loss = loss.sqrt()")
                for i in range(is_front_zero.shape[0]):
                    if is_front_zero[i].any():
                        # print("Front zero condition met",i)
                        l2_loss[i, :, [0,1,2,3,4,5]] = loss[i, :, [0,1,2,3,4,5]].sqrt()
                for j in range(is_back_zero.shape[0]):
                    if is_back_zero[j].any():
                        # print("Back zero condition met",j)
                        l2_loss[j, :, [50,51,52,53,54,55]] = loss[j, :, [50,51,52,53,54,55]].sqrt()
            # import ipdb; ipdb.set_trace()
            
                
            # l2_loss = loss.sqrt() 
            l2_loss = (l2_loss * expanded_state_elem_mask).sum() / expanded_state_elem_mask.sum()
            l2_loss_scaler = accelerator.gather(l2_loss).mean().item()
            # print("l2_loss_scaler", l2_loss_scaler)
            loss_for_log["overall_avg_sample_l2err"] += l2_loss_scaler
            print(f"Step {step}: overall_avg_sample_mse: {loss_for_log['overall_avg_sample_mse']}, overall_avg_sample_l2err: {loss_for_log['overall_avg_sample_l2err']}")

        for name in loss_for_log:
            if name in ["overall_avg_sample_mse", "overall_avg_sample_l2err"]:
                loss_scaler = loss_for_log[name]
                loss_for_log[name] = round(loss_scaler / (args.num_sample_batches), 4)
            else:
                loss_for_log[name] = round(loss_for_log[name] / loss_counter[name], 4)
        
        rdt.train()
        torch.cuda.empty_cache()

        return dict(loss_for_log)
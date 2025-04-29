import json
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import os
from accelerate import Accelerator

class CODI_Dataset(Dataset) :

    def __init__(self, data_dir, tokenizer, split = 'train', icot_length = 6, max_length = 256) :

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.split = split
        self.icot_length = icot_length
        self.max_length = max_length

        self.data = []
        for root, _, files in os.walk(os.path.join(data_dir, split)):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    df = pd.read_parquet(file_path)
                    self.data.extend(df.to_dict(orient='records'))

    def __len__(self) :

        return len(self.data)

    def __getitem__(self, idx) :

        if self.split == 'train' :

            # Get the question and answer
            question = self.data[idx]['quiz']
            answer = self.data[idx]['solution_text']

            # Construct the chain of thought
            cot_head = self.data[idx]['cot_head']
            cot_steps = self.data[idx]['cot_repeat_steps']
            cot_foot = self.data[idx]['cot_foot']

            # Combine all CoT components
            cot = f"{cot_head}\n" + "\n".join(cot_steps) + f"\n{cot_foot}"

            bos_input_ids = self.tokenizer('<|endoftext|>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            question_input_ids = self.tokenizer(question, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            cot_input_ids = self.tokenizer(cot, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            icot_input_ids = self.tokenizer('<bot>' + '<|endoftext|>' * self.icot_length + '<eot>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            symbol_input_ids = self.tokenizer('The Answer is:', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            answer_input_ids = self.tokenizer(answer, return_tensors = 'pt', add_special_tokens = False)['input_ids']

            if len(cot) != 0 :
                teacher_input_ids = torch.cat([bos_input_ids, question_input_ids, cot_input_ids, symbol_input_ids, answer_input_ids], dim = 1)
            else :
                teacher_input_ids = torch.cat([bos_input_ids, question_input_ids, symbol_input_ids, answer_input_ids], dim = 1)
            student_input_ids = torch.cat([bos_input_ids, question_input_ids, icot_input_ids, symbol_input_ids, answer_input_ids], dim = 1)

            teacher_pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(teacher_input_ids[0]))], dim = 1)
            student_pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(student_input_ids[0]))], dim = 1)
            teacher_input_ids = torch.cat([teacher_input_ids, teacher_pad_input_ids], dim = 1)
            student_input_ids = torch.cat([student_input_ids, student_pad_input_ids], dim = 1)

            teacher_attention_mask = torch.ones_like(teacher_input_ids)
            teacher_attention_mask[:, -len(teacher_pad_input_ids[0]):] = 0
            student_attention_mask = torch.ones_like(student_input_ids)
            student_attention_mask[:, -len(student_pad_input_ids[0]):] = 0

            teacher_loss_mask = teacher_attention_mask.clone()
            student_loss_mask = student_attention_mask.clone()
            teacher_loss_mask[:, :len(bos_input_ids[0]) + len(question_input_ids[0])] = 0
            student_loss_mask[:, :len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0])] = 0

            teacher_symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(cot_input_ids[0]) + len(symbol_input_ids[0]) - 1
            student_symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0]) + len(symbol_input_ids[0]) - 1

            student_bot_position = len(bos_input_ids[0]) + len(question_input_ids[0])

            return {
                'teacher_input_ids' : teacher_input_ids,
                'student_input_ids' : student_input_ids,
                'teacher_attention_mask' : teacher_attention_mask,
                'student_attention_mask' : student_attention_mask,
                'teacher_loss_mask' : teacher_loss_mask,
                'student_loss_mask' : student_loss_mask,
                'teacher_symbol_position' : teacher_symbol_position,
                'student_symbol_position' : student_symbol_position,
                'student_bot_position' : student_bot_position
            }

        if self.split == 'test' :

            question = self.data[idx]['quiz']
            answer = self.data[idx]['solution_text']

            bos_input_ids = self.tokenizer('<|endoftext|>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            question_input_ids = self.tokenizer(question, return_tensors = 'pt', add_special_tokens = False)['input_ids']
            icot_input_ids = self.tokenizer('<bot>' + '<|endoftext|>' * self.icot_length + '<eot>', return_tensors = 'pt', add_special_tokens = False)['input_ids']
            symbol_input_ids = self.tokenizer('The Answer is:', return_tensors = 'pt', add_special_tokens = False)['input_ids']

            input_ids = torch.cat([bos_input_ids, question_input_ids, icot_input_ids, symbol_input_ids], dim = 1)

            pad_input_ids = torch.cat([bos_input_ids for i in range(self.max_length - len(input_ids[0]))], dim = 1)
            input_ids = torch.cat([input_ids, pad_input_ids], dim = 1)

            attention_mask = torch.ones_like(input_ids)
            attention_mask[:, -len(pad_input_ids[0]):] = 0

            symbol_position = len(bos_input_ids[0]) + len(question_input_ids[0]) + len(icot_input_ids[0]) + len(symbol_input_ids[0]) - 1

            bot_position = len(bos_input_ids[0]) + len(question_input_ids[0])

            return {
                'answer' : answer,
                'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'bot_position' : bot_position,
                'symbol_position' : symbol_position
            }

class CODI_Model(nn.Module) :

    def __init__(self, model_path, icot_length = 6, alpha = 1, beta = 1, gamma = 1, max_length = 256) :

        super(CODI_Model, self).__init__()

        self.max_length = max_length
        self.icot_length = icot_length

        # Initialize Qwen2.5 tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-1M")
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<bot>', '<eot>']})
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the model with bfloat16 precision and move to CUDA
        self.model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct-1M",
            torch_dtype=torch.bfloat16
        ).to('cuda')

        # Resize token embeddings to accommodate new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)

        # Configure LoRA for Qwen2.5
        config = LoraConfig(
            r=128,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Updated for Qwen2.5 architecture
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            fan_in_fan_out=True
        )
        self.model = get_peft_model(self.model, config)

        self.proj = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size, dtype=torch.bfloat16),
            nn.LayerNorm(self.model.config.hidden_size, dtype=torch.bfloat16)
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.l1_loss = nn.SmoothL1Loss()

    def forward(self, inputs) :

        teacher_input_ids = inputs['teacher_input_ids'].to('cuda')
        teacher_inputs_embeds = self.model.get_input_embeddings()(teacher_input_ids).squeeze(dim = 1)
        teacher_attention_mask = inputs['teacher_attention_mask'].to('cuda')
        teacher_loss_mask = inputs['teacher_loss_mask'].to('cuda')
        teacher_labels = torch.roll(teacher_input_ids, shifts = -1, dims = 2).to('cuda')
        teacher_labels[:, :, -1] = self.tokenizer.eos_token_id
        teacher_outputs = self.model(inputs_embeds = teacher_inputs_embeds, attention_mask = teacher_attention_mask, output_hidden_states = True)
        teacher_logits = teacher_outputs['logits']
        teacher_hidden_states = teacher_outputs['hidden_states']
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        teacher_labels = teacher_labels.view(-1)
        teacher_loss_mask = teacher_loss_mask.view(-1)
        teacher_loss = self.ce_loss(teacher_logits, teacher_labels)
        teacher_loss = torch.mean(teacher_loss[teacher_loss_mask.bool()])

        student_input_ids = inputs['student_input_ids'].to('cuda')
        student_inputs_embeds = self.model.get_input_embeddings()(student_input_ids).squeeze(dim = 1)
        student_bot_position = inputs['student_bot_position'].to('cuda')
        student_attention_mask = inputs['student_attention_mask'].to('cuda')
        student_loss_mask = inputs['student_loss_mask'].to('cuda')
        student_labels = torch.roll(student_input_ids, shifts = -1, dims = 2).to('cuda')
        student_labels[:, :, -1] = self.tokenizer.eos_token_id
        for i in range(self.icot_length) :
            hidden_states = self.model(inputs_embeds = student_inputs_embeds, attention_mask = student_attention_mask, output_hidden_states = True).hidden_states[-1]
            for j in range(len(student_bot_position)) :
                student_inputs_embeds[j, student_bot_position[j] + 1] = self.proj(hidden_states[j, student_bot_position[j]])
                student_bot_position[j] += 1
        student_outputs = self.model(inputs_embeds = student_inputs_embeds, attention_mask = student_attention_mask, output_hidden_states = True)
        student_logits = student_outputs['logits']
        student_hidden_states = student_outputs['hidden_states']
        student_logits = student_logits.view(-1, student_logits.shape[-1])
        student_labels = student_labels.view(-1)
        student_loss_mask = student_loss_mask.view(-1)
        student_loss = self.ce_loss(student_logits, student_labels)
        student_loss = torch.mean(student_loss[student_loss_mask.bool()])

        teacher_symbol_positions = inputs['teacher_symbol_position']
        student_symbol_positions = inputs['student_symbol_position']

        distill_loss = 0
        for i in range(len(teacher_hidden_states) - 1) :
            teacher_distill_hidden_states = []
            student_distill_hidden_states = []
            for j in range(len(teacher_hidden_states[i])) :
                teacher_distill_hidden_states.append(teacher_hidden_states[i][j, teacher_symbol_positions[j]])
                student_distill_hidden_states.append(student_hidden_states[i][j, student_symbol_positions[j]])
            teacher_distill_hidden_states = torch.stack(teacher_distill_hidden_states)
            student_distill_hidden_states = torch.stack(student_distill_hidden_states)
            distill_loss += self.l1_loss(teacher_distill_hidden_states, student_distill_hidden_states) / torch.std(teacher_distill_hidden_states)
        distill_loss = distill_loss / (len(teacher_hidden_states) - 1)

        loss = self.alpha * teacher_loss + self.beta * student_loss + self.gamma * distill_loss

        return {
            'teacher_loss' : teacher_loss,
            'student_loss' : student_loss,
            'distill_loss' : distill_loss,
            'loss' : loss
        }

    def test(self, inputs) :

        true_answers = inputs['answer']
        input_ids = inputs['input_ids'].to('cuda')
        inputs_embeds = self.model.get_input_embeddings()(input_ids).squeeze(dim = 1)
        bot_position = inputs['bot_position'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')
        self.model.eval()
        for i in range(self.icot_length) :
            hidden_states = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask, output_hidden_states = True).hidden_states[-1]
            for j in range(len(bot_position)) :
                inputs_embeds[j, bot_position[j] + 1] = self.proj(hidden_states[j, bot_position[j]])
                bot_position[j] += 1
        symbol_positions = inputs['symbol_position']
        results = []
        eos_reached = [False] * len(symbol_positions)
        while torch.min(symbol_positions).item() < self.max_length - 1 :
            logits = self.model(inputs_embeds = inputs_embeds, attention_mask = attention_mask).logits
            next_ids = []
            for j in range(len(symbol_positions)) :
                if eos_reached[j] :
                    next_ids.append(self.tokenizer.eos_token_id)
                elif symbol_positions[j] < self.max_length - 1 :
                    next_id = torch.argmax(logits[j, symbol_positions[j]])
                    next_ids.append(next_id.item())
                    next_embeds = self.model.get_input_embeddings()(next_id)
                    symbol_positions[j] += 1
                    attention_mask[j, 0, symbol_positions[j]] = 1
                    inputs_embeds[j, symbol_positions[j]] = next_embeds
                else :
                    next_ids.append(self.tokenizer.eos_token_id)
                    eos_reached[j] = True
            results.append(next_ids)
            if next_ids == [self.tokenizer.eos_token_id for i in range(len(symbol_positions))] :
                break
        results = torch.tensor(results).transpose(0, 1).tolist()
        answers = []
        for result in results :
            answers.append(self.tokenizer.decode(result, skip_special_tokens = True))
        return {
            'true_answers' : true_answers,
            'answers' : answers
        }

def CODI_train(
    model_path = "Qwen/Qwen2.5-7B-Instruct-1M",
    data_dir = os.path.expandvars("$SLURM_TMPDIR/data/instruct"),
    epochs = 40,
    batch_size = 16,  # Increased for A100
    gradient_accumulation_steps = 4,  # Adjusted for larger batch size
    lr = 2e-4,  # Slightly increased learning rate
    weight_decay = 0.01,
    warmup_rate = 0.1,
    test_steps = 100,
    save_steps = 100,
    num_workers = 4, # For data loading
    patience = 3,  # Number of test steps to wait before early stopping
    min_delta = 0.001  # Minimum change in loss to be considered an improvement
) :

    accelerator = Accelerator()
    model = CODI_Model(model_path)

    # Load data from the new directory structure
    train_data = CODI_Dataset(data_dir, model.tokenizer, split='train')
    test_data = CODI_Dataset(data_dir, model.tokenizer, split='test')

    # Use DataLoader with multiple workers
    train_data_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster data transfer to GPU
    )
    test_data_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,           # Learning rate: Increased from 1e-4 to 2e-4 for faster convergence
        weight_decay=weight_decay, # L2 regularization: Helps prevent overfitting
        betas=(0.9, 0.999), # Momentum parameters:
                            # - First beta (0.9): Controls the exponential decay rate for the first moment estimates
                            # - Second beta (0.999): Controls the exponential decay rate for the second moment estimates
        eps=1e-8           # Epsilon: Small constant for numerical stability
    )

    total_steps = len(train_data_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(warmup_rate * total_steps),
        num_training_steps = total_steps
    )

    # Prepare everything for distributed training
    model, optimizer, train_data_loader, test_data_loader, scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, test_data_loader, scheduler
    )

    # Create checkpoint directory
    ckpt_dir = os.path.expandvars("$SLURM_TMPDIR/results/ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = os.path.join(ckpt_dir, "final")

    model.train()

    accumulated_steps = 0
    loss_list = []
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None


    for epoch in range(epochs) :
        for batch in train_data_loader :
            with accelerator.accumulate(model):
                loss = model(batch)['loss'] / gradient_accumulation_steps
                accelerator.backward(loss)
                loss_list.append(loss.item())
                accumulated_steps += 1

                if accumulated_steps % gradient_accumulation_steps == 0 :
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    print(f'epoch: {epoch}, step: {accumulated_steps / gradient_accumulation_steps}, loss: {sum(loss_list)}')
                    loss_list = []

                if (accumulated_steps / gradient_accumulation_steps) % test_steps == 0 :
                    # Evaluate on test set
                    model.eval()
                    test_loss = 0
                    test_batches = 0
                    with torch.no_grad():
                        for test_batch in test_data_loader:
                            test_result = model(test_batch)
                            test_loss += test_result['loss'].item()
                            test_batches += 1
                    test_loss /= test_batches
                    model.train()

                    print(f'Test loss: {test_loss:.4f}')

                    # Early stopping check
                    if test_loss < best_test_loss - min_delta:
                        best_test_loss = test_loss
                        patience_counter = 0
                        # Save and immediately use the best model state
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        best_model_state = accelerator.get_state_dict(unwrapped_model)
                        # Immediately load the best state back to continue training from the best point
                        accelerator.load_state_dict(unwrapped_model, best_model_state)
                        print(f'New best test loss: {test_loss:.4f}, saving model state')
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f'Early stopping triggered after {accumulated_steps / gradient_accumulation_steps} steps')
                            # Load best model state
                            unwrapped_model = accelerator.unwrap_model(model)
                            accelerator.load_state_dict(unwrapped_model, best_model_state)
                            # Save final model
                            accelerator.wait_for_everyone()
                            accelerator.save_state(final_ckpt_path)
                            return

                    # Save test results
                    test_result = CODI_test(model, test_data_loader)
                    with open(f'{os.environ.get("SLURM_TMPDIR")}/result/test/{int((accumulated_steps / gradient_accumulation_steps) / test_steps)}.json', 'a') as f :
                        json.dump(test_result, f)

                if (accumulated_steps / gradient_accumulation_steps) % save_steps == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_{int((accumulated_steps / gradient_accumulation_steps) / save_steps)}")
                    accelerator.save_state(checkpoint_path)

    # Save final model
    accelerator.wait_for_everyone()
    accelerator.save_state(final_ckpt_path)

def CODI_test(model, test_data_loader) :

    model.eval()
    answers = []
    true_answers = []
    for batch in tqdm(test_data_loader) :
        result = model.test(batch)
        answers = answers + result['answers']
        true_answers = true_answers + result['true_answers']
    return {
        'answers' : answers,
        'true_answers' : true_answers
    }

if __name__ == '__main__' :

    # 训练
    CODI_train()

    # 测试
    # num = 40
    # model = CODI_Model('gpt2')
    # model.load_state_dict(torch.load(f'../result/ckpt/{num}.pth'))
    # test_data = CODI_Dataset('../data/test.parquet', model.tokenizer, split = 'test')
    # test_data_loader = DataLoader(test_data, batch_size = 64, shuffle = False)
    # result = CODI_test(model, test_data_loader)
    # print(result)

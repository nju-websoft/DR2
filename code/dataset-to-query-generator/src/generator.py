import argparse
import json
import logging
import os

import data_process
import evaluate
import numpy as np
import torch
from tools import seed_everything
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AdamW, AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_scheduler)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model")


def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        if args.text_field == 'metadata+data':
            batch_inputs.append(sample['metadata'])
            batch_inputs.append(sample['data'])
            batch_targets.append(sample['keyword'])
            batch_targets.append(sample['keyword'])
        else:
            batch_inputs.append(sample[args.text_field])
            batch_targets.append(sample['keyword'])
    batch_data = tokenizer(
        text=batch_inputs,
        padding=True,
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenizer(
        text_target=batch_targets,
        padding=True,
        max_length=args.max_target_length,
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
    end_token_index = torch.where(labels == tokenizer.eos_token_id)[1]
    for idx, end_idx in enumerate(end_token_index):
        labels[idx][end_idx + 1:] = -100
    batch_data['labels'] = labels
    return batch_data


def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_batch_num = epoch * len(dataloader)

    model.train()
    for batch, batch_data in enumerate(dataloader, start=1):
        batch_data = batch_data.to(args.device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(args, dataloader, model, tokenizer):
    preds, labels = [], []
    rouge = evaluate.load('rouge')

    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            batch_data = batch_data.to(args.device)
            generated_tokens = model.generate(
                batch_data["input_ids"],
                attention_mask=batch_data["attention_mask"],
                max_length=args.max_target_length,
                num_beams=args.beam_search_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            ).cpu().numpy()
            if isinstance(generated_tokens, tuple):
                generated_tokens = generated_tokens[0]
            label_tokens = batch_data["labels"].cpu().numpy()

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
            label_tokens = np.where(label_tokens != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True,
                                                    clean_up_tokenization_spaces=False)
            preds += decoded_preds
            labels += decoded_labels
    scores = rouge.compute(predictions=preds, references=labels)
    result = {'rouge-1': scores['rouge1'] * 100, 'rouge-2': scores['rouge2'] * 100, 'rouge-l': scores['rougeL'] * 100}
    result['avg'] = np.mean(list(result.values()))
    return result


def train(args, train_dataset, dev_dataset, model, tokenizer):
    """ Train the model """
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collote_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    with open(os.path.join(args.output_dir, f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/', 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_avg_rouge = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}\n" + 30 * "-")
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        dev_rouges = test_loop(args, dev_dataloader, model, tokenizer)
        logger.info(
            f"Dev Rouge1: {dev_rouges['rouge-1']:>0.2f} Rouge2: {dev_rouges['rouge-2']:>0.2f} RougeL: {dev_rouges['rouge-l']:>0.2f}")
        rouge_avg = dev_rouges['avg']
        if rouge_avg > best_avg_rouge:
            if epoch > 0:
                for root, dirs, files in os.walk(os.path.join(args.output_dir, f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/'), topdown=False):
                    for name in files:
                        if name.endswith('.bin',):
                            os.remove(os.path.join(root, name))
            best_avg_rouge = rouge_avg
            logger.info(f'saving new weights to {args.output_dir}{args.text_field}/fold{args.fold}/...\n')
            save_weight = f'epoch_{epoch + 1}_dev_rouge_avg_{rouge_avg:0.4f}_weights.bin'
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/', save_weight))
    logger.info("Done!")


def test(args, test_dataset, model, tokenizer, save_weights: list):
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collote_fn)
    logger.info('***** Running testing *****')
    for save_weight in save_weights:
        logger.info(f'loading weights from {save_weight}...')
        model.load_state_dict(
            torch.load(os.path.join(args.output_dir, f'{args.text_field}/fold{args.fold}/best/', save_weight)))
        test_rouges = test_loop(args, test_dataloader, model, tokenizer)
        logger.info(
            f"Test Rouge1: {test_rouges['rouge-1']:>0.2f} Rouge2: {test_rouges['rouge-2']:>0.2f} RougeL: {test_rouges['rouge-l']:>0.2f}")


def predict(args, document: str, model, tokenizer):
    inputs = tokenizer(
        document,
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt"
    )
    inputs = inputs.to(args.device)
    with torch.no_grad():
        generated_tokens = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=args.max_target_length,
            num_beams=args.beam_search_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        ).cpu().numpy()

    if isinstance(generated_tokens, tuple):
        generated_tokens = generated_tokens[0]
    decoded_preds = tokenizer.decode(
        generated_tokens[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return decoded_preds


def compute_loss(args, document: str, prediction: str, model, tokenizer):
    inputs = tokenizer(
        document,
        padding=True,
        max_length=args.max_input_length,
        truncation=True,
        return_tensors="pt"
    )['input_ids']
    inputs = inputs.to(args.device)
    prediction_tokens = tokenizer(
        text_target=prediction,
        padding=True,
        max_length=args.max_target_length,
        truncation=True,
        return_tensors="pt"
    )['input_ids']
    prediction_tokens = prediction_tokens.to(args.device)
    with torch.no_grad():
        loss = model(input_ids=inputs, labels=prediction_tokens).loss
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.",
                        )

    parser.add_argument("--model_type",
                        default="bert", type=str, required=True
                        )
    parser.add_argument("--model_checkpoint",
                        default="bert-large-cased/", type=str, required=True,
                        help="Path to pretrained model or model identifier from huggingface.co/models",
                        )
    parser.add_argument("--max_input_length", default=256, type=int, required=True)
    parser.add_argument("--max_target_length", default=256, type=int, required=True)

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to save predicted labels.")
    parser.add_argument("--do_tune", action="store_true", help="Whether to tuning threshold.")

    parser.add_argument("--text_field", default="metadata", type=str, help="Whether to save predicted labels.")
    parser.add_argument("--fold", default=0, type=int, help="The fold of test collection.")
    parser.add_argument("--threshold", default=-1.0, type=float, help="The threshold of predicting loss.")

    # Other parameters
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--beam_search_size", default=4, type=int)
    parser.add_argument("--no_repeat_ngram_size", default=2, type=int)

    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer."
                        )
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training."
                        )
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some."
                        )

    args = parser.parse_args()

    train_dataset, valid_dataset, test_dataset = data_process.get_data(args.fold)
    if args.do_train and os.path.exists(args.output_dir + f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/') and os.listdir(
            args.output_dir + f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/'):
        raise ValueError(
            f'Output directory ({args.output_dir}{args.text_field}/fold{args.fold}/) already exists and is not empty.')
    if not os.path.exists(args.output_dir + f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/'):
        os.mkdir(args.output_dir + f'{args.text_field}/fold{args.fold}/bs_{args.batch_size}_lr_{args.learning_rate}/')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(args.device)
    # Training
    if args.do_train:
        # Set seed
        seed_everything(args.seed)
        train(args, train_dataset, valid_dataset, model, tokenizer)
    # Testing
    save_weights = [file for file in os.listdir(args.output_dir + f'{args.text_field}/fold{args.fold}/best/') if
                    file.endswith('.bin')]
    if args.do_test:
        test(args, test_dataset, model, tokenizer, save_weights)
    # Predicting
    if args.do_predict:
        for save_weight in save_weights:
            logger.info(f'loading weights from {save_weight}...')
            model.load_state_dict(
                torch.load(os.path.join(args.output_dir, f'{args.text_field}/fold{args.fold}/best/', save_weight)))
            logger.info(f'predicting labels of {save_weight}...')

            results = []
            predict_dataset = data_process.get_unannotated_dataset(args.text_field)
            model.eval()
            for s_idx in tqdm(range(len(predict_dataset))):
                sample = predict_dataset[s_idx]
                pred_summ = predict(args, sample[args.text_field], model, tokenizer)
                loss = compute_loss(args, sample[args.text_field], pred_summ, model, tokenizer)
                if args.threshold > 0 and loss > args.threshold:
                    continue
                results.append({
                    "dataset_id": sample['dataset_id'],
                    # "text": sample[args.text_field],
                    "prediction": pred_summ
                })
            logger.info(f"complete predicting, num results: {len(results)}")
            with open(os.path.join(args.output_dir,
                                   f'{args.text_field}/fold{args.fold}/best/' + f'{args.text_field}_fold{args.fold}_unannotated_pred.json'),
                                  'wt',
                                  encoding='utf-8') as f:
                f.write(json.dumps(results, ensure_ascii=False) + '\n')

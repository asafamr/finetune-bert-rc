import logging
import os
import random
import glob
import json
import numpy as np
import re
import torch
import fire
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from pytorch_transformers import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)
from rc_transformer import RCTransformer
from datautils import DataProcessor, stratified_sample_examples


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tens2list(tensor):
    return tensor.detach().cpu().tolist()


def train(train_dataset, model, fp16, n_gpus, n_epochs, learning_rate, gradient_accumulation_steps,
          per_gpu_train_batch_size, device, warmup_steps, max_grad_norm, callbacks, nopbar=False,
          layer_train_mask=None):
    """ Train the model """
    # tb_writer = SummaryWriter()

    total_batchsize = per_gpu_train_batch_size * max(1, n_gpus)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=total_batchsize)

    total_steps = len(train_dataloader) // gradient_accumulation_steps * n_epochs

    def gen_trainable_params():
        if layer_train_mask is None:
            yield from model.named_parameters()
        else:
            for pname, pten in model.named_parameters():
                match = re.findall('transformer\\.layer\\.([0-9]*)\\.', pname)
                if match:
                    layer_num = int(match[0])
                    if layer_num in layer_train_mask:
                        yield pname, pten
                else:
                    yield pname, pten

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in gen_trainable_params() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in gen_trainable_params() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=ADAM_EPSILON)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=FP16_OPT_LEVEL)

        # multi-gpu training (should be after apex fp16 initialization)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", n_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                total_batchsize * gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_steps)

    global_step = 0
    accum_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(n_epochs), desc="Epoch", disable=nopbar)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=nopbar)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)

            # all_input_ids, all_input_mask, all_segment_ids, all_label_ids, spans
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3],
                      'entstarts': batch[4].cpu(),
                      }
            loss = model(**inputs)

            if n_gpus > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            accum_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                for callback_mod, cb in callbacks:
                    if global_step % callback_mod == 0:
                        cb(global_step, model, optimizer, scheduler, accum_loss)  # raises exceptions for break

    return global_step, accum_loss / global_step


def evaluate(dataset, batch_size, model, device, scoring_cb, nopbar=False, score_bool_probs=False):
    # dataset, examples, features = load_and_cache_examples(args, tokenizer, processor, evaluate=True,
    #                                                       output_examples=True)

    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)
    all_results = []
    true_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=nopbar):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'entstarts': batch[4]
                      }

            logits = model(**inputs)
            if score_bool_probs:
                predicted = torch.softmax(logits, -1)[:, -1]
            else:
                predicted = logits.argmax(-1)

            all_results += tens2list(predicted)
            true_labels += tens2list(batch[3])

    return scoring_cb(all_results, true_labels)


def convert_examples_to_dataset(examples, label_list, max_seq_length,
                                tokenizer,
                                ):
    isxlnet = 'xlnet' in (type(tokenizer)).__name__.lower()
    label_map = {label: i for i, label in enumerate(label_list)}

    rows = []
    starte1, starte2 = tokenizer.convert_tokens_to_ids(['[E1]', '[E2]'])
    assert starte1 != starte2
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        with_markers = []
        curindex = 0
        for i, (start, end) in enumerate(example.entities_span):
            with_markers += example.text_tokens[curindex:start]
            with_markers.append(f'[E{i+1}]')
            with_markers += example.text_tokens[start:end]
            with_markers.append(f'[/E{i+1}]')
            curindex = end
        with_markers += example.text_tokens[curindex:]
        input_ids = tokenizer.encode(' '.join(with_markers), add_special_tokens=True)
        ent_starts = (
            min(input_ids.index(starte1), max_seq_length - 1), min(input_ids.index(starte2), max_seq_length - 1))
        if len(input_ids) > max_seq_length:
            logger.warning(f'trimming a {len(input_ids)} long sequence to {max_seq_length} (stupidly)  ')
            if isxlnet:
                input_ids = input_ids[-max_seq_length:]
            else:
                input_ids = input_ids[:max_seq_length]  # ostrich alg. these

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

        padding_length = max_seq_length - len(input_ids)
        if isxlnet:  # pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            segment_ids = ([4] * padding_length) + segment_ids
            segment_ids[-1] = 2  # cls seg id is 2
            ent_starts = tuple(x + padding_length for x in ent_starts)
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        label_id = label_map[example.label]
        if ex_index < 5:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % (example.guid))
            logger.debug("text: %s" % (' '.join(with_markers)))
            logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.debug("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.debug("label: %s (id = %d)" % (example.label, label_id))

        rows.append((input_ids, input_mask, ent_starts[0], ent_starts[1], segment_ids, label_id))

    all_input_ids, all_input_mask, ent1_starts, ent2_starts, all_segment_ids, all_label_ids = zip(*rows)

    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
    ent1_starts = torch.tensor(ent1_starts, dtype=torch.long)
    ent2_starts = torch.tensor(ent2_starts, dtype=torch.long)
    spans = torch.cat([ent1_starts.unsqueeze(1), ent2_starts.unsqueeze(1)], 1)
    all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
    all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)

    label_counts = torch.bincount(all_label_ids)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, spans)

    return dataset, label_counts


from pytorch_transformers import BertTokenizer, XLNetTokenizer, RobertaTokenizer, BertModel, XLNetModel, RobertaModel


def get_base_model_and_tokenizer(modeltype):
    conversion = dict(bert=('bert-large-cased', BertTokenizer, BertModel),
                      xlnet=('xlnet-large-cased', XLNetTokenizer, XLNetModel),
                      roberta=('roberta-large', RobertaTokenizer, RobertaModel))
    if modeltype not in conversion:
        raise Exception(f'--base_model should be one of these: {", ".join(conversion.keys())}')
    model_name, TokenizerClass, ModelClass = conversion[modeltype]
    tokenizer = TokenizerClass.from_pretrained(model_name, do_lower_case=False, cache_dir='./hugfacecache')
    model = ModelClass.from_pretrained(model_name, cache_dir='./hugfacecache')
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
FP16_OPT_LEVEL = 'O1'


def finetune(data_dir: str, output_dir: str, do_train: bool = True, model_name: str = 'roberta', task: str = 'tacred',
             max_seq_length: int = 300,
             evaluate_during_training: bool = True, eval_test: bool = False, per_gpu_batch_size: int = 2,
             learning_rate: float = 2e-5,
             gradient_accumulation_steps: int = 4, num_train_epochs: int = 100, seed: int = 42,
             stratify_train_k: int = -1,
             overwrite_output_dir: bool = False,
             logging_steps: int = 100, save_steps: int = 500, no_cuda: bool = False, warmup_steps: object = 300,
             fp16: bool = False, stop_train_low_score_k=5,
             verbose_logging: bool = True, class_weight_by_trainset=True,
             nopbar: bool = False, score_bool_probs=False, layer_train_mask=None,
             **additional: dict) -> list:
    """

    """
    all_params = dict(locals())

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG if verbose_logging else logging.INFO)

    device = 'cpu'
    n_gpu = 0
    if torch.cuda.is_available() and not no_cuda:
        n_gpu = torch.cuda.device_count()
        device = 'cuda:0'
    device = torch.device(device)

    logger.warning("Device: %s, n_gpu: %s,16-bits training: %s",
                   device, n_gpu, fp16)

    set_seed(seed)

    model, tokenizer = get_base_model_and_tokenizer(model_name)

    if task not in DataProcessor.get_task_mapping():
        raise Exception('Unknown task doesn\'t have a matching task processor: ' + task)
    task_processor = DataProcessor.get_task_mapping()[task](data_dir=data_dir,
                                                            **dict((k.replace('task_', ''), v) for k, v in
                                                                   additional.items() if k.startswith('task_')))

    model = RCTransformer(model, len(task_processor.get_labels()))
    model.to(device)

    logger.info("Training/evaluation parameters %s", all_params)

    if do_train and os.path.exists(output_dir) and os.listdir(output_dir) and not overwrite_output_dir:
        raise ValueError(f"Output directory ({output_dir}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.")
    elif not do_train:
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            raise ValueError(f"Evaluating an empty output directory ({output_dir}) ")

    os.makedirs(output_dir, exist_ok=True)

    train_examples = None
    train_ds = None

    eval_examples = None
    eval_ds = None

    if do_train:
        all_dev_results = []

        class StopTrainingExcpetion(Exception):
            pass

        train_examples = task_processor.get_train_examples()
        if stratify_train_k:
            train_examples = stratified_sample_examples(train_examples, stratify_train_k, seed)
        train_ds, class_counts = convert_examples_to_dataset(train_examples, task_processor.get_labels(),
                                                             max_seq_length, tokenizer)
        if class_weight_by_trainset:
            with torch.no_grad():
                class_counts = class_counts.to(torch.float) + 1.0
                total_size = torch.sum(class_counts)
                class_weights = (total_size / class_counts)
                class_weights /= torch.mean(class_weights)  # set mean class weight to 1
                model.set_class_weights(class_weights.to(device))

        if evaluate_during_training:
            eval_examples = task_processor.get_dev_examples()
            eval_ds, _ = convert_examples_to_dataset(eval_examples, task_processor.get_labels(), max_seq_length,
                                                     tokenizer)
        train_callbacks = []
        tb_writer = None
        if logging_steps > 0:
            all_main_scores = []
            tb_writer = SummaryWriter()
            last_loss = [0]

            def logging_cb(global_step, model, optimizer, scheduler, accum_loss):

                if evaluate_during_training:

                    results = evaluate(eval_ds, per_gpu_batch_size * n_gpu * 2, model, device, task_processor.score,
                                       nopbar=nopbar, score_bool_probs=score_bool_probs)
                    main_score = results['main']
                    best_main_score = max(all_main_scores + [0])
                    all_main_scores.append(main_score)
                    logging.info(f'Results step {global_step}:\n {results}')
                    if main_score > best_main_score:
                        logging.info(f'best score so far: {main_score}. saving model...')
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model
                        torch.save({'args': all_params,
                                    'tokenizer': tokenizer,
                                    'state_dict': model_to_save.state_dict(),
                                    'loss': accum_loss,
                                    'global_step': global_step
                                    }, os.path.join(output_dir, f'best_dev_model.bin'))
                    else:
                        past_k_results_no_record_broken = (len(all_main_scores) > stop_train_low_score_k > 0 and
                                                           best_main_score > max(
                                    all_main_scores[-stop_train_low_score_k:]))
                        stagnation = (len(all_main_scores) > stop_train_low_score_k > 3 and
                                      np.std(all_main_scores[-stop_train_low_score_k:]) < 0.025
                                      )
                        if past_k_results_no_record_broken or stagnation:
                            all_dev_results.append((global_step, results))
                            logging.info(
                                f'Looks like training has converged norec: {past_k_results_no_record_broken} '
                                f'stagnation:{stagnation}. shutting down...')
                            with open(os.path.join(output_dir, 'results.json'), 'w') as fout:
                                json.dump(all_dev_results, fout)
                            raise StopTrainingExcpetion()

                    all_dev_results.append((global_step, results))
                    for key, value in results['agg'].items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                mean_loss = (accum_loss - last_loss[0]) / logging_steps
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar('loss', mean_loss, global_step)
                last_loss[0] = accum_loss

            train_callbacks.append((logging_steps, logging_cb))

        if save_steps > 0:
            def save_cb(global_step, model, optimizer, scheduler, accum_loss):
                output_dir_cp = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
                os.makedirs(output_dir_cp, exist_ok=True)
                model_to_save = model.module if hasattr(model,
                                                        'module') else model  # Take care of distributed/parallel training

                torch.save({'args': all_params,
                            'tokenizer': tokenizer,
                            'state_dict': model_to_save.state_dict(),
                            'loss': accum_loss,
                            'global_step': global_step
                            }, os.path.join(output_dir_cp, f'cp_model.bin'))

            train_callbacks.append((save_steps, save_cb))

        try:
            train(train_ds, model, fp16, n_gpu, num_train_epochs, learning_rate, gradient_accumulation_steps,
                  per_gpu_batch_size, device, warmup_steps, MAX_GRAD_NORM, train_callbacks, nopbar, layer_train_mask)
        except StopTrainingExcpetion as e:
            logger.info('stopped training')

        if eval_test:
            # eval best model on testset
            eval_examples = task_processor.get_test_examples()
            eval_ds, _ = convert_examples_to_dataset(eval_examples, task_processor.get_labels(), max_seq_length,
                                                     tokenizer)
            best_model_path = os.path.join(output_dir, 'best_dev_model.bin')
            assert os.path.isfile(best_model_path), 'Best model checkpoint missing...'
            checkpoint_saved = torch.load(best_model_path)
            model.load_state_dict(checkpoint_saved['state_dict'])

            results = evaluate(eval_ds, per_gpu_batch_size * n_gpu * 2, model, device, task_processor.score,
                               nopbar=nopbar)
            logger.info(f'test final result: {results}')

            with open(os.path.join(output_dir, 'metrics.json'), 'w') as fout:
                json.dump(dict(results), fout)
            return results

        return all_dev_results

    else:  # evaluate
        if eval_test:
            eval_examples = task_processor.get_test_examples()
        else:
            eval_examples = task_processor.get_dev_examples()

        eval_ds, _ = convert_examples_to_dataset(eval_examples, task_processor.get_labels(), max_seq_length,
                                                 tokenizer)

        # checkpoints = list(
        #     c for c in sorted(glob.glob(output_dir + '/**/*.bin', recursive=True)))
        checkpoints = [os.path.join(output_dir, f'best_dev_model.bin')]
        all_results = []
        for checkpoint in checkpoints:
            checkpoint_saved = torch.load(checkpoint)
            checkpoint_step = 'final'
            matched_steps = re.findall('checkpoint-([0-9]*)', checkpoint)
            if matched_steps:
                checkpoint_step = matched_steps[0]

            model.load_state_dict(checkpoint_saved['state_dict'])
            model.to(device)
            results = evaluate(eval_ds, per_gpu_batch_size * n_gpu * 2, model, device, task_processor.score,
                               nopbar=nopbar)
            logger.info(f'finished evaluation of step {checkpoint_step}: {results}')
            all_results.append((checkpoint_step, results))
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as fout:
            json.dump(dict(results), fout, indent=2, sort_keys=True)

        logger.info("Results: {}".format(results))
        return all_results


if __name__ == "__main__":
    fire.Fire(finetune)
    # finetune('tacred/data/json', 'tempout', overwrite_output_dir=True, task='tacredbool', num_train_epochs=50,
    #          stop_train_low_score_k=10, model_name='xlnet', score_bool_probs=True, per_gpu_batch_size=8,
    #          max_seq_length=220, #no_cuda=True, fp16=False,
    #          layer_train_mask=[23],
    #          evaluate_during_training=True, logging_steps=50, eval_test=True, save_steps=-1,
    #          task_target_rel='org:top_members/employees', task_seed=2001, task_num_samps_pos_train=50,
    #          task_num_samps_neg_train=200,  # task_test_pos_size=200,
    #          task_p_train=0.8)
    # finetune('tacred/data/json-balanceddevtest', 'tempout', overwrite_output_dir=True, task='tacred', num_train_epochs=200,
    #          stop_train_low_score_k=10, model_name='xlnet',  per_gpu_batch_size=8,
    #          max_seq_length=220,fp16=True,stratify_train_k=30,
    #          layer_train_mask=[23],
    #          evaluate_during_training=True, logging_steps=50, eval_test=True, save_steps=-1,
    #          )

import os
from datautils import DataProcessor, InputExample
from transformers_rc_finetune import finetune
from multiprocessing import Process, Queue
from sklearn.metrics import roc_auc_score
import re
import sys

results_queue = Queue()


class TextProcessor(DataProcessor):
    def __init__(self, train_pos_text='', train_neg_text='', dev_pos_text='', dev_neg_text='', test_pos_text='',
                 test_neg_text=''):
        self.train_pos_text = train_pos_text
        self.train_neg_text = train_neg_text
        self.dev_pos_text = dev_pos_text
        self.dev_neg_text = dev_neg_text
        self.test_pos_text = test_pos_text
        self.test_neg_text = test_neg_text
        self.guid_counter = 10000

    def get_train_examples(self):
        return self.parse_text_to_samples(self.train_pos_text, 'yes_relation') + self.parse_text_to_samples(
            self.train_neg_text, 'no_relation')

    def get_dev_examples(self):
        return self.parse_text_to_samples(self.dev_pos_text, 'yes_relation') + self.parse_text_to_samples(
            self.dev_neg_text, 'no_relation')

    def get_test_examples(self):
        return self.parse_text_to_samples(self.test_pos_text, 'yes_relation') + self.parse_text_to_samples(
            self.test_neg_text, 'no_relation')

    def parse_text_to_samples(self, txt, label='no_relation'):
        rows = txt.split('\n')
        rows = [x.strip() for x in rows]
        rows = [x for x in rows if x and x[0] != '#']  # remove comments and empty lines
        input_examples = []
        for row in rows:

            # switch the type position [o:x John] -> [ John o] to make ltr parsing easier later
            rawtokens = re.sub('\\[\\s?([a-zA-Z])[^\\s]*([^\\]]*)\\]', ' [ \\2 \\1] ',
                               row).split()
            clean_tokens = []
            spans = {}
            last_brackets_pos = None
            for x in rawtokens:
                if x == '[':
                    last_brackets_pos = len(clean_tokens)
                elif x.endswith(']'):

                    spans[x[-2].lower()] = (last_brackets_pos, len(clean_tokens))
                else:
                    clean_tokens.append(x)
            assert 's' in spans and 'o' in spans, 'sentence ' + row + ' does not contain subject/object'
            self.guid_counter += 1
            input_examples.append(
                InputExample(guid=self.guid_counter, text_tokens=clean_tokens, entities_span=[spans['s'], spans['o']],
                             label=label))
        return input_examples

    def score(self, predicted, gold_labels):
        auc = roc_auc_score(gold_labels, predicted)
        return {'agg': dict(auc=auc), 'main': auc}

    # this is used later - taskname param
    @staticmethod
    def taskname():
        return 'boolrc'

    def get_labels(self):
        """See base class."""
        return ['no_relation',
                'yes_relation']


def train_roberta(model_name, positive_train, negative_train, positive_dev, negative_dev, gpu=0, verbose=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # these control over/under fitting
    max_num_epochs = 200
    stop_when_no_improvements_on_dev_for_k_evals = 5
    eval_dev_every_k_batches = 100

    # these are mainly for controlling cuda mem
    batch_size = 2
    accumelate_k_batches = 4
    # could also speed up things on gce
    use_fp16 = False

    def run(should_mute):
        if should_mute:
            sys.stdout = open(os.devnull, 'w')
        finetune(data_dir='.', output_dir=model_name, do_train=True, task='boolrc', nopbar=True,
                 overwrite_output_dir=True,
                 eval_test=False, score_bool_probs=True,
                 logging_steps=eval_dev_every_k_batches,
                 stop_train_low_score_k=stop_when_no_improvements_on_dev_for_k_evals, fp16=use_fp16,
                 num_train_epochs=max_num_epochs, per_gpu_batch_size=batch_size,
                 gradient_accumulation_steps=accumelate_k_batches,

                 task_train_pos_text=positive_train,
                 task_train_neg_text=negative_train,
                 task_dev_pos_text=positive_dev,
                 task_dev_neg_text=negative_dev,

                 )

    p = Process(target=run, kwargs={'should_mute': not verbose})
    p.start()
    p.join()


def eval_roberta(model_name, positive_test, negative_test, gpu=0, verbose=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    def run(should_mute):
        if should_mute:
            sys.stdout = open(os.devnull, 'w')
        finetune(data_dir='..', output_dir=model_name, do_train=False, task='boolrc', nopbar=True, eval_test=True,
                 score_bool_probs=True,

                 task_test_pos_text=positive_test,
                 task_test_neg_text=negative_test,

                 )

    p = Process(target=run, kwargs={'should_mute': not verbose})
    p.start()
    p.join()


if __name__ == '__main__':
    train_roberta('test_123',
                  '''
[o John] founded [s Unilever].
[o Dan] established [s Microsoft Corporation ] .
[o Laura], together with her co-founding friend Jill started the [s company] .
[o Tammy] is [s Osem] 's co-founder.
[o Bobby McGee] started [s Virgin records] in his back garage.
[o Bobby McGee] started [s Virgin records] in his back garage.

                    ''',
                  '''
[o Dan Smith] worked at [s Bell labs]. 
                  ''',
                  '''
[o John] is a co-founder of [s Disney]. 
                  ''',
                  '''
[o John] was the CEO [s microsoft]. 
                  ''',
                  )

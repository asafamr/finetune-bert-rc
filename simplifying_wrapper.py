import os
from datautils import DataProcessor, InputExample
from transformers_rc_finetune import finetune
from multiprocessing import Process
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
import sys
import logging
import pandas as pd


class TextProcessor(DataProcessor):
    # kinda ugly but let me keep the tested old code intact
    last_prediction = []
    guid_counter = 10000

    def __init__(self, data_dir, train_pos_text='', train_neg_text='', dev_pos_text='', dev_neg_text='',
                 test_pos_text='',
                 test_neg_text=''):
        self.train_pos_text = train_pos_text
        self.train_neg_text = train_neg_text
        self.dev_pos_text = dev_pos_text
        self.dev_neg_text = dev_neg_text
        self.test_pos_text = test_pos_text
        self.test_neg_text = test_neg_text

    def get_train_examples(self):
        return self.parse_text_to_samples(self.train_pos_text, 'yes_relation') + TextProcessor.parse_text_to_samples(
            self.train_neg_text, 'no_relation')

    def get_dev_examples(self):
        return self.parse_text_to_samples(self.dev_pos_text, 'yes_relation') + TextProcessor.parse_text_to_samples(
            self.dev_neg_text, 'no_relation')

    def get_test_examples(self):
        return self.parse_text_to_samples(self.test_pos_text, 'yes_relation') + TextProcessor.parse_text_to_samples(
            self.test_neg_text, 'no_relation')

    @staticmethod
    def parse_text_to_samples(txt, label='no_relation'):
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
            TextProcessor.guid_counter += 1
            input_examples.append(
                InputExample(guid=TextProcessor.guid_counter, text_tokens=clean_tokens,
                             entities_span=[spans['s'], spans['o']],
                             label=label))
        return input_examples

    def score(self, predicted, gold_labels):
        TextProcessor.last_prediction = predicted
        acc = accuracy_score(gold_labels, predicted)
        return {'agg': dict(acc=acc), 'main': acc}

    # this is used later - taskname param
    @staticmethod
    def taskname():
        return 'boolrc'

    def get_labels(self):
        """See base class."""
        return ['no_relation',
                'yes_relation']


def train_roberta(model_name, positive_train, negative_train, positive_dev, negative_dev, gpu=0, verbose=False):
    print('Training(this should take a couple of minutes)...')
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

    def run():
        finetune(data_dir='.', output_dir=model_name, do_train=True, task='boolrc', nopbar=True,
                 overwrite_output_dir=True,
                 eval_test=False, score_bool_probs=False,
                 logging_steps=eval_dev_every_k_batches,
                 stop_train_low_score_k=stop_when_no_improvements_on_dev_for_k_evals, fp16=use_fp16,
                 num_train_epochs=max_num_epochs, per_gpu_batch_size=batch_size,
                 gradient_accumulation_steps=accumelate_k_batches,

                 task_train_pos_text=positive_train,
                 task_train_neg_text=negative_train,
                 task_dev_pos_text=positive_dev,
                 task_dev_neg_text=negative_dev,

                 )

    p = Process(target=run)
    p.start()
    p.join()


def _print_report(test_pos_text, test_neg_text, predicted):
    txts = []
    gold = []
    rows = test_pos_text.split('\n')
    rows = [x.strip() for x in rows]
    rows = [x for x in rows if x and x[0] != '#']
    txts += rows
    gold += [1] * len(rows)

    rows = test_neg_text.split('\n')
    rows = [x.strip() for x in rows]
    rows = [x for x in rows if x and x[0] != '#']
    txts += rows
    gold += [0] * len(rows)

    df = pd.DataFrame(dict(txt=txts, gold=gold, pred=predicted))
    df['pred'] = predicted
    print('RC Report:')
    print('-' * 20)
    print()
    print('Done right:')
    print('-' * 10)
    print('True Positives: ')
    for t in df.query('gold == 1 and pred ==1').txt:
        print(t)
    print()
    print('True Negatives: ')
    for t in df.query('gold == 0 and pred ==0').txt:
        print(t)
    print()
    print()
    print('Mistakes:')
    print('-' * 10)
    print('False Positives(relation wrongfully predicted): ')
    for t in df.query('gold == 0 and pred == 1').txt:
        print(t)
    print()
    print('False Negatives(missed relation):')
    for t in df.query('gold == 1 and pred == 0').txt:
        print(t)

    print()
    print('Aggregated scores:')
    print('-' * 10)
    print(f'Accuracy:{accuracy_score(gold, predicted)}')
    print(f'Precision:{precision_score(gold, predicted)}')
    print(f'Recall:{recall_score(gold, predicted)}')
    print(f'F1:{f1_score(gold, predicted)}')
    print('-' * 10)


def eval_roberta(model_name, positive_test, negative_test, gpu=0, verbose=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


    def run():
        finetune(data_dir='..', output_dir=model_name, do_train=False, task='boolrc', nopbar=True, eval_test=True,
                 score_bool_probs=False,

                 task_test_pos_text=positive_test,
                 task_test_neg_text=negative_test,

                 )
        _print_report(positive_test, negative_test, TextProcessor.last_prediction)

    p = Process(target=run)
    p.start()
    p.join()


if __name__ == '__main__':
    eval_roberta('test_123', '''
[o James] established and headed [s Kayak ] .
[o Bob billy Joe] is a co-founder of the [s company] .

Five years later [s Waze], that was established by [o Mark] , was disbanded.



''', '''
[o Bob billy Joe] is a chairman of the [s company] .
[o Dan] established Microsoft Corporation but not [s Mazda] .
[o Laura], that I found funny,worked at the [s company] .

''',verbose=True)
#     train_roberta('test_123',
#                   '''
# [o John] founded [s Unilever].
# [o Dan] established [s Microsoft Corporation ] .
# [o Laura], together with her co-founding friend Jill started the [s company] .
# [o Tammy] is [s Osem] 's co-founder.
# [o Bobby McGee] started [s Virgin records] in his garage.
# [o Sementha] is the creator of [s Whatever inc.] and its CEO.
#
#                     ''',
#                   '''
# [o John] drives trucks for [s Unilever].
# [o Dan] established Microsoft Corporation  but only after he graduated from [s MIT].
# [o Laura] has never worked for [s Osem].
# [o Tammy] is [s Osem] 's  best customer.
# [o Bobby McGee] keeps his collection, which was bought from [s Virgin records], in his garage.
# [o Sementha] is the creator of the logo of [s Whatever inc.]
#
# [o John] [s founded]  Unilever.
#  Dan [o established] [s Microsoft Corporation ] .
# [o Laura], together with her co-founding [s friend ] Jill started the  company .
# Tammy [o is ] [s Osem] 's co-founder.
# Bobby McGee started [s Virgin records] in his [o garage.]
# Sementha is the creator [o of] [s Whatever inc.] and its CEO.
#                   ''',
#                   '''
# [o Bobby] is [s Disney]'s co-founder but never worked for Toyota .
# [o I] created a company and named it [s AI2D2] .
# [o Sammy] established [s Burekas] in the early 2000's.
#                   ''',
#                   '''
# [o Bobby] is a co-founder of Disney but never worked for [s  Toyota] .
# [o I] knew a company named [s AI2D2] .
#  Sammy [o established] [s Burekas] in the early 2000's.
# [s Sammy] established [o Burekas] in the early 2000's.
#
#                   ''',
#                   )

import os
from datautils import DataProcessor, InputExample
from transformers_rc_finetune import finetune
from multiprocessing import Process
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import multiprocessing
import re
import sys
import logging
import pandas as pd
import time, math


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


def set_verbose(verbose):
    # remove current loggers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if verbose:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    else:
        sys.stdout = open('/dev/null', 'w')
        sys.stderr = open('/dev/null', 'w')


def train_roberta(model_name, positive_train, negative_train, positive_dev, negative_dev, gpu=0, verbose=False):
    print('Training(this should take a couple of minutes)...')
    start_time = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for x in [positive_train, negative_train, positive_dev, negative_dev]:
        TextProcessor.parse_text_to_samples(x)  # this will throw now with parsing error

    # these control over/under fitting
    max_num_epochs = 200
    stop_when_no_improvements_on_dev_for_k_evals = 5
    eval_dev_every_k_batches = 100

    # these are mainly for controlling cuda mem
    batch_size = 3
    accumelate_k_batches = 4
    # could also speed up things on gce
    use_fp16 = True

    def run(verbose):
        set_verbose(verbose)
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

    p = Process(target=run, kwargs=dict(verbose=verbose))
    p.start()
    while True:
        p.join(timeout=5)
        if not p.is_alive():
            print()
            break
        if not verbose:
            print('.', end=' ')

    if p.exitcode != 0:
        print('FAILED. Total time: %d seconds' % (math.ceil(time.time() - start_time)))
    else:
        print('Done. Total time: %d seconds' % (math.ceil(time.time() - start_time)))
    return p.exitcode


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


def show_results_jup(results):
    import pandas as pd
    from IPython.display import display
    import qgrid
    from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

    df = pd.DataFrame(dict((x, results[x]) for x in results)).set_index('sentence')
    col_options = {
        'width': 80,
    }
    col_defs = {
        'sentence': {
            'width': 1000,
        }
    }
    for x in 'label prediction'.split():
        df[x] = df[x].astype('bool')

    df['correct'] = (df['label'] == df['prediction'])
    q=qgrid.show_grid(df, row_edit_callback=lambda x: False, column_options=col_options,
                    column_definitions=col_defs, show_toolbar=False, grid_options={'forceFitColumns': True})
    for x in accuracy_score,precision_score, recall_score, f1_score:
        score = x(df.label.values, df.prediction.values)
        print('%s: %.1f%%' % (x.__name__.replace('_score', '').capitalize().ljust(15), score * 100))
    display(q)

def eval_roberta(model_name, positive_test, negative_test, gpu=0, verbose=False):
    print('Evaluating...')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for x in [positive_test, negative_test]:
        TextProcessor.parse_text_to_samples(x)  # this will throw now with parsing error

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    def run(verbose, return_dict):
        set_verbose(verbose)
        finetune(data_dir='..', output_dir=model_name, do_train=False, task='boolrc', nopbar=True, eval_test=True,
                 score_bool_probs=False,

                 task_test_pos_text=positive_test,
                 task_test_neg_text=negative_test,

                 )
        rows_pos = [x.strip() for x in positive_test.split('\n') if x.strip() and x.strip()[1] != '#']
        rows_neg = [x.strip() for x in negative_test.split('\n') if x.strip() and x.strip()[1] != '#']
        return_dict['sentence'] = rows_pos + rows_neg
        return_dict['label'] = [1] * len(rows_pos) + [0] * len(rows_neg)
        return_dict['prediction'] = TextProcessor.last_prediction
        # _print_report(positive_test, negative_test, TextProcessor.last_prediction)

    p = Process(target=run, kwargs=dict(verbose=verbose, return_dict=return_dict))
    p.start()
    while True:
        p.join(timeout=5)
        if not p.is_alive():
            print()
            break
        if not verbose:
            print('.', end=' ')

    if p.exitcode != 0:
        print('FAILED')
    else:
        print('Done.')
    return p.exitcode, return_dict


if __name__ == '__main__':
    eval_roberta('test_123', '''
[o James] established and headed [s Kayak ] .
[o Bob billy Joe] is a co-founder of the [s company] .

Five years later [s Waze], that was established by [o Mark] , was disbanded.



''', '''
[o Bob billy Joe] is a chairman of the [s company] .
[o Dan] established Microsoft Corporation but not [s Mazda] .
[o Laura], that I found funny,worked at the [s company] .

''', verbose=True)
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

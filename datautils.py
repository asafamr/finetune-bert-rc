import logging
import os
import re
import pandas as pd
import numpy as np
import tempfile
import subprocess
from sklearn.metrics import roc_auc_score

from collections import Counter

logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, text_tokens, entities_span, label=None, **ignoredstuff):
        self.guid = guid
        self.text_tokens = text_tokens
        self.entities_span = entities_span
        self.label = label
    def __repr__(self):
        return 'EXAMPLE:'+str(self.__dict__)

# class InputFeatures(object):
#     def __init__(self, input_ids, input_mask, entity_starts, segment_ids, label_id):
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label_id = label_id
#         self.entity_starts = entity_starts


class DataProcessor():
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def score(self, predicted, gold_labels):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @staticmethod
    def taskname():
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @staticmethod
    def get_task_mapping():
        res = {}
        for subclass in DataProcessor.__subclasses__():
            res[subclass.taskname()] = subclass
        return res


class TACREDProcessor(DataProcessor):
    def __init__(self, data_dir, balance_dev_test_k=-1):
        self.data_dir = data_dir
        self.balance_dev_test_k = balance_dev_test_k
        # self.replace_dev_with_test=replace_dev_with_test

    def get_train_examples(self):
        return self.get_x_examples(os.path.join(self.data_dir, 'train.json'))

    def get_dev_examples(self):
        return self.get_x_examples(os.path.join(self.data_dir, 'dev.json'), self.balance_dev_test_k)

    def get_test_examples(self):
        return self.get_x_examples(os.path.join(self.data_dir, 'test.json'), self.balance_dev_test_k)

    @staticmethod
    def taskname():
        return 'tacred'

    def get_x_examples(self, json_path, balance_k=-1):
        assert os.path.isfile(json_path), "Dataset file doesn't exist"

        loaded = pd.read_json(json_path)[
            'id  token relation subj_end subj_start obj_start obj_end'.split()]

        if balance_k > 0:
            loaded = loaded.groupby(['relation'], group_keys=False).apply(
                lambda x: x.sample(n=min(balance_k, len(x)), random_state=1234))

        cleaningmap = {'-RRB-': ')', '-LRB-': '(', '-LSB-': '[',
                       '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
                       '&nbsp;': ' ', '&quot;': "'", '--': '-', '---': '-'}

        def clean_tokens(tokens):
            return [cleaningmap.get(x, x) for x in tokens]

        def row_to_example(row):
            return InputExample(guid=str(row['id']), text_tokens=clean_tokens(row.token),
                                entities_span=((row.obj_start, row.obj_end + 1), (row.subj_start, row.subj_end + 1)),
                                label=row.relation)

        return [row_to_example(row) for _, row in loaded.iterrows()]

    def score(self, predicted, gold_labels):

        NO_RELATION = self.get_labels().index('no_relation')

        correct_by_relation = Counter()  # true positives
        guessed_by_relation = Counter()  # pred relation count per rel
        gold_by_relation = Counter()  # gold relation count per rel

        # Loop over the data to compute a score
        for row in range(len(gold_labels)):
            gold = gold_labels[row]
            guess = predicted[row]

            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1

        prec_micro = 1.0
        if sum(guessed_by_relation.values()) > 0:
            prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
        recall_micro = 0.0
        if sum(gold_by_relation.values()) > 0:
            recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
        f1_micro = 0.0
        if prec_micro + recall_micro > 0.0:
            f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

        res = {'agg': {'p': 100.0 * prec_micro, 'r': 100.0 * recall_micro, 'f1': 100.0 * f1_micro},
               'main': 100.0 * f1_micro}

        for rel in gold_by_relation.keys():
            if rel == NO_RELATION:
                continue
            n_pred = guessed_by_relation.get(rel, 0)
            n_true_pos = correct_by_relation.get(rel, 0)
            n_gold = gold_by_relation.get(rel, 0)
            assert n_gold != 0
            if n_pred == 0:
                precision_rel = 1
                recall_rel = 0
            else:
                precision_rel = n_true_pos / n_pred
                recall_rel = n_true_pos / n_gold
            f1_rel = 2 * precision_rel * recall_rel / (precision_rel + recall_rel + 1e-8)

            rel = self.get_labels()[rel]
            res[rel] = {'p': 100.0 * precision_rel, 'r': 100.0 * recall_rel, 'f1': 100.0 * f1_rel}
        return res

    def get_labels(self):
        return ['no_relation', 'org:founded_by', 'per:employee_of',
                'org:alternate_names', 'per:cities_of_residence', 'per:children',
                'per:title', 'per:siblings', 'per:religion', 'per:age',
                'org:website', 'per:stateorprovinces_of_residence',
                'org:member_of', 'org:top_members/employees',
                'per:countries_of_residence', 'org:city_of_headquarters',
                'org:members', 'org:country_of_headquarters', 'per:spouse',
                'org:stateorprovince_of_headquarters',
                'org:number_of_employees/members', 'org:parents',
                'org:subsidiaries', 'per:origin',
                'org:political/religious_affiliation', 'per:other_family',
                'per:stateorprovince_of_birth', 'org:dissolved',
                'per:date_of_death', 'org:shareholders', 'per:alternate_names',
                'per:parents', 'per:schools_attended', 'per:cause_of_death',
                'per:city_of_death', 'per:stateorprovince_of_death', 'org:founded',
                'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth',
                'per:charges', 'per:country_of_death']


class TACREDBoolProcessor(DataProcessor):
    def __init__(self, data_dir, target_rel, seed, num_samps_pos_train, num_samps_neg_train, test_pos_size=200,
                 p_train=0.8):
        self.data_dir = data_dir
        self.target_rel = target_rel
        self.seed = seed

        df = pd.concat([self.get_x_examples(os.path.join(self.data_dir, 'train.json')),
                        self.get_x_examples(os.path.join(self.data_dir, 'dev.json'))])

        df = df.sample(len(df), random_state=1234)
        test_df = df.groupby('label', group_keys=False).apply(
            lambda x: x.iloc[:test_pos_size] if len(x) > test_pos_size else x.iloc[:0])
        train_dev_df = df.drop(test_df.index)

        additional = train_dev_df.iloc[:0]
        most_common_ners = test_df.groupby('label').agg(lambda x: x.value_counts().index[0])['ners'].to_dict()
        good_rels = []
        for rel, ners in most_common_ners.items():
            if rel == 'no_relation':
                continue
            current_negatives_with_same_ner = len(additional.query('label!=@rel and ners==@ners'))
            left_to_add = test_pos_size - current_negatives_with_same_ner
            if left_to_add <= 0:
                continue
            to_add = train_dev_df.query('label!=@rel and ners==@ners')
            if len(to_add) < left_to_add:
                continue
            good_rels.append(rel)
            to_add = to_add.sample(left_to_add, random_state=100)
            train_dev_df = train_dev_df.drop(to_add.index)
            additional = pd.concat([additional, to_add])

        test_df = pd.concat([additional, test_df]).reset_index()
        train_dev_df = train_dev_df.reset_index()

        final_good_rels = train_dev_df.query('label in @good_rels').groupby('label').count().query(
            'ners > 200').index.values.tolist()

        assert target_rel in final_good_rels, f'target relation must have enough samples, i.e. one of {final_good_rels}'

        # test does not change with seed
        common_ner = most_common_ners[target_rel]
        positives_test = test_df.query('label == @target_rel')
        negatives_test = test_df.query('label != @target_rel and ners == @common_ner').sample(test_pos_size,
                                                                                              random_state=1234)
        negatives_test['label'] = 'no_relation'

        rs = np.random.RandomState(self.seed)
        positives = train_dev_df.query('label == @target_rel').sample(num_samps_pos_train, random_state=rs)
        negatives = train_dev_df.query('label != @target_rel and ners == @common_ner').sample(num_samps_neg_train,
                                                                                              random_state=rs)
        negatives['label'] = 'no_relation'

        positives_train = positives.iloc[:int(num_samps_pos_train * p_train)]
        positives_dev = positives.iloc[int(num_samps_pos_train * p_train):]

        negatives_train = negatives.iloc[:int(num_samps_neg_train * p_train)]
        negatives_dev = negatives.iloc[int(num_samps_neg_train * p_train):]

        self.train = [InputExample(**x) for _, x in pd.concat([positives_train, negatives_train]).iterrows()]
        self.dev = [InputExample(**x) for _, x in pd.concat([positives_dev, negatives_dev]).iterrows()]
        self.test = [InputExample(**x) for _, x in pd.concat([positives_test, negatives_test]).iterrows()]

    def get_train_examples(self):
        return self.train

    def get_dev_examples(self):
        return self.dev

    def get_test_examples(self):
        return self.test

    @staticmethod
    def taskname():
        return 'tacredbool'

    def get_x_examples(self, json_path):
        assert os.path.isfile(json_path), "Dataset file doesn't exist"
        loaded = pd.read_json(json_path)[
            'id  token relation subj_end subj_start obj_start obj_end subj_type obj_type'.split()]

        cleaningmap = {'-RRB-': ')', '-LRB-': '(', '-LSB-': '[',
                       '-RSB-': ']', '-LCB-': '{', '-RCB-': '}',
                       '&nbsp;': ' ', '&quot;': "'", '--': '-', '---': '-'}

        def clean_tokens(tokens):
            return [cleaningmap.get(x, x) for x in tokens]

        rows = []
        for _, row in loaded.iterrows():
            rows.append((str(row['id']), clean_tokens(row.token), ((row.obj_start, row.obj_end + 1),
                                                                   (row.subj_start, row.subj_end + 1)), row.relation,
                         f'{row.subj_type},{row.obj_type}'))

        return pd.DataFrame(rows, columns='guid text_tokens entities_span label ners'.split()) \
            .set_index('guid', verify_integrity=True)

    def score(self, predicted, gold_labels):
        auc = roc_auc_score(gold_labels, predicted)
        return {'agg': dict(auc=auc), 'main': auc}

    def get_labels(self):
        return ['no_relation', self.target_rel]


class SemEvalProcessor(DataProcessor):

    def __init__(self, data_dir, split_seed=1234, train_p=0.90):
        self.seed = split_seed
        self.train_p = train_p
        self.data_dir = data_dir

    def get_train_examples(self):
        examples = self.get_x_examples(os.path.join(self.data_dir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT'))
        rs = np.random.RandomState(self.seed)
        mask = rs.rand(len(examples)) <= self.train_p
        return [x for x, y in zip(examples, mask) if y]

    def get_dev_examples(self):
        examples = self.get_x_examples(os.path.join(self.data_dir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT'))
        rs = np.random.RandomState(self.seed)
        mask = rs.rand(len(examples)) > self.train_p
        return [x for x, y in zip(examples, mask) if y]

    def get_test_examples(self):
        examples = self.get_x_examples(
            os.path.join(self.data_dir, 'SemEval2010_task8_testing_keys', 'TEST_FILE_FULL.TXT'))
        return examples

    def get_x_examples(self, json_path):
        assert os.path.isfile(json_path), "Dataset file doesn't exist"

        examples = []
        with open(json_path, 'r') as fin:
            try:
                while True:
                    r1 = next(fin)
                    label = next(fin).strip()
                    guid, sent = r1.split('\t')
                    guid = int(guid)
                    sent = sent.strip().strip('"').strip()
                    chunks = re.split('(</?e[12]>)', sent)
                    lene1 = len(chunks[chunks.index('<e1>') + 1:chunks.index('</e1>')][0].split())
                    lene2 = len(chunks[chunks.index('<e2>') + 1:chunks.index('</e2>')][0].split())
                    starte1 = len(
                        ' '.join([x for x in chunks[:chunks.index('<e1>')] if not re.match('</?e[12]>', x)]).split())
                    starte2 = len(
                        ' '.join([x for x in chunks[:chunks.index('<e2>')] if not re.match('</?e[12]>', x)]).split())
                    tokens = ' '.join([x for x in chunks if not re.match('</?e[12]>', x)]).split()
                    examples.append(InputExample(guid=str(guid), text_tokens=tokens,
                                                 entities_span=[(starte1, starte1 + lene1), (starte2, starte2 + lene2)],
                                                 label=label))
                    next(fin)
                    next(fin)
            except StopIteration as e:
                pass
        return examples

    def score(self, predicted, gold_labels):
        labels = self.get_labels()
        predicted = [labels[x] for x in predicted]
        gold_labels = [labels[x] for x in gold_labels]
        scorer_path = os.path.join(self.data_dir, 'SemEval2010_task8_scorer-v1.2', 'semeval2010_task8_scorer-v1.2.pl')
        with tempfile.NamedTemporaryFile('wt') as pred_file, tempfile.NamedTemporaryFile('wt') as gold_file:
            for i, x in enumerate(predicted):
                pred_file.write(f'{i+1}\t{x}\n')
            for i, x in enumerate(gold_labels):
                gold_file.write(f'{i+1}\t{x}\n')
            gold_file.flush()
            pred_file.flush()
            command = ['perl', scorer_path,
                       pred_file.name, gold_file.name]
            res = subprocess.check_output(command).decode('utf-8')
            parts = \
                res.split('<<< (9+1)-WAY EVALUATION TAKING DIRECTIONALITY INTO ACCOUNT -- OFFICIAL >>>:')[1].split(
                    'MACRO-averaged result (excluding Other):')[1].split()
            p = float(parts[2].replace('%', ''))
            r = float(parts[5].replace('%', ''))
            f1 = float(parts[8].replace('%', ''))
        return {'agg': dict(p=p, r=r, f1=f1), 'main': f1}

    @staticmethod
    def taskname():
        return 'semeval'

    def get_labels(self):
        """See base class."""
        return ['Message-Topic(e1,e2)',
                'Component-Whole(e2,e1)',
                'Message-Topic(e2,e1)',
                'Product-Producer(e1,e2)',
                'Cause-Effect(e1,e2)',
                'Content-Container(e1,e2)',
                'Other',
                'Member-Collection(e2,e1)',
                'Entity-Origin(e1,e2)',
                'Member-Collection(e1,e2)',
                'Component-Whole(e1,e2)',
                'Content-Container(e2,e1)',
                'Instrument-Agency(e1,e2)',
                'Entity-Destination(e2,e1)',
                'Instrument-Agency(e2,e1)',
                'Product-Producer(e2,e1)',
                'Entity-Destination(e1,e2)',
                'Cause-Effect(e2,e1)',
                'Entity-Origin(e2,e1)']


def stratified_sample_examples(examples, n, seed):
    if n <= 0:
        return examples
    rs = np.random.RandomState(seed)
    df = pd.DataFrame(vars(x) for x in examples)
    sample = df.groupby(['label'], group_keys=False).apply(lambda x: x.sample(n=min(n, len(x)), random_state=rs))
    return [InputExample(**x) for _, x in sample.iterrows()]


if __name__ == '__main__':
    # from pytorch_transformers import BertTokenizer
    #
    # bt = BertTokenizer.from_pretrained('bert-large-cased')
    # tacredreader = TACREDProcessor()
    # bt.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    # x = tacredreader.get_train_examples('/home/nlp/asafam/ai2/tacred/data/json')
    # x2 = convert_examples_to_features(x, tacredreader.get_labels(), 500, bt)
    # df.groupby('label', group_keys=False).apply(lambda x: x.sample(n=min(n, len(x)), random_state=rs))
    # print(vars(x2[0]))
    pass

    # frp = FewRelProcessor()

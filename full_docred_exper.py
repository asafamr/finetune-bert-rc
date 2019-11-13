import pandas as pd
import json
from collections import defaultdict
import fire
from simplifying_wrapper import train_roberta, eval_roberta, show_results_jup

with open('../DocRED/dev.json', 'rb') as fin:
    docredtest = json.load(fin)

with open('../DocRED/train_annotated.json', 'rb') as fin:
    docred = json.load(fin)


def get_relation_desc(relid):
    txt = '''P31	instance of
    P577	publication date
    P17	country
    P131	located in the administrative territorial entity
    P50	author
    P569	date of birth
    P27	country of citizenship
    P171	parent taxon
    P279	subclass of
    P19	place of birth
    P580	start time
    P570	date of death
    P361	part of
    P585	point in time
    P571	inception
    P276	location
    P582	end time
    P69	educated at
    P495	country of origin
    P136	genre
    P20	place of death
    P1001	applies to jurisdiction
    P155	follows
    P156	followed by
    P1412	languages spoken, written or signed
    P170	creator
    P39	position held
    P108	employer
    P527	has part
    P54	member of sports team
    P166	award received
    P123	publisher
    P175	performer
    P102	member of political party
    P137	operator
    P127	owned by
    P364	original language of performance work
    P22	father
    P57	director
    P264	record label
    P159	headquarters location
    P40	child
    P140	religion
    P1344	participant of
    P937	work location
    P463	member of
    P161	cast member
    P26	spouse
    P607	conflict
    P86	composer
    P179	part of the series
    P162	producer
    P576	dissolved, abolished or demolished
    P58	screenwriter
    P172	ethnic group
    P3373	sibling
    P36	capital
    P1366	replaced by
    P403	mouth of the watercourse
    P1365	replaces
    P749	parent organization
    P176	manufacturer
    P150	contains administrative territorial entity
    P551	residence
    P272	production company
    P241	military branch
    P25	mother
    P1376	capital of
    P1441	present in work
    P840	narrative location
    P30	continent
    P710	participant
    P449	original network
    P706	located on terrain feature
    P118	league
    P206	located in or next to body of water
    P400	platform
    P740	location of formation
    P112	founded by
    P800	notable work
    P178	developer
    P6	head of government
    P205	basin country
    P355	subsidiary
    P676	lyrics by
    P190	twinned administrative body
    P1056	product or material produced
    P674	characters
    P488	chairperson
    P37	official language
    P737	influenced by
    P194	legislative body
    P35	head of state
    P1336	territory claimed by
    P807	separated from
    P1198	unemployment rate'''
    relmap = dict(x.strip().split('\t') for x in txt.split('\n'))
    return relmap.get(relid, 'UNK')


def get_pairs(arr, bidir=True):
    arr = list(arr)
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            yield arr[i], arr[j]
            if bidir:
                yield arr[j], arr[i]


def get_rows(docs):
    all_rows = []
    docs_sents_tokens = {}
    for idoc, doc in enumerate(docs):

        vertices_by_isent = defaultdict(set)
        for ivert, vert in enumerate(doc['vertexSet']):
            for chunk in vert:
                vertices_by_isent[chunk['sent_id']].add((ivert, tuple(chunk['pos']), chunk['type']))
        for isent, sent in enumerate(doc['sents']):
            docs_sents_tokens[(idoc, isent)] = sent
            positives = {}
            for label in doc['labels']:
                head = label['h']
                tail = label['t']
                relation = label['r']
                head_verts = [x[1] for x in vertices_by_isent[isent] if x[0] == head]
                tail_verts = [x[1] for x in vertices_by_isent[isent] if x[0] == tail]
                if head_verts and tail_verts:
                    for h in head_verts:
                        for t in tail_verts:
                            positives[tuple(h), tuple(t)] = relation
            for v1, v2 in get_pairs(vertices_by_isent[isent]):

                v1span, v2span = v1[1], v2[1]
                #                     if v1span[0]>=v2span[0] and v1span[1]<=v2span[1] or v1span[0]<=v2span[0] and v1span[1]>=v2span[1]:
                # #                         print('overlap')
                #                         continue
                if v2span[1] > v1span[0] >= v2span[0] or v2span[1] > v1span[1] - 1 >= v2span[0] or \
                        v1span[1] > v2span[0] >= v1span[0] or v1span[1] > v2span[1] - 1 >= v1span[0]:
                    #                         print('overlap')
                    #                         print(vertices_by_isent[isent])
                    #                         print(list(enumerate(sent)))
                    continue
                v1type, v2type = v1[2], v2[2]
                guid = (tuple(v1span), tuple(v2span))
                if guid in positives:
                    rel = positives[guid]
                else:
                    rel = 'no_relation'
                all_rows.append(((idoc, isent), tuple(v1span), tuple(v2span), v1type, v2type, rel))
    #         print(sent,positives)
    df = pd.DataFrame(all_rows, columns='idoc_isent span1 span2 type1 type2 rel'.split())
    return df, docs_sents_tokens


def get_traintxts(df, sents, rel, n_postrain, n_negtrain, n_posdev, n_negdev, seed):
    positives = df[df.rel == rel].sample(n_postrain + n_posdev, random_state=seed)
    positives_train, positives_dev = positives[:n_postrain], positives[n_postrain:]

    negatives = df[df.rel != rel].sample(n_negtrain + n_negdev, random_state=seed)
    negatives_train, negatives_dev = negatives[:n_negtrain], negatives[n_negtrain:]

    def formatsent(tokens, span1, span2):
        new_tokens = [x.replace('[', '(').replace(']', ')') for x in tokens]
        new_tokens[span1[0]] = '[o ' + new_tokens[span1[0]]
        new_tokens[span1[1] - 1] = new_tokens[span1[1] - 1] + ']'

        new_tokens[span2[0]] = '[s ' + new_tokens[span2[0]]
        new_tokens[span2[1] - 1] = new_tokens[span2[1] - 1] + ']'

        return ' '.join(new_tokens)

    return ['\n'.join([formatsent(sents[x.idoc_isent], x.span1, x.span2) for _, x in y.iterrows()]) for y in
            [positives_train, negatives_train, positives_dev, negatives_dev]]


def run_exper(rel, numpos, numneg, seed):
    docreddf, docredsents = get_rows(docred)
    docredtestdf, docredtestsents = get_rows(docredtest)

    rel_description = get_relation_desc(rel)
    print(f'Relation: {rel}({rel_description}) Num pos:{numpos} Num neg:{numneg} seed:{seed}')

    assert sum(docredtestdf.rel == rel) > 50, 'not enough positive test examples(50 examples from docred dev)'
    assert sum(docreddf.rel == rel) > numpos + 20, 'not enough positive dev examples (20 examples for dev + num pos)'

    pos_train, neg_train, pos_dev, neg_dev = get_traintxts(docreddf, docredsents, rel, numpos, numneg, 20, 20, seed)

    pos_test, neg_test, _, _ = get_traintxts(docredtestdf, docredtestsents, rel, 50, 500, 0, 0, seed)
    #     return pos_train,neg_train,pos_dev,neg_dev
    # for x in get_exper('P571',10,10,3):
    #     print(x)
    #     print('-'*30)
    exitcode = train_roberta('model1', pos_train, neg_train, pos_dev, neg_dev, gpu=0, verbose=True)
    if exitcode == 0:
        exitcode, results = returnval, results = eval_roberta('model1', pos_test, neg_test, verbose=True)

    df = pd.DataFrame(dict((x, results[x]) for x in results))
    df.to_csv('out.csv')
    return results


if __name__ == '__main__':
    fire.Fire(run_exper)

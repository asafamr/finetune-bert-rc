from pytorch_transformers import RobertaModel, RobertaForMaskedLM, RobertaTokenizer

if __name__ == '__main__':
    _ = RobertaForMaskedLM.from_pretrained('roberta-large', cache_dir='./hugfacecache')
    _ = RobertaModel.from_pretrained('roberta-large', cache_dir='./hugfacecache')
    _ = RobertaTokenizer.from_pretrained('roberta-large', cache_dir='./hugfacecache')

    # _ = BertForMaskedLM.from_pretrained('bert-large-cased', cache_dir='./hugfacecache')
    # _ = BertModel.from_pretrained('bert-large-cased', cache_dir='./hugfacecache')
    # _ = BertTokenizer.from_pretrained('bert-large-cased', cache_dir='./hugfacecache')
    #
    # _ = XLNetLMHeadModel.from_pretrained('xlnet-large-cased', cache_dir='./hugfacecache')
    # _ = XLNetModel.from_pretrained('xlnet-large-cased', cache_dir='./hugfacecache')
    # _ = XLNetTokenizer.from_pretrained('xlnet-large-cased', cache_dir='./hugfacecache')

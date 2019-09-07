from utils_glue import SemEvalQMProcessor

data = SemEvalQMProcessor()
contents = data._read_tsv("/home/logesh/work/garage/playground/ABSA-BERT-pair/data/semeval2014/bert-pair/train_QA_M.csv")
examples = data._create_examples(contents,"utf-8")
print examples
import os
import argparse
import subprocess
import random
from tqdm import tqdm
import torch
import modeling
import data
import sys


SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)


BATCH_SIZE = 16
BATCHES_PER_EPOCH = 32
GRAD_ACC_SIZE = 2

MODEL_MAP = {
    'vanilla_bert': modeling.VanillaBertRanker,
    'cedr_pacrr': modeling.CedrPacrrRanker,
    'cedr_knrm': modeling.CedrKnrmRanker,
    'cedr_drmm': modeling.CedrDrmmRanker
}


def main(model, dataset, train_pairs, qrels, valid_run, qrelf, model_out_dir):
    LR = 0.001
    BERT_LR = 2e-5 # 1e-5
    MAX_EPOCH = 1000
#    warmup_steps = 1000

    params = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    non_bert_params = {'params': [v for k, v in params if not k.startswith('bert.')]}
    bert_params = {'params': [v for k, v in params if k.startswith('bert.')], 'lr': BERT_LR}
    optimizer = torch.optim.Adam([non_bert_params, bert_params], lr=LR)

#    scheduler = get_linear_schedule_with_warmup(
 #       optimizer, num_warmup_steps=warmup_steps, num_training_steps=(BATCH_SIZE * BATCHES_PER_EPOCH / GRAD_ACC_SIZE) * MAX_EPOCH)

    epoch = 0
    patience = 10
    print("*** start training ***")
    top_valid_score = None
    for epoch in range(MAX_EPOCH): 
        if patience < 0:
            break
        loss = train_iteration(model, optimizer, dataset, train_pairs, qrels, None)
        print('train epoch={} loss={}'.format(epoch, loss))
        valid_score = validate(model, dataset, valid_run, qrelf, epoch, model_out_dir)
        print('validation epoch={} score={}'.format(epoch, valid_score))
        patience -= 1
        if top_valid_score is None or valid_score > top_valid_score:
            patience = 10
            top_valid_score = valid_score
            print('new top validation score, saving weights')
            model.save(os.path.join(model_out_dir, 'weights_best.p'))
        model.save(os.path.join(model_out_dir, 'weights_last.p'))


def train_iteration(model, optimizer, dataset, train_pairs, qrels, scheduler):
    total = 0
    max_grad_norm = 1.0
    model.train()
    total_loss = 0.

    with tqdm('training', total=BATCH_SIZE * BATCHES_PER_EPOCH, ncols=80, desc='train', leave=False) as pbar:
        for record in data.iter_train_pairs(model, dataset, train_pairs, qrels, GRAD_ACC_SIZE):
            scores = model(record['query_tok'],
                           record['query_mask'],
                           record['doc_tok'],
                           record['doc_mask'])
            count = len(record['query_id']) // 2
            scores = scores.reshape(count, 2)
            loss = torch.mean(1. - scores.softmax(dim=1)[:, 0]) # pariwse softmax
            loss.backward()
            total_loss += loss.item()
            total += count
            if total % BATCH_SIZE == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
#                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
            pbar.update(count)
            if total >= BATCH_SIZE * BATCHES_PER_EPOCH:
                return total_loss


def validate(model, dataset, run, qrelf, epoch, model_out_dir):
    VALIDATION_METRIC = 'P.20'
    runf = os.path.join(model_out_dir, '{}.run'.format(epoch))
    run_model(model, dataset, run, runf)
    return trec_eval(qrelf, runf, VALIDATION_METRIC)


def run_model(model, dataset, run, runf, desc='valid'):
    BATCH_SIZE = 16
    rerank_run = {}
    with torch.no_grad(), tqdm(total=sum(len(r) for r in run.values()), ncols=80, desc=desc, leave=False) as pbar:
        model.eval()
        for records in data.iter_valid_records(model, dataset, run, BATCH_SIZE):
            scores = model(records['query_tok'],
                           records['query_mask'],
                           records['doc_tok'],
                           records['doc_mask'])
            for qid, did, score in zip(records['query_id'], records['doc_id'], scores):
                rerank_run.setdefault(qid, {})[did] = score.item()
            pbar.update(len(records['query_id']))
    with open(runf, 'wt') as runfile:
        for qid in rerank_run:
            scores = list(sorted(rerank_run[qid].items(), key=lambda x: (x[1], x[0]), reverse=True))
            for i, (did, score) in enumerate(scores):
                runfile.write(f'{qid} 0 {did} {i+1} {score} run\n')


def trec_eval(qrelf, runf, metric):
    trec_eval_f = 'bin/trec_eval'
    output = subprocess.check_output([trec_eval_f, '-m', metric, qrelf, runf]).decode().rstrip()
    output = output.replace('\t', ' ').split('\n')
    assert len(output) == 1
    return float(output[0].split()[2])


def main_cli():
    parser = argparse.ArgumentParser('CEDR model training and validation')
    parser.add_argument('--model', choices=MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--qrels', type=argparse.FileType('rt'))
    parser.add_argument('--train_pairs', type=argparse.FileType('rt'))
    parser.add_argument('--valid_run', type=argparse.FileType('rt'))
    parser.add_argument('--initial_bert_weights', type=argparse.FileType('rb'))
    parser.add_argument('--model_out_dir', default="../models/vbert")
    
    print(sys.argv)
    args = parser.parse_args(args=sys.argv[2:-1])
    dataset = data.read_datafiles(args.datafiles)
    qrels = data.read_qrels_dict(args.qrels)
    train_pairs = data.read_pairs_dict(args.train_pairs)
    valid_run = data.read_run_dict(args.valid_run)

    model = MODEL_MAP[args.model]().cuda()
    if args.initial_bert_weights is not None:
        model.load(args.initial_bert_weights.name)
    os.makedirs(args.model_out_dir, exist_ok=True)
    main(model, dataset, train_pairs, qrels, valid_run, args.qrels, args.model_out_dir)


if __name__ == '__main__':
    
    main_cli()





import argparse
import task
import data

import sys


def main_cli():

    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=task.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))

    print("*** args *** ")
    print(sys.argv)
    args = parser.parse_args(args=sys.argv[2:-1])

    model = task.MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load(args.weights_filename)
    task.run_model(model, dataset, run, args.out_path, desc='rerank')


if __name__ == '__main__':

    main_cli()

#!/opt/conda/bin/python
import main
import os
from datetime import datetime
import argparse

def make_exp_name(name):
    oname = 'EXP_'+datetime.now().strftime("%Y-%m-%d-%X")
    if name and len(name)>0:
        oname += '_{}'.format(name)
    return oname
    
def run_exp(args):

    args.dataset_file = 'LAMP_Covid'
    args.num_workers = 6
    
    # download the pretrained model
    import urllib.request
    from pathlib import Path
    pretrained_folder = Path('./_pretrained_model')
    pretrained_folder.mkdir(exist_ok=True)
    pretrained_path = pretrained_folder / 'detr-r50-e632da11.pth'
    if not pretrained_path.exists():
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth", str(pretrained_path))    
        
    args.resume = str(pretrained_folder / "detr-r50-e632da11.pth") # transfer learning from detr resnet50 model

    args.name = make_exp_name(args.name+'_'+args.custom_data_path.split('/')[-1])
    print('Running experiment {}'.format(args.name))

    args.output_dir = "_outputs_paper/{}".format(args.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main.main(**vars(args))
    return

def run_all_exp(pargs): 
    run_exp(pargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('RT LAMP DETR', parents=[main.get_args_parser()], add_help=False)
    parser.add_argument('--name',default='')
    
    pargs = parser.parse_args()
    run_all_exp(pargs)
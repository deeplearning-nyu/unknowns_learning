import tascrec_resnet
from tascrec_resnet import get_resnet18
from tascrec_resnet import cross_reclassification
from tascrec_resnet import rectify_model
from tascrec_resnet import evaluate_model
from tascrec_resnet import get_augmented_data
import time, datetime
import os
import argparse

script_name = os.path.basename(__file__)
start_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

sample_rate = 0.07 # compute from hyperparameter search
epochs = 100 
run_tag = '03-21-overnight'

config = {
    'balanced' : True, #balance training set
    'opt' : 'SGD', # adam or SGD
    'NUM_WORKERS' : 32,
    'learning_rate' : 5e-4,
    'log_filename' : "runs/logs/log_{}_{}"\
        .format(script_name, start_time),
    'learning_rate' : 5e-4,
    'validation_split' : 0.1
}

id_path = {
  "cifar10": "data/images_classic/cifar10",
  "cifar100": "data/images_classic/cifar100",
  "imagenet200": "data/images_largescale/imagenet200"
}

ood_path = {
  "cifar10": "data/images_classic/cifar10/test",
  "cifar100": "data/images_classic/cifar100/test",
  "mnist": "data/images_classic/mnist/ood",
  "places365": "data/images_classic/places365",
  "svhn": "data/images_classic/svhn",
  "texture": "data/images_classic/texture",
  "tin": "data/images_classic/tin/test",
  "tin597": "data/images_classic/tin597/test",
  "imagenet_1k": "data/images_largescale/imagenet_1k/val",
  "imagenet200": "data/images_largescale/imagenet200/test",
  "imagenet_c": "data/images_largescale/imagenet_c",
  "imagenet_r": "data/images_largescale/imagenet_r",
  "imagenet_v2": "data/images_largescale/imagenet_v2",
  "inaturalist": "data/images_largescale/inaturalist",
  "ninco": "data/images_largescale/ninco",
  "openimage_o": "data/images_largescale/openimage_o",
  "ssb_hard": "data/images_largescale/ssb_hard"
}

norm_dict = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'imagenet': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'imagenet200': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    'covid': [[0.4907, 0.4907, 0.4907], [0.2697, 0.2697, 0.2697]],
    'aircraft': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cub': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
    'cars': [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
}

config['id_path'] = id_path
config['ood_path'] = ood_path
config['norm_dict'] = norm_dict

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='ID argument')
parser.add_argument('--oodn', type=str, help='OOD # argument')
args = parser.parse_args()

ID = args.id
config['ID']= ID 
OOD_set = { 
    'cifar10' : ['mnist', 'cifar100', 'svhn', 'texture', 'tin', 'place365'],
    'cifar100' : ['mnist', 'cifar10', 'svhn', 'texture', 'tin', 'places365'],
    'imagenet200' : ['ninco', 'inaturalist', 'textures', 'openimageo', 'ssb_hard']
}
if ID not in OOD_set:
    raise ValueError(f"ID {ID} not found in OOD_set")

ood_idx = int(args.oodn)
if ood_idx < len(OOD_set[ID]):
    OOD = OOD_set[ID][ood_idx]
else:
    raise ValueError(f"OOD number outside range of OOD_set")
    

print('Processing {}-{}'.format(ID, OOD))
config['OOD'] = OOD 

log_sufix =  ('_{}_bal_{}_uus_reset_epochs{}_{}'
            .format(config['opt'],
                    config['balanced'],
                    epochs, 
                    ID+config['OOD']
                    ))

config['log_filename'] = config['log_filename'] + log_sufix

begin_time = time.time()

model = tascrec_resnet.get_resnet18(**config)

data_dict = tascrec_resnet.get_augmented_data(sample_rate, **config)

uus_mined = tascrec_resnet.cross_reclassification(model, data_dict['aug_train'],
                                        rebuild=1, num_epochs=epochs, 
                                        n_splits=3, **config)

print('CR completed after {}s'.format(int(time.time() - begin_time)))

for i in range(3):

    final_weights = tascrec_resnet.rectify_model(model, 
                                                   data_dict['aug_train'],
                                                   uus_mined, 
                                                   rebuild=1, num_epochs=epochs, **config)
    print('Rectify completed after {}s'.format(int(time.time() - begin_time)))

    accuracy, auroc, aupr, fpr_95 = tascrec_resnet.evaluate_model(model, 
                                                                    final_weights, 
                                                                    data_dict['eval'],
                                                                    **config)

    results_dir = "runs/results/"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, ID + "_" + run_tag + ".txt")

    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            f.write("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n"
                    .format("u.u.s", "FPR@95", "AUROC", "AUPR_OUT", "ACC", 'epochs', 'timestamp'))
            
    with open(results_file, "a") as f:
        f.write("{:<10} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10d} {:<10}\n"
                .format(OOD, fpr_95, auroc, aupr, accuracy, epochs, start_time))
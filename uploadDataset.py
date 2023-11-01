from datasets import load_dataset
import os

dataset_path = 'dataset/bill_dataset/'

dataset = load_dataset('imagefolder', data_dir=dataset_path, split= 'train')
dataset.push_to_hub('Franman/billetes-argentinos', token=os.environ['HF_WRITE_TOKEN'])

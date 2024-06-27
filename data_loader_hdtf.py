import os
import torch
import random
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import librosa    

def split_data(wav_dir, npy_dir, wav_res_dir, npy_res_dir):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    max_split_len = 20  #second
    audio_split_len = max_split_len * 16000
    frame_split_len = max_split_len * 25
    
    
    for r, ds, fs in os.walk(wav_dir):
        for f in tqdm(fs):
            if f.endswith("wav"):
                wav_p = os.path.join(r, f)
                item_name = os.path.splitext(f)[0]

                speech_array, sampling_rate = librosa.load(wav_p, sr=16000)
                input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                
                npy_n = f.replace(".wav", ".npy")
                npy_p = os.path.join(npy_dir, npy_n)
                with open(npy_p, "rb") as f:
                    npy = pickle.load(f)

                split_num = int(len(input_values) / audio_split_len)
                
                print(f"wav={wav_p} npy={npy_p} wav_len={len(input_values)} npy_len={len(npy)} split={split_num}")
                for i in range(0, split_num):
                    if not (i * audio_split_len <= len(input_values) and i * frame_split_len <= len(npy)):
                        continue
                    wav_f = input_values[i* audio_split_len:(i + 1) * audio_split_len]
                    feat_f = npy[i* frame_split_len:(i + 1)* frame_split_len]
                    
                    new_wav_p = os.path.join(wav_res_dir, f"{item_name}_{i}")
                    new_npy_p = os.path.join(npy_res_dir, f"{item_name}_{i}.npy")
                    
                    with open(new_npy_p, "wb") as f:
                        pickle.dump(feat_f, f)
                        
                    np.save(new_wav_p, wav_f)

                
                    print(f"wav={len(wav_f)} vecs={len(feat_f)}")
                os.remove(npy_p)



class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args, data, data_type="train"):
        self.data = data
        self.len = len(self.data)
        
        template_file = os.path.join(args.dataset, args.template_file)
        with open(template_file, 'rb') as fin:
            self.templates = pickle.load(fin)
        
        self.subjects_dict = [k for k in self.templates.keys()]
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(self.subjects_dict) - 4)
        self.args = args

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        wav_name = self.data[index]
        wav_path = os.path.join(self.args.dataset, "wav", self.data[index])
        
        audio = np.load(wav_path)
        file_name = os.path.splitext(wav_name)[0]
        with open(os.path.join(self.args.dataset, "vertices_npy", wav_name), "rb") as f:
            npys = pickle.load(f)

        vertice = [n['v3d'].reshape((-1)) for n in npys]
        
        subject = "_".join(file_name.split("_")[:-2])
        template = self.templates[subject]
        if self.data_type == "train":
            one_hot = self.one_hot_labels[self.subjects_dict.index(subject)]
        else:
            one_hot = self.one_hot_labels
        
        feat_len = int(len(vertice) / 25)
        clip = feat_len
        if feat_len > 10:
            clip = random.randint(0, feat_len - 10)
            audio = audio[clip * 16000:(clip + 10) * 16000]
            vertice = vertice[clip * 25:(clip + 10) * 25]
        print(f"feat_len={feat_len} clip={clip}")
        return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.dataset, args.wav_path)
    sub_projs_k = {}
    for line in open(os.path.join(args.dataset, "wav_list.txt")):
        f = line.strip()
        if f.endswith(".wav"):
            f, _ = os.path.splitext(f)
            
            identity = "_".join(f.split("_")[:-1])
            
            sub_projs_k[identity] = 1
    
    total_projs = len(sub_projs_k)
    val_num, test_num = 2, 2
    train_num = total_projs - val_num - test_num
    sub_projs = [k for k in sub_projs_k.keys()]
    train_case = set(sub_projs[0:train_num])
    val_case = set(sub_projs[train_num:(train_num + val_num)])
    test_case = set(sub_projs[(train_num + val_num):])
    args.train_subjects = " ".join(train_case)
    
    train_data = []
    val_data = []
    test_data = []
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith(".npy"):
                item = os.path.splitext(f)[0]
                sub_id = "_".join(item.split("_")[:-2])
                if sub_id in train_case:
                    train_data.append(f)
                elif sub_id in val_case:
                    val_data.append(f)
                elif sub_id in test_case:
                    test_data.append(f)
                else:
                    print(f"invalid id={f} sub_id={sub_id}")
    
    return train_data, val_data, test_data, train_case, val_case, test_case



    template_file = os.path.join(args.dataset, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin)
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith(".npy"):
                name = os.path.splitext(f)[0]
                sub_id = "_".join(name.split("_")[:-1])
                sub_id = f"{sub_id}.npy"
                
                wav_path = os.path.join(r, f)
                input_values = np.load(wav_path)
                key = f
                data[key]["audio"] = input_values
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]

    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
     'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}
   
    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print(len(train_data), len(valid_data), len(test_data))
    return train_data, valid_data, test_data, subjects_dict

def get_dataloaders(args):
    dataset = {}
    train_data, valid_data, test_data, train_case, val_case, test_case = read_data(args)
    train_data = Dataset(args, train_data, "train")
    dataset["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    valid_data = Dataset(args, valid_data, "val")
    dataset["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=False)
    test_data = Dataset(args, test_data, "test")
    dataset["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    
    print(f"train={train_case}\n val={val_case}\n test={test_case}")
    return dataset

if __name__ == "__main__":
    # get_dataloaders()
    split_data("/root/mlinxiang/vh_exp/FaceFormer/HDTF/wav_o", "/root/mlinxiang/vh_exp/FaceFormer/HDTF/vertices_npy_o", 
               "/root/mlinxiang/vh_exp/FaceFormer/HDTF/wav", "/root/mlinxiang/vh_exp/FaceFormer/HDTF/vertices_npy")
    
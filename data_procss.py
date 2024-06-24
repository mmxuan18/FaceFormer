import os, sys
import pickle
import numpy as np

def avg():
    npys = {}

    root_dir = "/root/mlinxiang/vh_exp/FaceFormer/HDTF/vertices_npy"
    for item in os.listdir(root_dir):
        if item.endswith(".npy"):
            npy_path = os.path.join(root_dir, item)
            
            frames = []
            with open(npy_path, "rb") as f:
                data = pickle.load(f)
                for frame in data:
                    frames.append(frame['v3d'])
                
                mean_frame = np.mean(frames, axis=0)
                npys[item] = np.reshape(mean_frame, (-1))

    with open("/root/mlinxiang/vh_exp/FaceFormer/HDTF/templates.pkl", "wb") as f:
        pickle.dump(npys, f)

def re_avg():
    sub_projs = {}
    with open("/root/mlinxiang/vh_exp/FaceFormer/HDTF/templates.pkl", "rb") as f:
        npys = pickle.load(f)
        for item in npys.keys():
            print(item)
            name = "_".join(item.split("_")[:-1])
            if name in sub_projs:
                sub_projs[name].append(npys[item])
            else:
                sub_projs[name] = [npys[item]]
        
        for item in sub_projs.keys():
            projs = sub_projs[item]
            sub_projs[item] = np.mean(projs, axis=0)
            print(f"projs={len(projs)} shape={sub_projs[item].shape}")
        
    with open("/root/mlinxiang/vh_exp/FaceFormer/HDTF/templates_new.pkl", "wb") as f:
        pickle.dump(sub_projs, f)
        
re_avg()
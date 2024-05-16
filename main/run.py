import os
from argparse import ArgumentParser
import torch
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from tqdm import tqdm

if torch.cuda.is_available() and torch.cuda.device_count()>0:
    device = torch.device('cuda:0')
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    device_name = torch.cuda.get_device_name(0)
    print(f"Device - GPU: {device_name}")
else:
    device = torch.device('cpu')
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    device_name = 'CPU'
    print("Device - CPU")



from size_depth_optimization.sdo_run import run_size_depth_opt

from MultiHMR.app import download_smplx
from MultiHMR.demo import forward_model, get_camera_parameters, load_model as _load_model 
from MultiHMR.utils import normalize_rgb, demo_color as color, create_scene

from GroupRec.modules import init, ModelLoader, DatasetLoader
from GroupRec.process import extract_valid_demo, to_device

def infer_HMR(fn, model, det_thresh, nms_kernel_size, fov):
    global device

    im = Image.open(fn)
    fov, p_x, p_y = fov, None, None # FOV=60 always here!
    img_size = model.img_size

    # Get camera information
    p_x, p_y = None, None
    K = get_camera_parameters(img_size, fov=fov, p_x=p_x, p_y=p_y, device=device)

    # Resise but keep aspect ratio
    img_pil = ImageOps.contain(im, (img_size,img_size)) # keep the same aspect ratio

    # Which side is too small/big
    width, height = img_pil.size
    pad = abs(width - height) // 2

    # Pad
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size, img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size, img_size)) # pad with zero on the smallest side

    # Numpy - normalize - torch.
    resize_img = normalize_rgb(np.asarray(img_pil))
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)

    img_array = np.asarray(img_pil_bis)
    img_pil_visu = Image.fromarray(img_array)

    humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size)

    return humans, img_pil_visu 

def infer_relation(model, loader, device):
    model.model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(loader), total=len(loader)):
            data = to_device(data, device)
            data = extract_valid_demo(data)

            # forward
            pred = model.model(data)
    
    return pred

def generate_3D_scene(humans, model, img_pil_visu):

    l_mesh = [humans[j]['verts_smplx'].detach().cpu().numpy() for j in range(len(humans))]
    l_face = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]
    scene = create_scene(img_pil_visu, l_mesh, l_face, color=color, metallicFactor=0., roughnessFactor=0.5)
    scene.export(glb_fn)
    print(f"Exporting scene in glb format") 


if __name__ == "__main__":
        #parser = ArgumentParser()
        #args = parser.parse_args()

        img_path = 'data/sample.png'
        out_folder = 'output'
        HMR_model_name = 'multiHMR_896_L_synth'
        task = 'relation'
        data_folder= './data'
        HGraph_model_name= 'relation_joint'
        HGraph_model_dir= 'data/relation_joint.pkl'
        dtype = torch.float32


        # Download SMPLX model and mean params
        download_smplx()

        glb_fn = f"{out_folder}.glb"

        # Load Multi-HMR model
        HMR_model = _load_model(HMR_model_name, device=device)
        # Infer sample image using Multi-HMR 
        HMR_out, img_visu = infer_HMR(fn=img_path, model=HMR_model, det_thresh= 0.1, nms_kernel_size= 3, fov=60)

        # Load Hypergraph model 
        _, _, smpl = init(dtype=dtype, note='demo')
        HGraph_model = ModelLoader(device=device, output=out_folder, model=HGraph_model_name, pretrain=True, pretrain_dir=HGraph_model_dir, data_folder=data_folder )
        # Load and preprocess data for Hypergraph Model
        HGraph_dataloader =  DatasetLoader(dtype=dtype, smpl=smpl, data_folder= data_folder, model=HGraph_model_name)
        HGraph_data = HGraph_dataloader.load_demo_data()
        HGraph_data.human_detection()
        #Infer sample image using Hypergraph
        HGraph_out = infer_relation(HGraph_model, HGraph_data, device=device)

        # Run size and depth optimization output params 
        Opt_out = run_size_depth_opt(HMR_out)

        # Pool translation params for vertices and recalculate using scale factor
        for i, human in enumerate(HMR_out):
             human['transl'] = HGraph_out[i]['pred_cam_t']
             human['verts_smplx'] += (human['transl'] - Opt_out[i]['translations'])
             human['verts_smplx'] *= Opt_out[i]['scale']

        # Generate 3D meshes and place into 3D scene
        generate_3D_scene(HMR_out, HMR_model, img_visu)



        
        
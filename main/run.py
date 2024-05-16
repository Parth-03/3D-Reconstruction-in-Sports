import os
from argparse import ArgumentParser
import torch
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path

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

from MultiHMR.demo import forward_model, get_camera_parameters, load_model as _load_model 
from MultiHMR.utils import normalize_rgb, demo_color as color, create_scene

def infer_HMR(fn, model, det_thresh, nms_kernel_size, fov):
    global device
    
    # Is it an image from example_data_dir ?
    #basename = Path(os.path.basename(fn)).stem
    #_basename = f"{basename}_{model_name}_thresh{int(det_thresh*100)}_nms{int(nms_kernel_size)}_fov{int(fov)}"
    #is_known_image = (basename in _list_examples_basename) # only images from example_data
    
    # Filenames
    #if not is_known_image:
    #_basename = 'output' # such that we do not save all the uploaded results - not sure ?
    #glb_fn = f"{_basename}.glb"
    #rend_fn = f"{_basename}.png"
    #glb_fn = os.path.join(tmp_data_dir, _glb_fn)
    #rend_fn = os.path.join(tmp_data_dir, _rend_fn)
    #os.makedirs(tmp_data_dir, exist_ok=True)

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

    #start = time.time()
    humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size)
    #print(f"Forward: {time.time() - start:.2f}sec")   
    #out = [glb_fn]
    #print(out)
    return humans, img_pil_visu 

def generate_3D_scene(humans, model, img_pil_visu):

    l_mesh = [humans[j]['verts_smplx'].detach().cpu().numpy() for j in range(len(humans))]
    l_face = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]
    scene = create_scene(img_pil_visu, l_mesh, l_face, color=color, metallicFactor=0., roughnessFactor=0.5)
    scene.export(glb_fn)
    print(f"Exporting scene in glb format") 


if __name__ == "__main__":
        parser = ArgumentParser()
        parser.add_argument("--HMR_model", type=str, default='multiHMR_896_L_synth')
        parser.add_argument("--img_path", type=str, default='data/sample.png')
        parser.add_argument("--out_folder", type=str, default='output')
        #parser.add_argument("--save_mesh", type=int, default=0, choices=[0,1])
        #parser.add_argument("--extra_views", type=int, default=0, choices=[0,1])
        #parser.add_argument("--det_thresh", type=float, default=0.1)
        #parser.add_argument("--nms_kernel_size", type=float, default=3)
        #parser.add_argument("--fov", type=float, default=60)
        #parser.add_argument("--distance", type=int, default=0, choices=[0,1], help='add distance on the reprojected mesh')
        #parser.add_argument("--unique_color", type=int, default=0, choices=[0,1], help='only one color for all humans')
        
        args = parser.parse_args()

        # Download SMPLX model and mean params
        #download_smplx()

        glb_fn = f"{args.out_folder}.glb"

        # Load HMR model
        HMR_model = _load_model(args.HMR_model, device=device)
        #model_name = args.model_name

        HMR_out, img_visu = infer_HMR(fn=args.img_path, model=HMR_model, det_thresh= 0.1, nms_kernel_size= 3, fov=60)

        Opt = run_size_depth_opt(HMR_out)

        generate_3D_scene(HMR_out, HMR_model, img_visu)



        
        
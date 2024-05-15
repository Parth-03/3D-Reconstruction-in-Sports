import argparse
import torch
import numpy as np
import pickle
import pandas as pd
import os
from PIL import Image, ImageOps
from utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from model import Model
import torch.optim as optim

from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
assert torch.cuda.is_available()

# remove extra humans based on detection score
from copy import copy
def align_humans(predictions, gts):
    predictions = copy(predictions)
    gts = copy(gts)

    aligned_preds = []
    aligned_gts = []
    for pred, gt in zip(predictions, gts):
        while len(pred) > len(gt):
            det_scores = [person['scores'] for person in pred]
            min_value = min(det_scores, key=lambda x: x.item())
            index = det_scores.index(min_value)
            pred = pred[:index] + pred[index+1:]
        while len(pred) < len(gt):
            gt = gt[:len(gt)-1]
            
        assert len(pred) == len(gt)
        aligned_preds.append(pred)
        aligned_gts.append(gt)

    assert len(aligned_preds) == len(aligned_gts)
    return aligned_preds, aligned_gts


def compute_loss(predictions, gts):
    # generate array of only vertex information
    pred_humans_with_only_vertices = []

    count = 0
    for humans in predictions:
        pred_vertices = []
        for human in humans:
            pred_vertices.append(human['verts_smplx'])
            count += 1
        pred_humans_with_only_vertices.append(pred_vertices)
    
    criterion = torch.nn.L1Loss()

    # convert to tensors
    pred_humans_with_only_vertices = torch.stack([tensor for sublist in pred_humans_with_only_vertices for tensor in sublist])
    gts = torch.from_numpy(np.stack([item for sublist in gts for item in sublist]))

    pred_humans_with_only_vertices = pred_humans_with_only_vertices.to(device)
    gts = gts.to(device)

    return criterion(pred_humans_with_only_vertices, gts)
    
def load_data(model):
    img_size = model.img_size

    train_x_path = "AGORA/train_0"
    train_x = [] # images
    train_y = [] # ground truth mesh vertices

    print("Loading Data")
    with open("AGORA/SMPLX/train_0_withjv.pkl", "rb") as file:
        df = pd.read_pickle(file)
        for filename in os.listdir(train_x_path):
            file_path = os.path.join(train_x_path, filename)
            x, img_pil_nopad = open_image(file_path, img_size)
            train_x.append(x)
            y = df[df['imgPath'] == filename.replace("_1280x720", "")]
            train_y.append(np.array(y['gt_verts'][0]))

    assert len(train_x) == len(train_y) == 1453 # Size of AGORA/train0
    print("Data has been loaded")

    return train_x, train_y

def train(model, train_x, train_y):
    import torch
    import torch.optim as optim

    torch.cuda.empty_cache()

    #model = load_model('multiHMR_896_L')

    p_x, p_y = None, None
    K = get_camera_parameters(model.img_size, fov=60, p_x=p_x, p_y=p_y)
    batch_size = 10
    num_epochs = 10  
    num_batches = len(train_x) // batch_size
    remainder = len(train_x) % batch_size

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    # Shuffle Data
    perm = torch.randperm(len(train_x))

    train_x = np.array(train_x)[perm]
    train_y = np.array(train_y)[perm]


    def process_batch(batch):
        batch_output = []
        for image in batch:
            input = torch.from_numpy(image).to(device)
            pred = model(input, 
                            is_training=True, 
                            nms_kernel_size=1,
                            det_thresh=0.4,
                            K=K)
            batch_output.append(pred)
        return batch_output

    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(num_batches):
            batch_x = train_x[i * batch_size: (i + 1) * batch_size]
            batch_y = train_y[i * batch_size: (i + 1) * batch_size]
            
            optimizer.zero_grad()

            batch_output = process_batch(batch_x)
            batch_output, batch_y = align_humans(batch_output, batch_y)
            batch_loss = compute_loss(batch_output, batch_y)

            batch_loss.backward()
            optimizer.step()

            epoch_losses.append(batch_loss.item())
            
            print(f"Epoch {epoch + 1}, Batch {(i + 1)}, Loss: {batch_loss.item()/batch_size}")
            del batch_x, batch_y, batch_output, batch_loss

            torch.cuda.empty_cache()

        if remainder > 0:
            remainder_x = train_x[num_batches * batch_size:]
            remainder_y = train_y[num_batches * batch_size:]

            optimizer.zero_grad()

            remainder_output = process_batch(remainder_x)
            remainder_output, remainder_y = align_humans(remainder_output, remainder_y)
            remainder_loss = compute_loss(remainder_output, remainder_y)

            remainder_loss.backward()
            optimizer.step()

            epoch_losses.append(remainder_loss.item())

            print(f"Epoch {epoch + 1}, Remainder Batch, Loss: {remainder_loss.item()/batch_size}")

            del remainder_x, remainder_y, remainder_output, remainder_loss
            torch.cuda.empty_cache()
        

        total_epoch_loss = sum(epoch_losses)
        average_epoch_loss = total_epoch_loss / len(train_x)
        print(f"Epoch {epoch + 1} Average Loss: {average_epoch_loss}")

## Below are functions from Demo.py

def open_image(img_path, img_size, device=torch.device('cuda')):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) # keep the same aspect ratio

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255))
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch. MODIFIED TO NOT GOT TO TORCH
    resize_img = normalize_rgb(resize_img)
    x = np.expand_dims(resize_img, axis=0)
    return x, img_pil_bis

def load_model(model_name, device=torch.device('cuda')):
    """ Open a checkpoint, build Multi-HMR using saved arguments, load the model weigths. """
    # Model
    ckpt_path = os.path.join(CACHE_DIR_MULTIHMR, model_name+ '.pt')
    if not os.path.isfile(ckpt_path):
        os.makedirs(CACHE_DIR_MULTIHMR, exist_ok=True)
        print(f"{ckpt_path} not found...")
        print("It should be the first time you run the demo code")
        print("Downloading checkpoint from NAVER LABS Europe website...")
        
        try:
            os.system(f"wget -O {ckpt_path} https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt")
            print(f"Ckpt downloaded to {ckpt_path}")
        except:
            assert "Please contact fabien.baradel@naverlabs.com or open an issue on the github repo"

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]['verts_smplx'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
            verts_list,
            faces_list,
            {'focal': focal, 'princpt': princpt},
            alpha=1.0,
            color=_color)

    return pred_rend_array, _color

def main():
    torch.cuda.empty_cache()
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    model = load_model('multiHMR_896_L')
    train_x, train_y = load_data(model)
    
    train(model, train_x, train_y)


if __name__ == "__main__":
    main()

import numpy as np
from easydict import EasyDict
import yaml
import torch
from random import random
import io
import argparse
import os
from models.PACO import PACO
import subprocess

# Load configuration from a YAML file
def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    return config

# Save the processed point cloud data to a .vg file
def save_vg(points, planes, normal, filepath):
    
    # Write point cloud data
    out = ''
    out += 'num_points: {}\n'.format(points.shape[0])
    output = io.StringIO()
    np.savetxt(output, points[:,:3], fmt="%.6f %.6f %.6f")  
    out += output.getvalue()
    output.close()
    
    # Write color data
    out += 'num_colors: {}\n'.format(points.shape[0])
    colors = np.ones((points.shape[0], 3)) * 128
    output = io.StringIO()
    np.savetxt(output, colors, fmt="%d %d %d") 
    out += output.getvalue()
    output.close()
    
    # Write normal vector data
    out += 'num_normals: {}\n'.format(points.shape[0])
    output = io.StringIO()
    np.savetxt(output, normal, fmt="%.6f %.6f %.6f")
    out += output.getvalue()
    output.close()

    # Write group (plane) information
    num_groups = planes.shape[0]
    out += 'num_groups: {}\n'.format(num_groups)

    j_base = 0
    for i in range(planes.shape[0]):
        out += 'group_type: {}\n'.format(0)
        out += 'num_group_parameters: {}\n'.format(4)
        out += 'group_parameters: {} {} {} {}\n'.format(*planes[i])
        out += 'group_label: group_{}\n'.format(i)
        out += 'group_color: {} {} {}\n'.format(random(), random(), random())
        out += 'group_num_point: {}\n'.format(1280)
        # points in group the indice is i
        for j in range(j_base, j_base + 1280):
            out += '{} '.format(j)
        j_base += 1280
        out += '\nnum_children: {}\n'.format(0)

    with open(filepath, 'w') as fout:
        fout.writelines(out)  
     
def inference_single(root, pc_file):  
    # Initialize the PaCo model with the config      
    config = cfg_from_yaml_file(root + "/cfgs/model.yaml")
    checkpoint = torch.load(root + "/ckpt/checkpoint.pth")
    base_model = PACO(config.model)
   
    base_model.load_state_dict(checkpoint)
    base_model.to('cuda')
    base_model.eval()
    
    # Load the point cloud data
    pc =  np.load(pc_file).astype(np.float32)
    pc = torch.from_numpy(pc).float().unsqueeze(0).cuda()

    ret, class_prob = base_model(pc)

    dense_points = ret[-1].squeeze(0).detach().cpu().numpy().reshape(-1, 1280, 3)
    plane_params = ret[-2].squeeze(0).detach().cpu().numpy().reshape(-1, 1, 4)
    class_prob = class_prob.squeeze(0).detach().cpu().numpy()
    mask = class_prob [:, 0] > 0.7 # choose primitives
    
    plane_params = plane_params[mask]
    dense_points = dense_points[mask].reshape(-1, 3)
    normals = np.repeat(plane_params[:, :, :3], 1280, axis=1).reshape(-1, 3)
    plane_params = plane_params.reshape(-1, 4)
    
    tgt_file = pc_file.replace('/pc', '/vg').replace('.npy', '.vg')
    obj_file = pc_file.replace('/pc', '/obj').replace('.npy', '.obj')
    if not os.path.exists(os.path.dirname(tgt_file)):
        os.makedirs(os.path.dirname(tgt_file))
    if not os.path.exists(os.path.dirname(obj_file)):
        os.makedirs(os.path.dirname(obj_file))
        
    # Save the processed point cloud to a .vg file  
    save_vg(dense_points, plane_params, normals, tgt_file)
    
    # Run external polyfit tool to fit the planes and generate the .obj file
    bash_command_template = root + '/PolyFit/polyfit {} {}'
    bash_command = bash_command_template.format(tgt_file, obj_file)
    subprocess.run(bash_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--root', type=str, default=current_directory, help='PaCo directory path')
    parser.add_argument('--pc_file', type=str, required=True, help='incomplete point cloud file')
    args = parser.parse_args()
    inference_single(args.root, args.pc_file)

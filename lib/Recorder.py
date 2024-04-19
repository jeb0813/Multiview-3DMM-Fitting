import torch
import numpy as np
import os
import cv2
import trimesh
import pyrender


class Recorder():
    def __init__(self, save_folder, camera, visualize=False, save_vertices=False):

        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

        self.camera = camera

        self.visualize = visualize
        self.save_vertices = save_vertices

    def log(self, log_data):
        frames = log_data['frames']
        face_model = log_data['face_model']
        intrinsics = log_data['intrinsics']
        extrinsics = log_data['extrinsics']
     
        with torch.no_grad():
            vertices, landmarks = log_data['face_model']()
        
        for n, frame in enumerate(frames):
            face_model.save('{:s}/{:05d}.npz'.format(self.save_folder,n), batch_id=n)
            np.save('{:s}/{:05d}_lmk_3d.npy'.format(self.save_folder,n), landmarks[n].cpu().numpy())
            if self.save_vertices:
                np.save('{:s}/{:05d}_vertices.npy'.format(self.save_folder,n), vertices[n].cpu().numpy())

            if self.visualize:
                for v in range(intrinsics.shape[1]):
                    faces = log_data['face_model'].faces.cpu().numpy()
                    mesh_trimesh = trimesh.Trimesh(vertices=vertices[n].cpu().numpy(), faces=faces)
                    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

                    self.camera.init_renderer(intrinsic=intrinsics[n, v], extrinsic=extrinsics[n, v])
                    render_image = self.camera.render(mesh)

                    cv2.imwrite('{:s}/{:05d}_vis_{:d}.jpg'.format(self.save_folder,n,v), render_image[:,:,::-1])

                
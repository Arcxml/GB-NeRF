from pathlib import Path
import collections
import os
import struct
from typing import Dict, List, Literal, Optional, Tuple, Any
import imageio.v2 as imageio
import cv2

import numpy as np
from enum import Enum
import argparse
#from rich.progress import track




CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])

class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "equirectangular": CameraModel.EQUIRECTANGULAR,
}

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)
class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
            )
        assert len(cameras) == num_cameras
    return cameras
def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images
def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


project2world_cache = {}


# permute from opencv to nerf coordinate
def convert_pose(C2W):  # yz turn to their negative direction
    flip_yz = np.eye(4)
    flip_yz[1, 1] = 1
    flip_yz[2, 2] = 1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def project2world(uv_A, z_A, c2w_A, c2w_A_inv, K, K_inv):
    key = str(uv_A)


    pt_z_A = z_A[uv_A[1],uv_A[0]]  #u->w,v->h
    pt_z_A = np.array([pt_z_A])[None, :, None]

    if key in project2world_cache:
        return project2world_cache[key], pt_z_A.reshape(-1)
    
    n_depths = pt_z_A.shape[1]
    assert n_depths == 1
    #(f"{uv_A= } {K= }{c2w_A= } {pt_z_A= }")
    xyz_A_camera = (np.stack([uv_A[0], uv_A[1], 1])[None, None, :] * pt_z_A) @ np.linalg.inv(K).T #前面那一块也是坐标的转制，camera=k@coo(转制之后就是coo^t @ k^t)
    xyz_A_world = np.concatenate([xyz_A_camera, np.ones([1, n_depths, 1])], axis=2) @ c2w_A.T#加那个Ones是为了跟后面那个3*4匹配
    
    #print(f"{xyz_A_camera= } {xyz_A_world= }")
    project2world_cache[key] = xyz_A_world
    return xyz_A_world, pt_z_A.reshape(-1)


# camera coordinate: x -> right y-> down (no direction inverse between camera coordinate and pixel coordinate)
def reprojection(uv_A, z_A, c2w_A, c2w_A_inv, c2w_B, c2w_B_inv, K_A, K_A_inv, K_B, K_B_inv):
    xyz_A_world, pt_z_A = project2world(uv_A, z_A, c2w_A, c2w_A_inv, K_A, K_A_inv)
    #print(f"{c2w_B= }{K_B= }")
    uvs_B = (xyz_A_world @ c2w_B_inv.T)[:, :, :3] @ K_B.T 
    zs_B = uvs_B[:, :, 2:].reshape(-1)
    uvs_B = (uvs_B[:, :, :2] / uvs_B[:, :, 2:]).astype(np.int32) #映射到图像坐标之后的深度应该是1，这样都除以这个深度，使deep是1
    return uvs_B[0], zs_B, pt_z_A


def unmasked_counterparts(u_A, v_A, src_z, src_c2w, src_c2w_inv, tgt_c2w, tgt_c2w_inv, src_K, src_K_inv, tgt_K, tgt_K_inv):
    uvs_B, zs_B, pt_z_A = reprojection([u_A, v_A], src_z, src_c2w, src_c2w_inv, tgt_c2w, tgt_c2w_inv, src_K, src_K_inv, tgt_K, tgt_K_inv)
    uvs_B = uvs_B[0]
    return uvs_B, zs_B


def trans():
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db

    Returns:
        The number of registered images.
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary( Path("../data/1/sparse/0/cameras.bin")) 
    im_id_to_image = read_images_binary(Path("../data/1/sparse/0/images.bin"))

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        #c2w[0:3, 1:3] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1

        name = im_data.name.split('.')[0]
        name = Path(f"../data/1/images/{name}.png")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])

    out["frames"] = frames
    return out

    ##transformers end
    ##  out["fl_x"]out["fl_y"] out["cx"] out["cy"] out["k1"] out["k2"] out["p1"] out["p2"] out["frames"]

    #  frame = {
    #         "file_path": name.as_posix(),
    #         "transform_matrix": c2w.tolist(),
    #         "colmap_im_id": im_id,
    #     }

    # applied_transform = np.eye(4)[:3, :]
    # applied_transform = applied_transform[np.array([1, 0, 2]), :]
    # applied_transform[2, :] *= -1
    # out["applied_transform"] = applied_transform.tolist()

    # with open( "transforms.json", "w", encoding="utf-8") as f:
    #     json.dump(out, f, indent=4)
def main(args):
    ### start warpping!!!
    out = trans()
    source_view, target_view = args.source_view, args.target_view
   
    #source_depth, target_depth = args.source_depth, args.target_depth
    source_depth = args.source_depth
    source_mask, target_mask = args.source_mask, args.target_mask
    output = args.output
    source_view_bkp = source_view
    target_view_bkp = target_view

    # output directory
    if not os.path.exists(output):
        os.makedirs(output)     

    # read mask
    # source_mask = np.load(source_mask).astype(np.float32)#npy
    # target_mask = np.load(target_mask).astype(np.float32)
    source_mask = imageio.imread(source_mask)
    print(source_mask.shape)
    target_mask = imageio.imread(target_mask)
   # print(source_mask.shape)
    h, w = source_mask.shape[0],source_mask.shape[1]
    #print(f"{h=} {w=}")

    # read depth
    # source_depth = np.fromfile(source_depth, dtype='float32')#必须是float32
    # target_depth = np.fromfile(target_depth, dtype='float32')
    # source_depth = source_depth.reshape(h, w)
    # target_depth = target_depth.reshape(h, w)
    source_depth = imageio.imread(source_depth)
    print(source_depth.shape)
    source_depth = source_depth.reshape(h, w)


    # read image
    source_view = imageio.imread(source_view)
    target_view = imageio.imread(target_view)
    hs, ws = source_view.shape[:2]
    ht, wt = target_view.shape[:2]
    

    if hs != h or ws != w:
        source_view = cv2.resize(source_view, (w, h), cv2.INTER_LINEAR)#为啥是跟mask对齐而不是跟图片对

    if ht != h or wt != w:
        target_view = cv2.resize(target_view, (w, h), cv2.INTER_LINEAR)

    # read camera intrinsics and extrinsics

    # source_idx = int(source_view_bkp.split('/')[-1].split('.')[0])
    # target_idx = int(target_view_bkp.split('/')[-1].split('.')[0])
  
    for i in range(len(out['frames'])):
        x_ext = out['frames'][i]
        if x_ext['file_path'].split('/')[-1].split('.')[0] == source_view_bkp.split('/')[-1].split('.')[0]:
            source_idx = i
            print(f"{source_idx=}")
            break
    

    for i in range(len(out['frames'])):
        x_ext = out['frames'][i]
        if x_ext['file_path'].split('/')[-1].split('.')[0] == target_view_bkp.split('/')[-1].split('.')[0]:
            target_idx = i
            print(f"{target_idx=}")
            break
    # if out['frames']['file_path'] == source_view_bkp:
    #     source_idx = out['frames']['colmap_im_id']-1
    #     source_ext = out['frames'][source_idx]
             
    source_ext = out['frames'][source_idx]
    assert source_ext['file_path'].split('/')[-1] == source_view_bkp.split('/')[-1]
    target_ext = out['frames'][target_idx]
    assert target_ext['file_path'].split('/')[-1] == target_view_bkp.split('/')[-1]
    # the target and the source should have the same w h fl_x fl_y cx cy
    if out['w'] != w:
        out['fl_x'] = w / out['w'] * out['fl_x']
        out['cx'] = w / out['w'] * out['cx']
        out['w'] = w
    if out['h'] != h:
        out['fl_y'] = h / out['h'] * out['fl_y']
        out['cy'] = h / out['h'] * out['cy']
        out['h'] = h

    

    source_c2w = source_ext['transform_matrix']
    target_c2w = target_ext['transform_matrix']

    source_K = np.array([[out['fl_x'], 0, out['cx']], [0, out['fl_y'], out['cy']], [0, 0, 1]])
    source_K_inv = np.linalg.inv(source_K)
    target_K = np.array([[out['fl_x'], 0, out['cx']], [0, out['fl_y'], out['cy']], [0, 0, 1]])
    target_K_inv = np.linalg.inv(target_K)
   
    masked_coords = np.where(np.array(source_mask == 1))
    #print(f"{masked_coords=}")
    masked_coords_test = np.where(np.array(source_mask == 1))
    # print(f"{max(masked_coords_test[0])=}")
    # print(f"{max(masked_coords_test[1])=}")

    # convert poses from opencv to opengl############本来不就是opengl
    src_c2w = convert_pose(source_c2w)
    src_c2w_inv = np.linalg.inv(src_c2w)

    tgt_c2w = convert_pose(target_c2w)
    tgt_c2w_inv = np.linalg.inv(tgt_c2w)
    # src_c2w = source_c2w
    # src_c2w_inv = np.linalg.inv(src_c2w)

    # tgt_c2w = target_c2w
    # tgt_c2w_inv = np.linalg.inv(tgt_c2w)


    # visualization
    recon_tgt_mask = np.zeros_like(target_mask)
    target_mask_bkp = target_mask[:, :, None]
    recon_tgt = target_view * (1 - target_mask_bkp)
    combined_tgt = target_view * (1 - target_mask_bkp)
    unfilled_regions = target_mask.copy()
    combined_unfilled_regions = target_mask.copy()

    for v_A, u_A in zip(list(masked_coords[0]), list(masked_coords[1])):
        #print(f"{source_view[v_A, u_A, :] =}")
        res, z_val = unmasked_counterparts(u_A, v_A, source_depth, src_c2w, src_c2w_inv, tgt_c2w, tgt_c2w_inv, source_K, source_K_inv, target_K, target_K_inv)
        proj_u, proj_v = res[0], res[1]  # W and H dimension #project point
        if proj_v < h and proj_u < w:
            #print(f"{proj_v=} {proj_u=}")
            recon_tgt_mask[proj_v, proj_u] = 1 #project过去之后变成白的（这个初始值是全黑的，表示的是warp过去了哪些
            unfilled_regions[proj_v, proj_u] = 0 #把能project变成黑的，剩下的就是白的（就是warp了之后还没有覆盖到的点
            
            recon_tgt[proj_v, proj_u, :] = source_view[v_A, u_A, :] #原来点上的，投影到目标图片（投影过去了之后的
            combined_tgt[v_A, u_A, :] = source_view[v_A, u_A, :] #原来点上的，粘贴过来就还是这个点（哪些点呗投影过去了
            combined_unfilled_regions[v_A, u_A] = 0 #把能原来的点变成黑的，剩下的就是白的（这个的初始是Mask
      
    
    # save results
    proj_mask = os.path.join(output, 'proj_mask.png')
    proj_result = os.path.join(output, 'proj_result.png')
    combined_result = os.path.join(output, 'combined_result.png')
    unfilled_mask = os.path.join(output, 'unfilled_regions.png')
    combined_unfilled_mask = os.path.join(output, 'combined_unfilled_regions.png')
    imageio.imwrite(proj_mask, recon_tgt_mask)
    imageio.imwrite(proj_result, recon_tgt)
    imageio.imwrite(combined_result, combined_tgt)
    imageio.imwrite(unfilled_mask, unfilled_regions)
    imageio.imwrite(combined_unfilled_mask, combined_unfilled_regions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_view', type=str, default='../data/1/images_4/20220819_104506.png')
    parser.add_argument('--target_view', type=str, default='../data/1/images_4/20220819_104507.png')
    parser.add_argument('--source_depth', type=str, default='../data/1/images_4/depth/image.png')
    # parser.add_argument('--target_depth', type=str, default='../depths/depth_50.bin')
    parser.add_argument('--source_mask', type=str, default='../data/1/images_4/label/20220819_104506.png')
    parser.add_argument('--target_mask', type=str, default='../data/1/images_4/label/20220819_104507.png')#data/1/images_4/20220819_104221.png
    #parser.add_argument('--ext', type=str, default='transforms.json', help='external camera parameters')
    parser.add_argument('--output', type=str, default='.')
    args = parser.parse_args()
    main(args)

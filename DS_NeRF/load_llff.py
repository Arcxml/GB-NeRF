import cv2
import numpy as np
import os
import imageio
from pathlib import Path
from .colmapUtils.read_write_model import *
from .colmapUtils.read_write_dense import *
import json


# Slightly modified version of LLFF data loading code
# see https://github.com/Fyusion/LLFF for original

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any(
        [f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg,
                        '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, prepare=False, refined=False,
               use_MVSeg=False, origin=False,cream_loss=False,use_ref=False,args=None):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))

    # 3 x 5 x N
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images')))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape
    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    if not origin: 
        imgdir = os.path.join(basedir, 'images' + sfx)
    else:
        imgdir = os.path.join(basedir, 'images' + sfx + '/RGB_inpainted')
    
    # if use_clipaway_dataset:
    #     imgdir = os.path.join(basedir, 'images' + sfx + '/RGB_inpainting_clipaway')
   
    print(f"{imgdir=}")
    mskdir = os.path.join(basedir, 'images' + sfx + '/label')
    
    depthdir = os.path.join(basedir, 'images' + sfx + '/Depth_inpainted')


    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')]
    mskfiles = [os.path.join(mskdir, f.split('.')[0] + '.png') for f in sorted(os.listdir(mskdir)) if
                'cutout' not in f and 'pseudo' not in f and
                (f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png'))]

    try:
        depthfiles = [os.path.join(depthdir, f.split('.')[0] + '.png') for f in sorted(os.listdir(depthdir)) if
                      f.endswith('JPG') or f.endswith('jpg') or f.endswith('jpeg') or f.endswith('png')]
    except:
        depthfiles = mskfiles

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            # return imageio.imread(f, ignoregamma=True)
            return imageio.imread(f)
        else:
            return imageio.imread(f)

    imgs = []
    for i, f in enumerate(imgfiles):
        if use_ref:
            if i == 68-40:  # 第69张图片，索引从0开始           
                f = os.path.join(basedir, '1_out.png')
                print("have changed 1_out.png ")
        img = imread(f)[..., :3] / 255.
        imgs.append(img)
    imgs = np.stack(imgs, -1)

    masks = []
    mask_indices = []
    for i, f in enumerate(mskfiles):
        try:
            msk = imread(f)
            msk = msk / msk.max()
            if len(msk.shape) > 2:
                msk = msk[:, :, 0]
            if msk.shape != (imgs.shape[0], imgs.shape[1]):
                print(f"{msk.shape=}")
                msk = cv2.resize(
                    msk, (imgs.shape[1], imgs.shape[0]), interpolation=cv2.INTER_NEAREST)
                print(msk.shape)
            # todo comment this or change the dilation iterations
            # msk = cv2.dilate(msk, np.ones((5, 5), np.uint8), iterations=5)
            masks.append(msk)
            mask_indices.append(i)
        except:
            masks.append(-np.ones((imgs.shape[0], imgs.shape[1])))

    inpainted_depths = []
    for i, f in enumerate(depthfiles):
        
        try:
            if use_ref:
                if i == 68-40:  # 第69张图片的深度信息
                    f = os.path.join(basedir, '1_out_pred.png')  # 假设特殊深度图的文件名
            
            msk = imread(f)

            msk = msk / 255.
            if len(msk.shape) > 2:
                msk = msk[:, :, 0]
            if msk.shape != (imgs.shape[0], imgs.shape[1]):
                print(msk.shape)
                msk = cv2.resize(
                    msk, (imgs.shape[1], imgs.shape[0]), interpolation=cv2.INTER_NEAREST)
                print(msk.shape)
            inpainted_depths.append(msk)
        except:
            inpainted_depths.append(-np.ones((imgs.shape[0], imgs.shape[1])))

    masks = np.stack(masks, -1)
    masks = masks / np.max(masks)

    inpainted_depths = np.stack(inpainted_depths, -1)
    # inpainted_depths = inpainted_depths / np.max(inpainted_depths)

    print('Loaded image data', bds.shape, imgs.shape, poses.shape, inpainted_depths.shape, masks.shape)
    return poses, bds, imgs, masks, inpainted_depths, mask_indices


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array(
            [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    def p34_to_44(p): return np.concatenate(
        [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i,
                                [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(
        p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array(
            [radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(
        poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds, sc, np.linalg.inv(p34_to_44(c2w[None]))


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False,
                   spherify_hack=True, prepare=False, refined=False, use_MVSeg=False, origin=False,cream_loss=False,use_ref=False,args=None):
    poses, bds, imgs, masks, inpainted_depths, mask_indices = _load_data(basedir,
                                                                         factor=factor,
                                                                         prepare=prepare,
                                                                         refined=refined,
                                                                         use_MVSeg=use_MVSeg,
                                                                         origin=origin,
                                                                         cream_loss=cream_loss, 
                                                                         use_ref=use_ref,
                                                                         args=args)  # factor=8 downsamples original imgs by 8x
    
    print('Loaded', basedir, bds.min(), bds.max())

    # print('poses_bound.npy:\n', poses[:,:,0])

    # Correct rotation matrix ordering and move variable dim to axis 0
    # [-u, r, -t] -> [r, u, -t]
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    masks = np.moveaxis(masks, -1, 0).squeeze().astype(np.float32)
    inpainted_depths = np.moveaxis(
        inpainted_depths, -1, 0).squeeze().astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    print("bds:", bds[0])

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc


    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds, _, _ = spherify_poses(poses, bds)

    elif spherify_hack:
        poses_spherify, render_poses_spherify, bds_spherify, sc, T = spherify_poses(
            poses, bds)
        # from IPython import embed; embed()
        bds = bds_spherify / sc
        def p34_to_44(p): return np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])],
                                                1)
        poses44 = p34_to_44(poses[:, :, :-1])
        poses44_spherify = p34_to_44(poses_spherify[:, :, :-1])
        render_poses44_spherify = p34_to_44(render_poses_spherify[:, :, :-1])

        render_poses = []
        for i in range(render_poses44_spherify.shape[0]):
            # from IPython import embed; embed()
            tmp = render_poses44_spherify[i].copy()
            tmp[:3, 3] /= sc
            tmp = np.linalg.inv(T[0]) @ tmp
            tmp = np.hstack([tmp[:3, :], poses[0, :3, -1:]])
            render_poses.append(
                tmp
            )
    else:
        pass
    c2w = poses_avg(poses)
    print('recentered', c2w.shape)
    print(c2w[:3, :4])

    # Get spiral
    # Get average pose
    up = normalize(poses[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
    dt = .75
    mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
    focal = mean_dz

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
    rads = np.percentile(np.abs(tt), 90, 0)
    c2w_path = c2w
    N_views = 120
    N_rots = 2
    if path_zflat:
        zloc = -close_depth * .1
        c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
        rads[2] = 0.
        N_rots = 1
        N_views /= 2

    # Generate poses for spiral path
    render_poses = render_path_spiral(
        c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
    # end of else

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)
    # masks = (masks > 0.5).astype(np.float32)
    masks = (masks).astype(np.float32)
    inpainted_depths = (inpainted_depths).astype(np.float32)

    if masks.shape[-1] == 3:
        masks = masks[:, :, :, 0].squeeze()

    if inpainted_depths.shape[-1] == 3:
        inpainted_depths = inpainted_depths[:, :, :, 0].squeeze()

    
    print('Loaded image data2:', images.shape, poses.shape, render_poses.shape, i_test.shape, inpainted_depths.shape, masks.shape)
    poses_test = poses[0:40, :,:]
    poses = poses[40:, :,:]  # note that we manually select the training poses, which is slightly different SPIn-NeRF
    return images, poses, bds, render_poses, i_test, masks, inpainted_depths, mask_indices, poses_test
    # return images[:40, ...], poses[:40, ...], bds, render_poses[:40, ...], i_test, masks[:40, ...], inpainted_depths[:40, ...], mask_indices
    
    
def get_poses(images):
    poses = []
    for i in images:
        R = images[i].qvec2rotmat()
        t = images[i].tvec.reshape([3, 1])
        bottom = np.array([0, 0, 0, 1.]).reshape([1, 4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        poses.append(c2w)
    return np.array(poses)


def load_colmap_depth(basedir, factor=8, bd_factor=.75, prepare=False):
    data_file = Path(basedir) / 'colmap_depth.npy'

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(
        Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)
    # factor=8 downsamples original imgs by 8x
    _, bds_raw, _, _, _, _ = _load_data(
        basedir, factor=factor, prepare=prepare)
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    data_list = []
    for id_im in range(1, len(images) + 1-40): # -40
        depth_list = []
        coord_list = []
        weight_list = []
        id_im = id_im
        for i in range(len(images[id_im+40].xys)): # +40
            point2D = images[id_im+40].xys[i]  #+40
            id_3D = images[id_im+40].point3D_ids[i] #+40
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @
                     (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(point2D / factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "weight": np.array(weight_list)})
        else:
            pass
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list


def load_sensor_depth(basedir, factor=8, bd_factor=.75):
    data_file = Path(basedir) / 'colmap_depth.npy'

    images = read_images_binary(Path(basedir) / 'sparse' / '0' / 'images.bin')
    points = read_points3d_binary(
        Path(basedir) / 'sparse' / '0' / 'points3D.bin')

    Errs = np.array([point3D.error for point3D in points.values()])
    Err_mean = np.mean(Errs)
    print("Mean Projection Error:", Err_mean)

    poses = get_poses(images)
    # factor=8 downsamples original imgs by 8x
    _, bds_raw, _, _, _, _ = _load_data(basedir, factor=factor)
    bds_raw = np.moveaxis(bds_raw, -1, 0).astype(np.float32)
    # print(bds_raw.shape)
    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds_raw.min() * bd_factor)

    near = np.ndarray.min(bds_raw) * .9 * sc
    far = np.ndarray.max(bds_raw) * 1. * sc
    print('near/far:', near, far)

    depthfiles = [Path(basedir) / 'depth' / f for f in sorted(os.listdir(Path(basedir) / 'depth')) if
                  f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    depths = [imageio.imread(f) for f in depthfiles]
    depths = np.stack(depths, 0)

    data_list = []
    for id_im in range(1, len(images) + 1):
        depth_list = []
        coord_list = []
        weight_list = []
        for i in range(len(images[id_im].xys)):
            point2D = images[id_im].xys[i]
            id_3D = images[id_im].point3D_ids[i]
            if id_3D == -1:
                continue
            point3D = points[id_3D].xyz
            depth = (poses[id_im - 1, :3, 2].T @
                     (point3D - poses[id_im - 1, :3, 3])) * sc
            if depth < bds_raw[id_im - 1, 0] * sc or depth > bds_raw[id_im - 1, 1] * sc:
                continue
            err = points[id_3D].error
            weight = 2 * np.exp(-(err / Err_mean) ** 2)
            depth_list.append(depth)
            coord_list.append(point2D / factor)
            weight_list.append(weight)
        if len(depth_list) > 0:
            print(id_im, len(depth_list), np.min(depth_list),
                  np.max(depth_list), np.mean(depth_list))
            data_list.append(
                {"depth": np.array(depth_list), "coord": np.array(coord_list), "weight": np.array(weight_list)})
        else:
            print(id_im, len(depth_list))
    # json.dump(data_list, open(data_file, "w"))
    np.save(data_file, data_list)
    return data_list

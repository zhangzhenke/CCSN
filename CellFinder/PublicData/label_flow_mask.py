import torch
import numpy as np
import fastremap


from torch.nn.functional import grid_sample
from skimage import morphology
from scipy.ndimage import mean, find_objects
from scipy.ndimage.filters import maximum_filter1d


torch_GPU = torch.device("cuda")
torch_CPU = torch.device("cpu")


# 将标签（细胞核掩码）转换为梯度流，用于训练模型。
def labels_to_flows(labels, use_gpu=False, device=None, redo_flows=False):

    # Labels b x 1 x h x w
    labels = labels.cpu().numpy().astype(np.int16)
    nimg = len(labels)


    # 如果 labels 列表中的标签图像原本是二维的 (height, width)，经过这段代码处理后，它们将变为三维的 (1, height, width)。
    if labels[0].ndim < 3:
        labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]


    # 如果标签的第一个维度为 1 或标签的维度小于 3，或者需要重新计算梯度流（redo_flows=True），则重新计算梯度流。
    if labels[0].shape[0] == 1 or labels[0].ndim < 3 or redo_flows:

        # 使用 fastremap.renumber 确保标签图像中的每个连通区域（即每个细胞核）都有一个唯一的整数标签。
        # 使用 masks_to_flows 函数计算每个标签图像的梯度流。
        labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
        veci = [
            # 对于每个标签图像 labels[n]，获取其第一个通道（h,w）（假设标签图像是三维的，形状为 (1, height, width)）
            masks_to_flows(labels[n][0], use_gpu=use_gpu, device=device)
            for n in range(nimg)
        ]

        # 将标签、距离变换、向量流和热图（边界和掩码）组合在一起，形成最终的梯度流。
        flows = [
            # labels[n]：第 n 个标签图像，形状为 [1, h, w]。
            # labels[n] > 0.5：对第 n 个标签图像进行二值化，形状为 [1, h, w]。二值化结果将是一个布尔数组，表示每个像素是否为前景。
            # veci[n]：第 n 个标签图像的梯度流，形状为 [2, h, w]。梯度流包含两个通道，分别表示每个像素的水平和垂直流动方向。
            # 将上述三个数组在第一个维度上进行拼接，形成一个多通道数组。拼接后的数组形状为 [4, h, w]
            np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(np.float32)
            for n in range(nimg)
        ]

    return np.array(flows)


# 模型输出转换为掩码
def compute_masks(
    dP,
    cellprob,
    p=None,
    niter=200,
    cellprob_threshold=0.4,
    flow_threshold=0.4,
    interp=True,
    resize=None,
    use_gpu=False,
    device=None,
):
    """compute masks using dynamics from dP, cellprob, and boundary"""
    # cp_mask是0,1矩阵
    cp_mask = cellprob > cellprob_threshold
    # 小孔是指被前景（通常是白色或 True）包围的背景（通常是黑色或 False）区域。这个函数会填充这些小孔，使得图像更加连贯。
    cp_mask = morphology.remove_small_holes(cp_mask, area_threshold=16)
    # 移除二值图像中面积小于 min_size 的连通区域（即小物体）
    cp_mask = morphology.remove_small_objects(cp_mask, min_size=16)


    # 检查 cp_mask 是否包含任何细胞：
    if np.any(cp_mask):  
        
        # 调用 follow_flows 函数计算流动方向 p 和索引 inds
        if p is None:
            p, inds = follow_flows(
                dP * cp_mask / 5.0,
                niter=niter,
                interp=interp,
                use_gpu=use_gpu,
                device=device,
            )

            # 返回一个全零的掩码和流动方向矩阵。
            if inds is None:
                shape = resize if resize is not None else cellprob.shape
                mask = np.zeros(shape, np.uint16)
                p = np.zeros((len(shape), *shape), np.uint16)
                return mask, p


        # 根据流动方向 p: C H W 和细胞掩码 cp_mask 计算细胞掩码 mask: H W
        mask = get_masks(p, iscell=cp_mask)


        # 如果 mask 中的最大值大于 0 且 flow_threshold 不为 None 且大于 0，
        # 则调用 remove_bad_flow_masks 函数过滤不良的流动掩码。
        if mask.max() > 0 and flow_threshold is not None and flow_threshold > 0:
            # make sure labels are unique at output of get_masks
            mask = remove_bad_flow_masks(
                mask, dP, threshold=flow_threshold, use_gpu=use_gpu, device=device
            )

        # 返回一个全零的掩码和流动方向矩阵。
        else:  # nothing to compute, just make it compatible
            print('---0---')
            shape = resize if resize is not None else cellprob.shape
            mask = np.zeros(shape, np.uint16)
            p = np.zeros((len(shape), *shape), np.uint16)
    
    # 返回一个全零的掩码和流动方向矩阵。
    else:
        print('---zero---')
        shape = resize if resize is not None else cellprob.shape
        mask = np.zeros(shape, np.uint16)
        p = np.zeros((len(shape), *shape), np.uint16)


    return mask, p



# 扩散计算，生成梯度流 mu。
def _extend_centers_gpu(
    neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device("cuda")
):
    if device is not None:
        device = device
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)

    T = torch.zeros((nimg, Ly, Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    for i in range(n_iter):
        T[:, meds[:, 0], meds[:, 1]] += 1
        Tneigh = T[:, pt[:, :, 0], pt[:, :, 1]]
        Tneigh *= isneigh
        T[:, pt[0, :, 0], pt[0, :, 1]] = Tneigh.mean(axis=1)
    del meds, isneigh, Tneigh
    T = torch.log(1.0 + T)
    # gradient positions
    grads = T[:, pt[[2, 1, 4, 3], :, 0], pt[[2, 1, 4, 3], :, 1]]
    del pt
    dy = grads[:, 0] - grads[:, 1]
    dx = grads[:, 2] - grads[:, 3]
    del grads
    mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)
    return mu_torch



def diameters(masks):
    _, counts = np.unique(np.int32(masks), return_counts=True)
    counts = counts[1:]
    md = np.median(counts ** 0.5)
    if np.isnan(md):
        md = 0
    md /= (np.pi ** 0.5) / 2
    return md, counts ** 0.5


# 通过扩展标签图像的中心点并进行扩散计算，生成梯度流。
def masks_to_flows_gpu(masks, device=None):

    if device is None:
        device = torch.device("cuda")

    Ly0, Lx0 = masks.shape
    # 创建一个比原始图像大2个像素的填充图像 masks_padded，并将原始标签图像放置在填充图像的中心。
    Ly, Lx = Ly0 + 2, Lx0 + 2
    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks


    # 使用 np.nonzero 获取标签图像中非零像素的坐标 y 和 x。
    y, x = np.nonzero(masks_padded)
    # 生成这些像素的邻居坐标 neighborsY 和 neighborsX，并将它们堆叠成 neighbors。
    neighborsY = np.stack((y, y - 1, y + 1, y, y, y - 1, y - 1, y + 1, y + 1), axis=0)
    neighborsX = np.stack((x, x, x, x - 1, x + 1, x - 1, x + 1, x - 1, x + 1), axis=0)
    neighbors = np.stack((neighborsY, neighborsX), axis=-1)


    # 使用 find_objects 函数找到每个标签的切片 slices。
    slices = find_objects(masks)

    # 计算每个标签的中心点 centers，中心点是标签中距离中位数最近的像素。
    centers = np.zeros((masks.max(), 2), "int")
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si

            yi, xi = np.nonzero(masks[sr, sc] == (i + 1))
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi - xmed) ** 2 + (yi - ymed) ** 2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i, 0] = ymed + sr.start
            centers[i, 1] = xmed + sc.start


    # 获取邻居像素的标签 neighbor_masks，并检查这些邻居是否属于同一个标签 isneighbor。
    neighbor_masks = masks_padded[neighbors[:, :, 0], neighbors[:, :, 1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = np.array(
        [[sr.stop - sr.start + 1, sc.stop - sc.start + 1] for sr, sc in slices]
    )
    # 计算扩展的迭代次数 n_iter。
    n_iter = 2 * (ext.sum(axis=1)).max()

    # 调用 _extend_centers_gpu 函数进行扩散计算，生成梯度流 mu。
    mu = _extend_centers_gpu(
        neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter, device=device
    )

    # 对梯度流进行归一化处理。
    mu /= 1e-20 + (mu ** 2).sum(axis=0) ** 0.5

    # 创建一个与原始图像大小相同的零数组 mu0，并将计算得到的梯度流放回原始图像的位置。
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y - 1, x - 1] = mu
    # 创建一个与 mu0 大小相同的零数组 mu_c。
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c


# 将标签图像转换为梯度流。梯度流是一个向量场，表示每个像素的流动方向和大小。
def masks_to_flows(masks, use_gpu=False, device=None):

    # 如果 masks 中的最大值为0，或者非零元素的数量为1，则认为掩码是空的，返回一个全零的梯度流。
    if masks.max() == 0 or (masks != 0).sum() == 1:
        # dynamics_logger.warning('empty masks!')
        return np.zeros((2, *masks.shape), "float32")


    # 设置设备
    if use_gpu:
        if use_gpu and device is None:
            device = torch_GPU
        elif device is None:
            device = torch_CPU
        masks_to_flows_device = masks_to_flows_gpu


    # 处理三维掩码：
    if masks.ndim == 3:
        Lz, Ly, Lx = masks.shape
        mu = np.zeros((3, Lz, Ly, Lx), np.float32)
        for z in range(Lz):
            mu0 = masks_to_flows_device(masks[z], device=device)[0]
            mu[[1, 2], z] += mu0
        for y in range(Ly):
            mu0 = masks_to_flows_device(masks[:, y], device=device)[0]
            mu[[0, 2], :, y] += mu0
        for x in range(Lx):
            mu0 = masks_to_flows_device(masks[:, :, x], device=device)[0]
            mu[[0, 1], :, :, x] += mu0
        return mu
    
    # 处理二维掩码：形状为 (Ly, Lx)
    elif masks.ndim == 2:
        mu, mu_c = masks_to_flows_device(masks, device=device)
        return mu

    else:
        raise ValueError("masks_to_flows only takes 2D or 3D arrays")


def steps2D_interp(p, dP, niter, use_gpu=False, device=None):
    shape = dP.shape[1:]
    if use_gpu:
        if device is None:
            device = torch_GPU
        shape = (
            np.array(shape)[[1, 0]].astype("float") - 1
        )  # Y and X dimensions (dP is 2.Ly.Lx), flipped X-1, Y-1
        pt = (
            torch.from_numpy(p[[1, 0]].T).float().to(device).unsqueeze(0).unsqueeze(0)
        )  # p is n_points by 2, so pt is [1 1 2 n_points]
        im = (
            torch.from_numpy(dP[[1, 0]]).float().to(device).unsqueeze(0)
        )  # covert flow numpy array to tensor on GPU, add dimension
        # normalize pt between  0 and  1, normalize the flow
        for k in range(2):
            im[:, k, :, :] *= 2.0 / shape[k]
            pt[:, :, :, k] /= shape[k]

        # normalize to between -1 and 1
        pt = pt * 2 - 1

        # here is where the stepping happens
        for t in range(niter):
            # align_corners default is False, just added to suppress warning
            dPt = grid_sample(im, pt, align_corners=False)

            for k in range(2):  # clamp the final pixel locations
                pt[:, :, :, k] = torch.clamp(
                    pt[:, :, :, k] + dPt[:, k, :, :], -1.0, 1.0
                )

        # undo the normalization from before, reverse order of operations
        pt = (pt + 1) * 0.5
        for k in range(2):
            pt[:, :, :, k] *= shape[k]

        p = pt[:, :, :, [1, 0]].cpu().numpy().squeeze().T
        return p

    else:
        assert print("ho")


# 计算流动方向 p 和索引 inds
def follow_flows(dP, mask=None, niter=200, interp=True, use_gpu=True, device=None):
    shape = np.array(dP.shape[1:]).astype(np.int32)
    niter = np.uint32(niter)

    p = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    p = np.array(p).astype(np.float32)

    inds = np.array(np.nonzero(np.abs(dP[0]) > 1e-3)).astype(np.int32).T

    if inds.ndim < 2 or inds.shape[0] < 5:
        return p, None

    if not interp:
        assert print("woo")

    else:
        p_interp = steps2D_interp(
            p[:, inds[:, 0], inds[:, 1]], dP, niter, use_gpu=use_gpu, device=device
        )
        p[:, inds[:, 0], inds[:, 1]] = p_interp

    return p, inds


# 计算每个掩码的流场误差
def flow_error(maski, dP_net, use_gpu=False, device=None):

    if dP_net.shape[1:] != maski.shape:
        print("ERROR: net flow is not same size as predicted masks")
        return

    # 从掩码生成的流场
    dP_masks = masks_to_flows(maski, use_gpu=use_gpu, device=device)
    # 初始化流场误差数组
    flow_errors = np.zeros(maski.max())
    # 计算每个掩码的流场误差
    for i in range(dP_masks.shape[0]):
        # 误差计算使用均方误差（MSE）
        flow_errors += mean(
            (dP_masks[i] - dP_net[i] / 5.0) ** 2,
            maski,
            index=np.arange(1, maski.max() + 1),
        )

    # 返回流场误差（flow_errors）和从掩码生成的流场（dP_masks）
    return flow_errors, dP_masks


# 过滤不良的流动掩码
def remove_bad_flow_masks(masks, flows, threshold=0.4, use_gpu=False, device=None):

    # 计算流场误差：
    merrors, _ = flow_error(masks, flows, use_gpu, device)
    # 找到误差超过阈值的掩码：
    badi = 1 + (merrors > threshold).nonzero()[0]
    # 移除误差超过阈值的掩码：
    masks[np.isin(masks, badi)] = 0
    return masks


# 根据流动方向 p 和细胞掩码 cp_mask 计算细胞掩码 mask
def get_masks(p, iscell=None, rpad=20):
    pflows = []
    edges = []
    shape0 = p.shape[1:]
    dims = len(p)

    for i in range(dims):
        pflows.append(p[i].flatten().astype("int32"))
        edges.append(np.arange(-0.5 - rpad, shape0[i] + 0.5 + rpad, 1))

    h, _ = np.histogramdd(tuple(pflows), bins=edges)
    hmax = h.copy()
    for i in range(dims):
        hmax = maximum_filter1d(hmax, 5, axis=i)

    seeds = np.nonzero(np.logical_and(h - hmax > -1e-6, h > 10))
    Nmax = h[seeds]
    isort = np.argsort(Nmax)[::-1]
    for s in seeds:
        s = s[isort]

    pix = list(np.array(seeds).T)

    shape = h.shape
    if dims == 3:
        expand = np.nonzero(np.ones((3, 3, 3)))
    else:
        expand = np.nonzero(np.ones((3, 3)))
    for e in expand:
        e = np.expand_dims(e, 1)

    for iter in range(5):
        for k in range(len(pix)):
            if iter == 0:
                pix[k] = list(pix[k])
            newpix = []
            iin = []
            for i, e in enumerate(expand):
                epix = e[:, np.newaxis] + np.expand_dims(pix[k][i], 0) - 1
                epix = epix.flatten()
                iin.append(np.logical_and(epix >= 0, epix < shape[i]))
                newpix.append(epix)
            iin = np.all(tuple(iin), axis=0)
            for p in newpix:
                p = p[iin]
            newpix = tuple(newpix)
            igood = h[newpix] > 2
            for i in range(dims):
                pix[k][i] = newpix[i][igood]
            if iter == 4:
                pix[k] = tuple(pix[k])

    M = np.zeros(h.shape, np.uint32)
    for k in range(len(pix)):
        M[pix[k]] = 1 + k

    for i in range(dims):
        pflows[i] = pflows[i] + rpad
    M0 = M[tuple(pflows)]

    # remove big masks
    uniq, counts = fastremap.unique(M0, return_counts=True)
    big = np.prod(shape0) * 0.9
    bigc = uniq[counts > big]
    if len(bigc) > 0 and (len(bigc) > 1 or bigc[0] != 0):
        M0 = fastremap.mask(M0, bigc)
    fastremap.renumber(M0, in_place=True)  # convenient to guarantee non-skipped labels
    M0 = np.reshape(M0, shape0)



    # mask（即 M0）的形状与输入的流动方向数组 p 的形状相同，去掉第一个维度后的形状。
    # 具体来说，如果 p 的形状是 (dims, height, width) 或 (dims, depth, height, width)，
    # 那么 mask 的形状将是 (height, width) 或 (depth, height, width)。
    return M0

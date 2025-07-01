import numpy as np
import cv2
import torch
import random

from statistics import mean
from segment_anything import SamPredictor, sam_model_registry
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from pathlib import Path

model_type = 'vit_b'
checkpoint = '/hsfscqjf1/ST_CQ/P24Z28400N0269/zhangzhenke/ssam/300_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
sam_model.to(device)
sam_model.train()
#print(sam_model)

del checkpoint
torch.cuda.empty_cache()
import gc
gc.collect()


checkpoint_path = "/hsfscqjf1/ST_CQ/P24Z28400N0269/zhangzhenke/ssam/best_model.pth"


bbox_coords = {}
ground_truth_masks = {}
# 遍历文件夹中的文件
for f in sorted(Path('/hsfscqjf1/ST_CQ/P24Z28400N0269/zhangzhenke/ssam/data/cellbindb_he/annotations/train/').iterdir())[300:400]:
    # 提取文件名的前缀
    k = f.stem[:]
    #print(k)

    # 读取图像并转换为灰度图
    im = cv2.imread(f.as_posix(), -1).astype(np.uint16)

    classes = np.unique(im)
    bbox_coords[k] = []
    ground_truth_masks[k] = []  

    for class_id in classes:
        if class_id == 0:  # 跳过背景类别
            continue
        binary_mask = (im == class_id).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(binary_mask)
        for label in range(1, num_labels):
            ground_truth_masks[k].append(labels == label)            
            component_pixels = np.argwhere(labels == label)
            x_min, y_min = component_pixels.min(axis=0)
            x_max, y_max = component_pixels.max(axis=0)
            bbox_coords[k].append(np.array([y_min, x_min, y_max, x_max]))
    #print(bbox_coords[k])
    #print(ground_truth_masks[k])    




transformed_data = defaultdict(dict)
for k in bbox_coords.keys():
    image = cv2.imread(f'/hsfscqjf1/ST_CQ/P24Z28400N0269/zhangzhenke/ssam/data/cellbindb_he/images/train/{k}.png')
    #print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
    
    input_image = sam_model.preprocess(transformed_image)
    original_image_size = image.shape[:2]
    input_size = tuple(transformed_image.shape[-2:])

    transformed_data[k]['image'] = input_image
    transformed_data[k]['input_size'] = input_size
    transformed_data[k]['original_image_size'] = original_image_size


    del input_image_torch, transformed_image, input_image
    torch.cuda.empty_cache()
    import gc
    gc.collect()



# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 1e-4
wd = 0
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = torch.nn.BCEWithLogitsLoss()
keys = list(bbox_coords.keys())


num_epochs = 25
losses = []
for epoch in range(num_epochs):
    epoch_losses = []
    # Just train on the first 20 examples
    for k in keys[:]:
        input_image = transformed_data[k]['image'].to(device)
        input_size = transformed_data[k]['input_size']
        original_image_size = transformed_data[k]['original_image_size']
        
        # No grad here as we don't want to optimise the encoders
        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            
            prompt_box = bbox_coords[k]
            #print('prompt_box', prompt_box)

            processed_boxes = []
            for prompt_box_ in prompt_box:
                processed = transform.apply_boxes(prompt_box_, original_image_size)
                processed_boxes.append(processed)
            box = np.concatenate(processed_boxes, axis=0)
            #print('box', box)

            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            #print('box_torch', box_torch)
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
            #print('sparse_embeddings, dense_embeddings', sparse_embeddings.shape, dense_embeddings.shape)

        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        #binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        #binary_mask = torch.max(binary_mask.squeeze(1), dim=0)[0]  # 沿着第0维取最大值
        #print('upscaled_masks', upscaled_masks.squeeze(1)[0])

        #gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
        #gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
        gt_mask = torch.from_numpy(np.array(ground_truth_masks[k])).to(device)
        #print('gt_mask ', gt_mask[0])
        

        loss = loss_fn(upscaled_masks.squeeze(1), gt_mask.to(torch.float32))
        #print('loss', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())


        del input_image, image_embedding, box_torch, box, sparse_embeddings, dense_embeddings, low_res_masks, iou_predictions, upscaled_masks, gt_mask, loss,prompt_box, processed_boxes, processed
        torch.cuda.empty_cache()
        import gc
        gc.collect()


    losses.append(epoch_losses)
    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    if (epoch % 5) == 0:
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': sam_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, checkpoint_path)
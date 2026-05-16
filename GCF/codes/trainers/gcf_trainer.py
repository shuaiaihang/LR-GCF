import os
import random
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prettytable import PrettyTable
from colorama import Fore
from ._base import BaseTrainer


class GCFTrainer(BaseTrainer):

    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 criterions=None,
                 metrics=None,
                 logger=None,
                 device='cuda',
                 resume_from=None,
                 labeled_bs=8,
                 data_parallel=False,
                 ckpt_save_path=None,
                 max_iter=6000,
                 eval_interval=1000,
                 save_ckpt_interval=2000,
                 consistency=0.1,
                 consistency_rampup=40.0,
                 attn_consistency_weight=1.5) -> None:

        super(GCFTrainer, self).__init__(model, optimizer, scheduler, criterions, metrics, logger, device,
                                           resume_from, labeled_bs, consistency, consistency_rampup, data_parallel,
                                           ckpt_save_path, max_iter, eval_interval, 0,
                                           save_ckpt_interval)

        self.attn_consistency_weight = attn_consistency_weight

    @staticmethod
    def _dice_loss(pred, target, smooth=1e-5):
        assert pred.shape[0] == target.shape[0]
        intersect = torch.sum(torch.mul(pred, target))
        return 1 - (2 * intersect + smooth) / (torch.sum(pred.pow(2)) + torch.sum(target.pow(2)) + smooth)
    

    def _compute_ce_dice_loss_with_confidence(self, predictions, pseudo_logits, ce_weight=0.5, dice_weight=0.5):
        _, C, _, _ = predictions.shape
        pseudo_probs = F.softmax(pseudo_logits, dim=1)  # [B, C, H, W]
        _, pseudo_labels = torch.max(pseudo_probs, dim=1)  # [B, H, W]
        ce_loss = F.cross_entropy(predictions, pseudo_labels)
        predictions_soft = F.softmax(predictions, dim=1)  # [B, C, H, W]
        pseudo_labels_one_hot = F.one_hot(pseudo_labels, num_classes=C).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        dice_loss = 0.0
        for c in range(1, C):  # 跳过背景类
            pred_c = predictions_soft[:, c].flatten()  # [B*H*W]
            target_c = pseudo_labels_one_hot[:, c].flatten()  # [B*H*W]
            dice_loss += self._dice_loss(pred_c, target_c)
        dice_loss = dice_loss / (C - 1)
        total_loss = ce_weight * ce_loss + dice_weight * dice_loss
        return total_loss
    

    def inverse_rotate_using_labels(self, predictions, rot_labels, n=7):
        """
        根据真实旋转标签将分割预测反转回原始方向
        
        参数:
            predictions: 分割预测 [B, C, H, W]
            rot_labels: 真实旋转标签 [B, n, n]，每个值表示该patch的旋转次数(0,1,2,3)
            n: 网格大小，默认7
            
        返回:
            inverse_predictions: 反转后的分割预测 [B, C, H, W]
        """
        B, C, H, W = predictions.shape
        ph, pw = H // n, W // n
        
        inverse_predictions = predictions.clone()
        
        for b in range(B):
            for i in range(n):
                for j in range(n):
                    k = rot_labels[b, i, j].item()
                    
                    # 如果原来没有旋转(0度)，不需要反转
                    if k == 0:
                        continue
                    
                    # 计算patch坐标
                    h0, h1 = i * ph, (i + 1) * ph if i < n-1 else H
                    w0, w1 = j * pw, (j + 1) * pw if j < n-1 else W
                    
                    # 提取patch
                    patch = predictions[b:b+1, :, h0:h1, w0:w1]  # [1, C, ph, pw]
                    
                    # 反向旋转：如果原来旋转了k次，现在要旋转(4-k)次来恢复
                    inverse_k = (4 - k) % 4
                    if inverse_k > 0:
                        inverse_patch = torch.rot90(patch, inverse_k, dims=(2, 3))
                        inverse_predictions[b:b+1, :, h0:h1, w0:w1] = inverse_patch
        
        return inverse_predictions

    def local_random_rot_select_n(self,
                                  image: torch.Tensor,
                                  label: torch.Tensor,
                                  n: int = 7,
                                  m_min: int = 4,
                                  m_max: int = 5):
    
        # 保证 label 带通道
        if label.ndim == 2:
            label = label.unsqueeze(0)  # → [1,H,W]

        _, H, W = image.shape
        ph = H // n
        pw = W // n

        new_image = image.clone()
        new_label = label.clone()
        ks = torch.zeros((n, n), dtype=torch.long, device=image.device)

        # 随机决定本次要旋转多少块
        m = random.randint(m_min, m_max)
        # 从 n*n 个块中随机选 m 个不重复索引
        all_indices = [(i, j) for i in range(n) for j in range(n)]
        chosen = random.sample(all_indices, m)

        for (i, j) in chosen:
            k = random.randint(1, 3)
            ks[i, j] = k

            # 块坐标
            h0 = i * ph
            h1 = H if i == n-1 else (i+1) * ph
            w0 = j * pw
            w1 = W if j == n-1 else (j+1) * pw

            # 切片
            patch_img = image[:, h0:h1, w0:w1]    # [C, ph', pw']
            patch_lbl = label[:, h0:h1, w0:w1]    # [L, ph', pw']

            # 旋转
            rot_img = torch.rot90(patch_img, k, dims=(1, 2))
            rot_lbl = torch.rot90(patch_lbl, k, dims=(1, 2))

            # 写回
            new_image[:, h0:h1, w0:w1] = rot_img
            new_label[:, h0:h1, w0:w1] = rot_lbl

        return new_image, new_label, ks
    


    def attn_consistency_loss(self, attn_list1, attn_list2, batch_size, temperature=0.01):
        """
        计算注意力一致性损失
        只对top_k差异最大的token对计算MSE
        """
        # 检查是否有注意力数据
        if not attn_list1 or not attn_list2:
            return torch.tensor(0.0, device=self.device)
        
        epsilon = 1e-8
        top_k_ratio = 0.1
        losses = []
        
        def compute_layer_loss(attn1, attn2, num_blocks=1):
            """
            计算单层的注意力一致性损失
            attn1, attn2: [B*num_blocks, num_tokens, num_tokens]
            """
            attn1_scaled = attn1 / temperature
            attn2_scaled = attn2 / temperature
            p_all = F.softmax(attn1_scaled, dim=-1) + epsilon
            q_all = F.softmax(attn2_scaled, dim=-1) + epsilon
            
            total_loss = 0.0
            valid_count = 0
            
            for b in range(batch_size):
                for block_idx in range(num_blocks):
                    idx = b * num_blocks + block_idx
                    if idx >= p_all.shape[0]:
                        continue
                        
                    p_block = p_all[idx]
                    q_block = q_all[idx]
                    
                    # 计算差异并选择top_k
                    diff = p_block - q_block
                    diff_norms = torch.norm(diff, dim=1, p=2)
                    num_tokens = diff_norms.shape[0]
                    top_k = max(1, int(num_tokens * top_k_ratio))
                    
                    _, top_indices = torch.topk(diff_norms, k=top_k, largest=True)
                    p_selected = p_block[top_indices]
                    q_selected = q_block[top_indices]
                    
                    loss_mse = F.mse_loss(p_selected, q_selected, reduction='mean')
                    total_loss += loss_mse
                    valid_count += 1
            
            return total_loss / valid_count if valid_count > 0 else 0.0
        
        # 处理各层的W-MSA和SW-MSA
        for layer_idx in range(3):  # 3层
            # W-MSA
            if 'w_msa' in attn_list1 and 'w_msa' in attn_list2:
                w_msa_1 = attn_list1['w_msa']
                w_msa_2 = attn_list2['w_msa']
                if len(w_msa_1) > layer_idx and len(w_msa_2) > layer_idx:
                    w_attn1 = w_msa_1[layer_idx]
                    w_attn2 = w_msa_2[layer_idx]
                    # 第一层只有1个block,其他层有4个block
                    num_blocks = 1 if layer_idx == 0 else 4
                    losses.append(compute_layer_loss(w_attn1, w_attn2, num_blocks))
            
            # SW-MSA (第一层没有SW-MSA)
            if layer_idx > 0 and 'sw_msa' in attn_list1 and 'sw_msa' in attn_list2:
                sw_msa_1 = attn_list1['sw_msa']
                sw_msa_2 = attn_list2['sw_msa']
                if len(sw_msa_1) > layer_idx and len(sw_msa_2) > layer_idx:
                    sw_attn1 = sw_msa_1[layer_idx]
                    sw_attn2 = sw_msa_2[layer_idx]
                    losses.append(compute_layer_loss(sw_attn1, sw_attn2, num_blocks=4))
        
        if len(losses) > 0:
            return sum(losses) / len(losses)
        else:
            return torch.tensor(0.0, device=self.device)


    def train_step(self, batch_data, step):

        log_infos, scalars = {}, {}
        data_1, label_1 = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        labeled_bs = self.labeled_bs
        
        # 处理完整batch数据（包括有标签和无标签）
        batch_size = data_1.shape[0]

        # 为所有样本生成两次不同的增强
        images_aug1, labels_aug1, ks_list1 = [], [], []  # 第一次增强
        images_aug2, labels_aug2, ks_list2 = [], [], []  # 第二次增强
        
        for i in range(batch_size):
            image = data_1[i]
            label = label_1[i] if i < labeled_bs else torch.zeros_like(label_1[0])  # 无标签数据用零填充
            

            # 第一次分块旋转
            img_rot1, lbl_rot1, ks1 = self.local_random_rot_select_n(image, label, n=8, m_min=4, m_max=4)
            images_aug1.append(img_rot1)
            labels_aug1.append(lbl_rot1)
            ks_list1.append(ks1)
            
            # 第二次分块旋转 - 不同的随机种子
            img_rot2, lbl_rot2, ks2 = self.local_random_rot_select_n(image, label, n=8, m_min=4, m_max=4)
            images_aug2.append(img_rot2)
            labels_aug2.append(lbl_rot2)
            ks_list2.append(ks2)

        # 堆叠数据
        data_aug1 = torch.stack(images_aug1, dim=0).to(self.device)  
        label_aug1 = torch.stack(labels_aug1, dim=0).to(self.device)
        rot_labels1 = torch.stack(ks_list1, dim=0).to(self.device)
        
        data_aug2 = torch.stack(images_aug2, dim=0).to(self.device)  
        label_aug2 = torch.stack(labels_aug2, dim=0).to(self.device)
        rot_labels2 = torch.stack(ks_list2, dim=0).to(self.device)
        
        # 前向传播 - 两个不同增强都经过两个decoder
        # 第一个增强数据的前向传播
        outputs1 = self.model(data_aug1, self.device)
        seg1_1 = outputs1['seg_tf1']  # 增强1 → decoder1 (student)
        seg1_2 = outputs1['seg_tf2']  # 增强1 → decoder2 (teacher)
        rot_logits1 = outputs1['rotation_logits']
        attn_list1_1 = outputs1['attn_list1']  # decoder1的注意力
        attn_list1_2 = outputs1['attn_list2']  # decoder2的注意力

        # 第二个增强数据的前向传播
        outputs2 = self.model(data_aug2, self.device)
        seg2_1 = outputs2['seg_tf1']  # 增强2 → decoder1 (student)
        seg2_2 = outputs2['seg_tf2']  # 增强2 → decoder2 (teacher)
        rot_logits2 = outputs2['rotation_logits']
        attn_list2_1 = outputs2['attn_list1']  # decoder1的注意力
        attn_list2_2 = outputs2['attn_list2']  # decoder2的注意力

        # === 损失计算 ===
        
        # 1. 监督损失 - 对有标签数据的两个decoder都计算
        # decoder1 (student)的监督损失
        loss_sup1_dec1 = sum(criterion(seg1_1[:labeled_bs], label_aug1[:labeled_bs]) for criterion in self.criterions)
        loss_sup2_dec1 = sum(criterion(seg2_1[:labeled_bs], label_aug2[:labeled_bs]) for criterion in self.criterions)
        loss_sup_dec1 = (loss_sup1_dec1 + loss_sup2_dec1)
        
        # decoder2 (teacher)的监督损失
        loss_sup1_dec2 = sum(criterion(seg1_2[:labeled_bs], label_aug1[:labeled_bs]) for criterion in self.criterions)
        loss_sup2_dec2 = sum(criterion(seg2_2[:labeled_bs], label_aug2[:labeled_bs]) for criterion in self.criterions)
        loss_sup_dec2 = (loss_sup1_dec2 + loss_sup2_dec2)
        
        # 总监督损失
        loss_sup = (loss_sup_dec1 + loss_sup_dec2) 
        
        # 2. 旋转损失 - 计算全部数据（有标签+无标签）的student decoder
        loss_rot1 = F.cross_entropy(rot_logits1.permute(0,2,3,1).reshape(-1,4), rot_labels1.reshape(-1))
        loss_rot2 = F.cross_entropy(rot_logits2.permute(0,2,3,1).reshape(-1,4), rot_labels2.reshape(-1))
        loss_rot = (loss_rot1 + loss_rot2) 
        
        # 3. 注意力一致性损失
        loss_attn_cons = torch.tensor(0.0, device=self.device)
        if self.attn_consistency_weight > 0:
            # 第一个解码器的注意力一致性：A1和B1
            loss_attn_1 = self.attn_consistency_loss(attn_list1_1, attn_list2_1, batch_size)
            # 第二个解码器的注意力一致性：A2和B2
            loss_attn_2 = self.attn_consistency_loss(attn_list1_2, attn_list2_2, batch_size)
            loss_attn_cons = (loss_attn_1 + loss_attn_2) / 2

        # 4. 无标签数据的损失 - CPS交叉监督（带置信度过滤）
        loss_unsup = torch.tensor(0.0, device=self.device)

        
        if labeled_bs < batch_size :  # 确保有无标签数据
            # 获取无标签部分的输出
            seg1_1_u = seg1_1[labeled_bs:]  # 增强1 → decoder1 (无标签)
            seg1_2_u = seg1_2[labeled_bs:]  # 增强1 → decoder2 (无标签)
            seg2_1_u = seg2_1[labeled_bs:]  # 增强2 → decoder1 (无标签)
            seg2_2_u = seg2_2[labeled_bs:]  # 增强2 → decoder2 (无标签)
            
            # 获取真实旋转标签（无标签部分）
            rot_labels1_u = rot_labels1[labeled_bs:]
            rot_labels2_u = rot_labels2[labeled_bs:]
            
            # 将两个decoder的分割预测都反转回原始方向（用于生成伪标签logits）
            seg1_1_u_inverse = self.inverse_rotate_using_labels(seg1_1_u, rot_labels1_u, n=8)
            seg1_2_u_inverse = self.inverse_rotate_using_labels(seg1_2_u, rot_labels1_u, n=8)
            seg2_1_u_inverse = self.inverse_rotate_using_labels(seg2_1_u, rot_labels2_u, n=8)
            seg2_2_u_inverse = self.inverse_rotate_using_labels(seg2_2_u, rot_labels2_u, n=8)
            

            # CPS交叉监督损失（无置信度过滤）：
            # decoder2增强1的伪标签 -> 监督decoder1增强2 (交叉增强)
            loss_2to1_cross = self._compute_ce_dice_loss_with_confidence(
                seg2_1_u_inverse, seg1_2_u_inverse.detach())
            # decoder1增强1的伪标签 -> 监督decoder2增强2 (交叉增强)
            loss_1to2_cross = self._compute_ce_dice_loss_with_confidence(
                seg2_2_u_inverse, seg1_1_u_inverse.detach())
            # decoder2增强2的伪标签 -> 监督decoder1增强1 (交叉增强)
            loss_2to1_cross2 = self._compute_ce_dice_loss_with_confidence(
                seg1_1_u_inverse, seg2_2_u_inverse.detach())
            # decoder1增强2的伪标签 -> 监督decoder2增强1 (交叉增强)
            loss_1to2_cross2 = self._compute_ce_dice_loss_with_confidence(
                seg1_2_u_inverse, seg2_1_u_inverse.detach())


            # 总的CPS损失 (权重从2提升到2.5)
            loss_unsup = (loss_2to1_cross + loss_1to2_cross + loss_2to1_cross2 + loss_1to2_cross2) / 2  
        # cps_weight只在step>1000时计算，否则为0


        cps_weight = self.get_current_consistency_weight(step // 100) 

        
        # 总损失（提升各损失权重以增强学习）
        total_loss = loss_sup + 0.5*loss_rot + cps_weight * loss_unsup + 3*loss_attn_cons
        # 日志
        log_infos['loss/sup'] = round(loss_sup.item(), 4)
        log_infos['loss/rot'] = round(loss_rot.item(), 4)
        log_infos['loss/unsup'] = round(loss_unsup.item(), 4)
        log_infos['loss/attn_cons'] = f"{loss_attn_cons.item():.2e}"  # 科学计数法


        log_infos['loss/total'] = round(total_loss.item(), 4)
        log_infos['cps_weight'] = round(cps_weight, 4)

        scalars['loss/sup'] = loss_sup.item()
        scalars['loss/rot'] = loss_rot.item()
        scalars['loss/unsup'] = loss_unsup.item()
        scalars['loss/attn_cons'] = loss_attn_cons.item()
        scalars['loss/total'] = total_loss.item()
        scalars['cps_weight'] = cps_weight

        # 旋转预测准确率
        with torch.no_grad():
            rot_pred1 = rot_logits1.argmax(dim=1)
            rot_pred2 = rot_logits2.argmax(dim=1)
            acc_rot1 = (rot_pred1 == rot_labels1).float().mean()
            acc_rot2 = (rot_pred2 == rot_labels2).float().mean()
            acc_rot = (acc_rot1 + acc_rot2) / 2
            log_infos["acc/rot"] = round(acc_rot.item(), 4)
            scalars["acc/rot"] = round(acc_rot.item(), 4)

        return total_loss, log_infos, scalars


    def _val_step(self, batch_data, inference_fn):
        data, labels = batch_data['image'].to(self.device), batch_data['label'].to(self.device)
        preds = inference_fn(data, self.device)
        return {metric.name: metric(preds, labels) for metric in self.metrics}

    @torch.no_grad()
    def _val_decoder(self, val_loader, inference_fn, test=False):
        self.model.eval()
        val_res = None
        val_scalars = {}
        if self.logger is not None:
            self.logger.info('Evaluating...')
        if test:
            val_loader = tqdm(val_loader, desc='Testing', unit='batch',
                              bar_format='%s{l_bar}{bar}{r_bar}%s' % (Fore.LIGHTCYAN_EX, Fore.RESET))

        for batch_data in val_loader:
            batch_res = self._val_step(batch_data, inference_fn)
            if val_res is None:
                val_res = batch_res
            else:
                for metric_name in val_res.keys():
                    for key in val_res[metric_name].keys():
                        val_res[metric_name][key] += batch_res[metric_name][key]

        for metric_name in val_res.keys():
            for key in val_res[metric_name].keys():
                val_res[metric_name][key] = val_res[metric_name][key] / len(val_loader)
                val_scalars[f'val_tf/{metric_name}.{key}'] = val_res[metric_name][key]

            val_res_list = list(val_res[metric_name].values())
            val_res[metric_name]['Mean'] = np.mean(val_res_list[1:])
            val_scalars[f'val_tf/{metric_name}.Mean'] = val_res[metric_name]['Mean']

        val_table = PrettyTable()
        val_table.field_names = ['Metirc'] + list(list(val_res.values())[0].keys())
        for metric_name in val_res.keys():
            if metric_name in ['Dice', 'Jaccard', 'Acc', 'IoU', 'Recall', 'Precision']:
                temp = [float(format(_ * 100, '.2f')) for _ in val_res[metric_name].values()]
            else:
                temp = [float(format(_, '.2f')) for _ in val_res[metric_name].values()]
            val_table.add_row([metric_name] + temp)
        return val_res, val_scalars, val_table

    def val_tf(self, val_loader, test=False):
        return self._val_decoder(val_loader, self.model.inference_tf_1, test)

    def val_tf_2(self, val_loader, test=False):
        return self._val_decoder(val_loader, self.model.inference_tf_2, test)

    def train(self, train_loader, val_loader):
        max_epoch = self.max_iter // len(train_loader) + 1
        step = self.start_step
        best_performance = 0.0  # 初始化 best performance
        self.model.train()
        with tqdm(total=self.max_iter - self.start_step, bar_format='[{elapsed}<{remaining}] ') as pbar:
            for _ in range(max_epoch):
                for batch_data in train_loader:
                    loss, log_infos, scalars = self.train_step(batch_data, step)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    if (step + 1) % 10 == 0:
                        scalars.update({'lr': self.scheduler.get_lr()[0]})
                        log_infos.update({'lr': self.scheduler.get_lr()[0]})
                        self.logger.update_scalars(scalars, step + 1)
                        self.logger.info(f'[{step + 1}/{self.max_iter}] {log_infos}')

                    if (step + 1) % self.eval_interval == 0:
                        if val_loader is not None:
                            val_res_1, val_scalars_1, val_table_1 = self.val_tf(val_loader)
                            self.logger.info(f'val_tf result:\n{val_table_1.get_string()}')
                            self.logger.update_scalars(val_scalars_1, step + 1)
                            self.model.train()
                            
                            val_res_2, val_scalars_2, val_table_2 = self.val_tf_2(val_loader)
                            self.logger.info(f'val_tf_2 result:\n{val_table_2.get_string()}')
                            self.logger.update_scalars(val_scalars_2, step + 1)
                            self.model.train()
                            
                            # 保存 best 模型
                            # 找到两个验证结果中的最佳 Dice
                            performance_1 = val_res_1['Dice']['Mean']
                            performance_2 = val_res_2['Dice']['Mean']
                            performance = max(performance_1, performance_2)
                            
                            if performance > best_performance:
                                best_performance = performance
                                if not os.path.exists(self.ckpt_save_path):
                                    os.makedirs(self.ckpt_save_path)
                                save_mode_path = os.path.join(self.ckpt_save_path, 
                                                             'iter_{}_dice_{}.pth'.format(step + 1, round(best_performance, 4)))
                                save_best_path = os.path.join(self.ckpt_save_path, 'best_model.pth')
                                torch.save(self.model.state_dict(), save_mode_path)
                                torch.save(self.model.state_dict(), save_best_path)
                                self.logger.info(f'Save best model at iter {step + 1}, dice: {best_performance:.4f} (val_tf: {performance_1:.4f}, val_tf_2: {performance_2:.4f})')

                    if (step + 1) % self.save_ckpt_interval == 0:
                        if not os.path.exists(self.ckpt_save_path):
                            os.makedirs(self.ckpt_save_path)
                        self.save_ckpt(step + 1, f'{self.ckpt_save_path}/iter_{step + 1}.pth')
                    step += 1
                    pbar.update(1)
                    if step >= self.max_iter:
                        break
                if step >= self.max_iter:
                    break

        if not os.path.exists(self.ckpt_save_path):
            os.makedirs(self.ckpt_save_path)
            torch.save(self.model.state_dict(), f'{self.ckpt_save_path}/ckpt_final.pth')

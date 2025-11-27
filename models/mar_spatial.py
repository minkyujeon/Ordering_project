from functools import partial
import math
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    device = order.device

    if isinstance(mask_len, torch.Tensor):
        mask_len = int(mask_len.item())
    else:
        mask_len = int(mask_len)

    masking = torch.zeros(bsz, seq_len, device=device)

    index = order[:, :mask_len]  # [B, K]
    src   = torch.ones_like(index, dtype=masking.dtype, device=device)  # [B, K]

    masking = masking.scatter(dim=-1, index=index, src=src)
    return masking.bool()

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 mask_ratio_min=0.7,
                 label_drop_prob=0.1,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 num_sampling_steps='100',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))

        # --------------------------------------------------------------------------
        # MAR variant masking ratio
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # --------------------------------------------------------------------------
        # MAR encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # MAR decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            grad_checkpointing=grad_checkpointing
        )
        self.diffusion_batch_mul = diffusion_batch_mul
        self._register_static_grids()

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def _register_static_grids(self):
        """Pre-calculate coordinate grids for spatial ordering"""
        h, w = self.seq_h, self.seq_w
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.flatten()
        x = x.flatten()
        
        r2 = (y - cy)**2 + (x - cx)**2
        theta = torch.atan2(y - cy, x - cx)
        
        self.register_buffer('grid_r2', r2)
        self.register_buffer('grid_theta', theta)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    # ==========================================================================
    # ORDERING LOGIC
    # ==========================================================================

    def sample_orders_new(self, bsz, mode="random", random_block_order=False, random_within_block=False):
        device = self.encoder_pos_embed_learned.device
        
        if mode == "center_spiral":
            # Generates Center -> Out
            # We want Center at END of order array to be unmasked first.
            # r2 sort Descending: Edge (Big r2) ... Center (Small r2)
            return self.sample_orders_spiral(bsz, descending=True)
            
        elif mode == "edge_spiral":
            # Generates Edge -> In
            # We want Edge at END of order array.
            # r2 sort Ascending: Center (Small r2) ... Edge (Big r2)
            return self.sample_orders_spiral(bsz, descending=False)

        elif mode == "blockwise":
            return self.sample_orders_blockwise(bsz, random_block_order=random_block_order, 
                                                random_within_block=random_within_block)
        
        elif mode == "multiscale":
            return self.sample_orders_multiscale(bsz)
        
        elif mode == "center_perturb":
            return self.sample_orders_center_out_with_perturb(bsz)
            
        else:
            # Original fully-random order
            orders = []
            for _ in range(bsz):
                # Fast GPU permutation
                orders.append(torch.randperm(self.seq_len, device=device))
            return torch.stack(orders)

    def sample_orders_spiral(self, bsz, descending=True):
        """
        Vectorized Spiral Ordering.
        descending=True  => [Edge ... Center] => Reveals Center First (Center-Out)
        descending=False => [Center ... Edge] => Reveals Edge First (Edge-In)
        """
        # Use pre-calculated buffers
        r2 = self.grid_r2
        theta = self.grid_theta
        
        # Combine radius and angle for sorting key
        # Add small noise to angle to break perfect symmetries
        score = r2 + (theta + math.pi) / (2 * math.pi + 1e-6) * 0.1
        
        # Sort
        _, indices = torch.sort(score, descending=descending)
        
        return indices.unsqueeze(0).repeat(bsz, 1)

    def sample_orders_blockwise(self, bsz, block_h=4, block_w=4,
                            random_block_order=False,
                            random_within_block=False):
        """
        Blockwise ordering:
        - First choose an order over blocks (block grid),
        - Then choose an order of tokens within each block.
        """
        device = self.encoder_pos_embed_learned.device
        H, W = self.seq_h, self.seq_w

        # compute block grid size
        n_blocks_h = (H + block_h - 1) // block_h
        n_blocks_w = (W + block_w - 1) // block_w

        # list all blocks by (block_row, block_col, block_id)
        blocks = []
        for bh in range(n_blocks_h):
            for bw in range(n_blocks_w):
                block_id = bh * n_blocks_w + bw
                blocks.append((bh, bw, block_id))

        base_block_ids = np.arange(len(blocks))

        orders = []
        for _ in range(bsz):
            # choose block order
            if random_block_order:
                block_order = base_block_ids.copy()
                np.random.shuffle(block_order)
            else:
                block_order = base_block_ids  # fixed scan

            token_order = []
            for bid in block_order:
                bh, bw, _ = blocks[bid]
                h_start = bh * block_h
                w_start = bw * block_w
                h_end = min(h_start + block_h, H)
                w_end = min(w_start + block_w, W)

                # all tokens in this block
                block_tokens = []
                for i in range(h_start, h_end):
                    for j in range(w_start, w_end):
                        idx = i * W + j
                        block_tokens.append(idx)

                block_tokens = np.array(block_tokens, dtype=np.int64)
                if random_within_block:
                    np.random.shuffle(block_tokens)

                token_order.extend(block_tokens.tolist())

            orders.append(token_order)

        # Convert to tensor on the correct device
        return torch.as_tensor(np.stack(orders), device=device, dtype=torch.long)

    def sample_orders_multiscale(self, bsz, strides=(4, 2, 1), shuffle_within_level=True):
        """
        Multiscale center-ish order:
        - At stride s, pick a sub-grid of positions,
        - Sort by center distance within each scale,
        - Coarse scales first, fine scales later.
        """
        device = self.encoder_pos_embed_learned.device
        H, W = self.seq_h, self.seq_w
        cy = (H - 1) / 2.0
        cx = (W - 1) / 2.0

        used = set()
        levels = []  # list of lists of idx

        for s in strides:
            level_idxs = []
            # choose representative positions at stride s
            for i in range(0, H, s):
                for j in range(0, W, s):
                    idx = i * W + j
                    if idx in used:
                        continue
                    used.add(idx)
                    di = i - cy
                    dj = j - cx
                    r2 = di * di + dj * dj
                    level_idxs.append((r2, idx))
            level_idxs.sort(key=lambda x: x[0])
            level_ids = [idx for (_, idx) in level_idxs]
            levels.append(level_ids)

        # any remaining positions not covered (just in case)
        remaining = []
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if idx not in used:
                    di = i - cy
                    dj = j - cx
                    r2 = di * di + dj * dj
                    remaining.append((r2, idx))
        remaining.sort(key=lambda x: x[0])
        rem_ids = [idx for (_, idx) in remaining]
        if rem_ids:
            levels.append(rem_ids)

        # now build per-batch orders
        orders = []
        for _ in range(bsz):
            order = []
            for lvl in levels:
                lvl_arr = np.array(lvl, dtype=np.int64)
                if shuffle_within_level:
                    np.random.shuffle(lvl_arr)
                order.extend(lvl_arr.tolist())
            # To ensure Center-Out (Center revealed first), Center must be at END.
            # Currently, levels are sorted by r2 (Small r2 first).
            # So `order` starts with Center (Small r2) and ends with Edge.
            # Masking logic masks Prefix.
            # So Center is Masked. Edge is Unmasked first.
            # This is Edge-In.
            # To get Center-Out, we reverse the order.
            orders.append(order[::-1])

        return torch.as_tensor(np.stack(orders), device=device, dtype=torch.long)

    def sample_orders_center_out_with_perturb(self, bsz, swap_frac=0.1):
        """
        Center -> edge base order, then per-image random perturbations
        via random swaps.
        """
        # Base Center Out
        # Descending sort of R2 gives [Edge ... Center]. 
        # Center is at Tail -> Revealed First.
        base_order = self.sample_orders_spiral(1, descending=True)[0] # [L]
        
        # Perturbation logic in numpy/cpu for flexibility
        orders = []
        base_np = base_order.cpu().numpy()
        L = len(base_np)
        n_swaps = int(L * swap_frac)
        
        for _ in range(bsz):
            o = base_np.copy()
            for _ in range(n_swaps):
                i = np.random.randint(0, L)
                j = np.random.randint(0, L)
                o[i], o[j] = o[j], o[i]
            orders.append(o)
            
        return torch.as_tensor(np.stack(orders), device=self.encoder_pos_embed_learned.device, dtype=torch.long)

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        
        # Mask the PREFIX. 
        # If Order = [A, B, C, D] and we mask 2.
        # Masked: A, B. Visible: C, D.
        # In inference, we go from All Masked -> Zero Masked.
        # We reveal the SUFFIX first.
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # dropping
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # pad mask tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # decoder position embedding
        x = x_after_pad + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x


    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward(self, imgs, labels, order_mode='random', random_block_order=False, random_within_block=False):
        class_embedding = self.class_emb(labels)
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        
        # Generate orders based on requested mode
        orders = self.sample_orders_new(bsz=x.size(0), mode=order_mode,
                                        random_block_order=random_block_order, 
                                        random_within_block=random_within_block)
        mask = self.random_masking(x, orders)

        x_enc = self.forward_mae_encoder(x, mask, class_embedding)
        z = self.forward_mae_decoder(x_enc, mask)
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False, order_mode='random',
                      random_block_order=False, random_within_block=False):

        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        
        # One-time order generation
        orders = self.sample_orders_new(bsz, mode=order_mode,
                                        random_block_order=random_block_order, 
                                        random_within_block=random_within_block)

        indices = list(range(num_iter))
        if progress: indices = tqdm(indices)
        
        for step in indices:
            cur_tokens = tokens.clone()

            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
            else:
                tokens_in = tokens
                mask_in = mask

            x = self.forward_mae_encoder(tokens_in, mask_in, class_embedding)
            z = self.forward_mae_decoder(x, mask_in)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # Handle CFG batch size mismatch
            if cfg != 1.0:
                orders_inference = torch.cat([orders, orders], dim=0)
            else:
                orders_inference = orders

            mask_next = mask_by_order(mask_len[0], orders_inference, mask_in.shape[0], self.seq_len)
            
            if step >= num_iter - 1:
                mask_to_pred = mask_in.bool()
            else:
                mask_to_pred = torch.logical_xor(mask_in.bool(), mask_next.bool())
            
            mask = mask_next
            
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0) if mask_to_pred.shape[0] != z.shape[0] else mask_to_pred

            z_step = z[mask_to_pred.nonzero(as_tuple=True)]
            
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            else:
                cfg_iter = cfg
                
            sampled_token_latent = self.diffloss.sample(z_step, temperature, cfg_iter)
            
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                mask = mask.chunk(2, dim=0)[0]

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        tokens = self.unpatchify(tokens)
        return tokens

def mar_base(**kwargs):
    model = MAR(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mar_large(**kwargs):
    model = MAR(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
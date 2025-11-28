from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block
from models.diffloss import DiffLoss

def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda()).bool()
    return masking

class MAR(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
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

    def initialize_weights(self):
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p
        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w
        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x

    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders
    
    def sample_raster_orders(self, bsz):
        order = np.arange(self.seq_len)
        orders = np.tile(order, (bsz, 1))
        orders = torch.Tensor(orders).cuda().long()
        return orders

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                             src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks: x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks: x = block(x)
        x = self.encoder_norm(x)
        return x

    def forward_mae_decoder(self, x, mask):
        x = self.decoder_embed(x)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_after_pad = mask_tokens.clone()
        x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        x = x_after_pad + self.decoder_pos_embed_learned

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks: x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks: x = block(x)
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

    # --------------------------------------------------------------------------
    # AR Training Logic
    # --------------------------------------------------------------------------
    def forward_ar(self, imgs, labels, mode='raster'):
        bsz = imgs.shape[0]
        class_embedding = self.class_emb(labels)
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()

        if mode == 'raster':
            orders = self.sample_raster_orders(bsz)
        elif mode == 'random':
            orders = self.sample_orders(bsz)
        else:
            raise ValueError(f"Unknown AR mode: {mode}")

        # --- FIX: Sample ONE scalar timestep for the whole batch ---
        # This ensures every item in the batch has the same number of visible tokens.
        # So 'forward_mae_encoder' can successfully reshape to [Batch, Num_Visible, Dim].
        
        # Sample t from [0, seq_len - 1]
        # t represents the index of the token we are predicting right now.
        # So 0 to t-1 are visible.
        num_visible_scalar = torch.randint(0, self.seq_len, (1,), device=x.device).item()
        
        # 'rank' tells us the position of each token in the order
        rank = torch.argsort(orders, dim=1)
        
        # Mask construction (Same for all batch items in terms of count, different indices)
        # mask = 1 if rank >= num_visible (Hidden)
        # mask = 0 if rank < num_visible (Visible)
        mask = (rank >= num_visible_scalar).float()
        
        # Target: The token at rank == num_visible_scalar
        mask_for_loss = (rank == num_visible_scalar).float()

        # Forward
        x_enc = self.forward_mae_encoder(x, mask, class_embedding)
        z_dec = self.forward_mae_decoder(x_enc, mask)
        
        # Loss
        loss = self.forward_loss(z=z_dec, target=gt_latents, mask=mask_for_loss)
        return loss

    def forward(self, imgs, labels, mode='mar'):
        if mode == 'mar':
            class_embedding = self.class_emb(labels)
            x = self.patchify(imgs)
            gt_latents = x.clone().detach()
            orders = self.sample_orders(bsz=x.size(0))
            mask = self.random_masking(x, orders)
            
            x = self.forward_mae_encoder(x, mask, class_embedding)
            z = self.forward_mae_decoder(x, mask)
            loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
            return loss
            
        elif mode == 'ar_raster':
            return self.forward_ar(imgs, labels, mode='raster')
            
        elif mode == 'ar_random':
            return self.forward_ar(imgs, labels, mode='random')
            
        else:
            raise ValueError(f"Unknown mode {mode}")

    # --------------------------------------------------------------------------
    # 1. Parallel MAR Sampling
    # --------------------------------------------------------------------------
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", labels=None, temperature=1.0, progress=False):
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_orders(bsz)

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
                class_embedding_in = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
            else:
                tokens_in = tokens
                class_embedding_in = class_embedding
                mask_in = mask

            x = self.forward_mae_encoder(tokens_in, mask_in, class_embedding_in)
            z = self.forward_mae_decoder(x, mask_in)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # Handle CFG batch size mismatch
            if cfg != 1.0:
                orders_inf = torch.cat([orders, orders], dim=0)
            else:
                orders_inf = orders

            mask_next = mask_by_order(mask_len[0], orders_inf, mask_in.shape[0], self.seq_len)
            
            if step >= num_iter - 1:
                mask_to_pred = mask_in.bool()
            else:
                mask_to_pred = torch.logical_xor(mask_in.bool(), mask_next.bool())
            
            mask = mask_next
            
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            else:
                cfg_iter = cfg
                
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                mask = mask.chunk(2, dim=0)[0]

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        tokens = self.unpatchify(tokens)
        return tokens
        
    # --------------------------------------------------------------------------
    # 2. Token-by-Token AR Sampling
    # --------------------------------------------------------------------------
    def sample_tokens_ar(self, bsz, mode='raster', cfg=1.0, labels=None, temperature=1.0, progress=False):
        mask = torch.ones(bsz, self.seq_len).cuda() # Start all masked
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()

        if mode == 'ar_raster' or mode == 'raster':
            orders = self.sample_raster_orders(bsz)
        elif mode == 'ar_random' or mode == 'random':
            orders = self.sample_orders(bsz)
        else:
            raise ValueError(f"Unknown AR sampling mode: {mode}")

        indices = list(range(self.seq_len)) 
        if progress: indices = tqdm(indices)
            
        for step in indices:
            cur_tokens = tokens.clone()
            
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
                
            if not cfg == 1.0:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                class_embedding_in = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
            else:
                tokens_in = tokens
                class_embedding_in = class_embedding
                mask_in = mask

            x = self.forward_mae_encoder(tokens_in, mask_in, class_embedding_in)
            z = self.forward_mae_decoder(x, mask_in)

            # Identify target indices (step-th element in order)
            # orders is [B, L]. We want the column at 'step'.
            # But wait: orders[b, step] is the index of the token processed at 'step'.
            if cfg != 1.0:
                # Repeat orders for CFG batch
                orders_inf = torch.cat([orders, orders], dim=0)
            else:
                orders_inf = orders
                
            target_indices = orders_inf[:, step] # [B_cfg]
            
            mask_to_pred = torch.zeros_like(mask_in)
            mask_to_pred.scatter_(1, target_indices.unsqueeze(1), 1.0)
            mask_to_pred = mask_to_pred.bool()

            z_target = z[mask_to_pred.nonzero(as_tuple=True)]
            
            sampled_token_latent = self.diffloss.sample(z_target, temperature, cfg)
            
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                # Update mask only for the real batch (not CFG duplicate)
                target_indices = target_indices[:bsz]

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()
            
            # Unmask
            mask.scatter_(1, target_indices.unsqueeze(1), 0.0)

        return self.unpatchify(tokens)

def mar_base(**kwargs):
    return MAR(encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
               decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

def mar_large(**kwargs):
    return MAR(encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
               decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
               mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
def mar_huge(**kwargs):
    model = MAR(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

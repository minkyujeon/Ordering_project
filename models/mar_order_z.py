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
    """
    Constructs binary mask (1=Masked, 0=Visible).
    Indices in order[:, :mask_len] are masked.
    Indices in order[:, mask_len:] are visible.
    """
    device = order.device
    if isinstance(mask_len, torch.Tensor):
        mask_len = int(mask_len.item())
    else:
        mask_len = int(mask_len)

    masking = torch.zeros(bsz, seq_len, device=device)
    index = order[:, :mask_len] 
    src = torch.ones_like(index, dtype=masking.dtype, device=device)
    masking = masking.scatter(dim=-1, index=index, src=src)
    return masking.bool()


class TimmOrderHead(nn.Module):
    """
    Transformer-based Order Predictor.
    Input: Contextualized Features (Decoder Output) [B, L, Dim]
    Output: Priority Scores [B, L]
    """
    def __init__(self, token_dim, depth=2, num_heads=4, mlp_ratio=4.0, 
                 attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(token_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, token_dim)) 

        self.blocks = nn.ModuleList([
            Block(dim=token_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=True, norm_layer=nn.LayerNorm, 
                  proj_drop=proj_dropout, attn_drop=attn_dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        x = self.norm_in(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(x.size(0), x.size(1), -1)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm_out(x)
        return self.head(x).squeeze(-1) # [B, L]


class MAR(nn.Module):
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

        self.vae_embed_dim = vae_embed_dim
        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.label_drop_prob = label_drop_prob
        
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25)

        # Encoder
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))
        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        # Order Head
        self.order_head = TimmOrderHead(
            token_dim=decoder_embed_dim,
            depth=4,
            num_heads=4,
            mlp_ratio=4,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout
        )

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
        
        self.initialize_weights()

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

    def sample_random_orders(self, bsz):
        device = self.encoder_pos_embed_learned.device
        orders = []
        for _ in range(bsz):
            order = np.arange(self.seq_len)
            np.random.shuffle(order)
            orders.append(order)
        return torch.as_tensor(np.stack(orders), device=device, dtype=torch.long)

    def random_masking(self, x, orders):
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        mask = torch.zeros(bsz, seq_len, device=x.device)
        mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens], src=torch.ones(bsz, seq_len, device=x.device))
        return mask

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        if self.training:
            drop_latent_mask = torch.rand(bsz, device=x.device) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        # Drop Masked Tokens
        x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)

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

        # Restore Mask Tokens
        mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
        x_full = mask_tokens.clone()
        x_full[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        x_full = x_full + self.decoder_pos_embed_learned

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x_full = checkpoint(block, x_full)
        else:
            for block in self.decoder_blocks:
                x_full = block(x_full)
        x_full = self.decoder_norm(x_full)

        x_out = x_full[:, self.buffer_size:]
        x_out = x_out + self.diffusion_pos_embed_learned
        return x_out

    def forward_loss(self, z, target, mask):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    # --------------------------------------------------------------------------
    # 1. TRAINING LOGIC (One-Shot Gumbel)
    # --------------------------------------------------------------------------
    def forward_oneshot_gumbel(self, imgs, labels, gumbel_scale=1.0):
        x_tokens = self.patchify(imgs)
        gt_latents = x_tokens.clone().detach()
        
        with torch.no_grad():
            gt_latents_proj = self.z_proj(gt_latents)

        orders = self.sample_random_orders(bsz=imgs.size(0))
        mask = self.random_masking(x_tokens, orders) 

        with torch.no_grad():
            if labels is None:
                class_embedding = self.fake_latent.repeat(imgs.shape[0], 1)
            else:
                class_embedding = self.class_emb(labels)
            
            x_enc = self.forward_mae_encoder(x_tokens, mask, class_embedding)
            z_dec = self.forward_mae_decoder(x_enc, mask)

            # Compute Error in FP32 to avoid underflow/overflow
            z_dec_f32 = z_dec.float()
            gt_latents_proj_f32 = gt_latents_proj.float()
            recon_error = torch.mean((z_dec_f32 - gt_latents_proj_f32)**2, dim=-1) 
            
            mean_err = recon_error.mean(dim=1, keepdim=True)
            std_err = recon_error.std(dim=1, keepdim=True) + 1e-6
            normalized_error = (recon_error - mean_err) / std_err

        scores = self.order_head(z_dec.detach()) 
        
        # Cast Scores to FP32 for Gumbel/Softmax
        masked_scores = scores.float().masked_fill(~mask.bool(), -1e4)
        
        # Generate Gumbel noise in FP32
        u = torch.rand_like(masked_scores)
        g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        
        # Softmax in FP32
        selection_probs = torch.softmax(masked_scores + gumbel_scale * g, dim=-1)
        
        # Loss in FP32
        loss_ordering = (selection_probs * normalized_error).sum(dim=-1).mean()
        
        return loss_ordering

    def forward(self, imgs, labels, order_mode='random', **kwargs):
        if order_mode == 'gradient':
            return self.forward_oneshot_gumbel(imgs, labels, **kwargs)
        else:
            class_embedding = self.class_emb(labels)
            x = self.patchify(imgs)
            gt_latents = x.clone().detach()
            orders = self.sample_random_orders(bsz=x.size(0))
            mask = self.random_masking(x, orders)
            x_enc = self.forward_mae_encoder(x, mask, class_embedding)
            z = self.forward_mae_decoder(x_enc, mask)
            loss = self.forward_loss(z=z, target=gt_latents, mask=mask)
            return loss

    # --------------------------------------------------------------------------
    # 2. INFERENCE LOGIC
    # --------------------------------------------------------------------------
    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="linear", 
                      labels=None, temperature=1.0, progress=False, order_mode='random'):
        
        if order_mode == 'random':
            return self._sample_tokens_random(bsz, num_iter, cfg, cfg_schedule, labels, temperature, progress)

        device = self.fake_latent.device
        mask = torch.ones(bsz, self.seq_len, device=device)
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim, device=device)
        
        indices = list(range(num_iter))
        if progress: indices = tqdm(indices)

        for step in indices:
            cur_tokens = tokens.clone()
            
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)

            if cfg != 1.0:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                class_emb_in = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_in = torch.cat([mask, mask], dim=0)
            else:
                tokens_in = tokens
                class_emb_in = class_embedding
                mask_in = mask

            x_enc = self.forward_mae_encoder(tokens_in, mask_in, class_emb_in)
            z_dec = self.forward_mae_decoder(x_enc, mask_in)

            scores = self.order_head(z_dec) 

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            target_k = int(np.floor(self.seq_len * mask_ratio))
            target_k = max(1, min(int(mask[0].sum().item()) - 1, target_k))
            
            # FIX: Cast to float32 for sorting to avoid potential FP16 overflow with +1e4
            sort_key = scores.float() + (1 - mask_in.float()) * 1e4
            argsort = torch.argsort(sort_key, dim=-1, descending=True)
            
            hardest_indices = argsort[:, -target_k:]
            mask_next = torch.zeros_like(mask_in)
            mask_next.scatter_(1, hardest_indices, 1.0)
            
            if step >= num_iter - 1:
                mask_to_pred = mask_in.bool()
            else:
                mask_to_pred = (mask_in.bool() & (~mask_next.bool()))
            mask = mask_next[:bsz]

            z_sample = z_dec[mask_to_pred.nonzero(as_tuple=True)]
            
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - target_k) / self.seq_len
            else:
                cfg_iter = cfg
                
            sampled = self.diffloss.sample(z_sample, temperature, cfg_iter)
            
            if cfg != 1.0:
                sampled, _ = sampled.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                mask = mask.chunk(2, dim=0)[0]
                
            tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled
            
        return self.unpatchify(tokens)

    def _sample_tokens_random(self, bsz, num_iter, cfg, cfg_schedule, labels, temperature, progress):
        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()
        orders = self.sample_random_orders(bsz)

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        for step in indices:
            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if not cfg == 1.0:
                tokens = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            x = self.forward_mae_encoder(tokens, mask, class_embedding)

            # mae decoder
            z = self.forward_mae_decoder(x, mask)

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(torch.Tensor([1]).cuda(),
                                     torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = mask_by_order(mask_len[0], orders, bsz, self.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError
            sampled_token_latent = self.diffloss.sample(z, temperature, cfg_iter)
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
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
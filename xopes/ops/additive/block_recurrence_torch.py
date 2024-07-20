class BlockRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, BLOCK=256, mask=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        # test
        b, h, n, d = q.shape
        e = v.shape[-1]
        # print(b, h, n, d, e)
        # BLOCK = n
        NUM_BLOCK = n // BLOCK
        o = torch.zeros((b, h, n, e), dtype=q.dtype, device=q.device)
        # print(s.shape, use_decay)

        g_max = torch.max(g, dim=-2, keepdim=True).values
        if mask == None:
            mask = (torch.tril(torch.ones(BLOCK, BLOCK))).to(q)

        # other
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        # k = k
        g_sum = 0
        g_cumsum = []
        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = si + BLOCK

            gi = torch.exp(g[:, :, si:ei].contiguous().to(torch.float32) - g_max)
            qi = q[:, :, si:ei].contiguous().to(torch.float32)
            ki = k[:, :, si:ei].contiguous().to(torch.float32) * gi
            vi = v[:, :, si:ei].contiguous().to(torch.float32)

            gi_cumsum = g_sum + torch.cumsum(gi, dim=-2)
            qi = qi / gi_cumsum

            # print(qi.dtype, q_decay.dtype, kv.dtype)
            o_inter = torch.matmul(qi.to(kv.dtype), kv).to(torch.float32)

            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * mask
            o_intra = torch.matmul(qk, vi.to(torch.float32))

            o[:, :, si:ei] = o_intra + o_inter
            kv += torch.matmul(ki.transpose(-1, -2).to(vi.dtype), vi)

            g_sum = gi_cumsum[:, :, -1:]
            g_cumsum.append(gi_cumsum)

        g_cumsum = torch.cat(g_cumsum, dim=-2)

        ctx.save_for_backward(q, k, v, g, mask, g_max, g_cumsum)
        ctx.BLOCK = BLOCK
        ctx.NUM_BLOCK = NUM_BLOCK

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, g, mask, g_max, g_cumsum = ctx.saved_tensors
        BLOCK = ctx.BLOCK
        NUM_BLOCK = ctx.NUM_BLOCK
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        g = g.contiguous()

        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        dg = torch.zeros_like(g, dtype=torch.float32)

        b, h, n, d = q.shape
        e = v.shape[-1]

        # q
        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        # k = k - k_max
        # k_sum = 0
        for i in range(0, NUM_BLOCK):
            si = i * BLOCK
            ei = si + BLOCK
            gi = torch.exp(g[:, :, si:ei].contiguous().to(torch.float32) - g_max)
            gi_cumsum = g_cumsum[:, :, si:ei].contiguous()
            # k_cumsum = k_sum + torch.cumsum(ki, dim=-2)
            # ki = ki
            ki = k[:, :, si:ei].contiguous().to(torch.float32) * gi
            vi = v[:, :, si:ei].contiguous().to(torch.float32)
            doi = do[:, :, si:ei].contiguous().to(torch.float32)

            dq_inter = torch.matmul(
                doi.to(kv.dtype).to(torch.float32), kv.transpose(-1, -2)
            )

            dqk = torch.matmul(doi, vi.transpose(-1, -2)) * mask
            dq_intra = torch.matmul(dqk, ki)

            dq[:, :, si:ei] = (dq_intra + dq_inter) / gi_cumsum
            kv += torch.matmul(ki.transpose(-1, -2).to(vi.dtype), vi)

        # k, v
        dkv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        ds = 0
        # ds_cumsum = torch.zeros(b, h, BLOCK, d)
        for i in range(NUM_BLOCK - 1, -1, -1):
            si = i * BLOCK
            ei = si + BLOCK

            gi = torch.exp(g[:, :, si:ei].contiguous().to(torch.float32) - g_max)
            ki_ = k[:, :, si:ei].contiguous().to(torch.float32)
            ki = ki_ * gi
            gi_cumsum = g_cumsum[:, :, si:ei].contiguous()
            qi = q[:, :, si:ei].contiguous().to(torch.float32) / gi_cumsum
            # k_cumsum_reverse = k_sum_reverse + torch.cumsum(torch.flip(ki, dims=(-2,)), dim=-2)
            vi = v[:, :, si:ei].contiguous().to(torch.float32)
            doi = do[:, :, si:ei].contiguous().to(torch.float32)
            # qi = qi

            # dv
            qk = torch.matmul(qi, ki.transpose(-1, -2)) * mask
            dv_intra = torch.matmul(qk.transpose(-1, -2), doi)
            dv_inter = torch.matmul(ki.to(dkv.dtype), dkv)
            dv[:, :, si:ei] = dv_intra + dv_inter

            # dk
            dk_inter = torch.matmul(vi.to(dkv.dtype), dkv.transpose(-1, -2))
            dqk = torch.matmul(doi, vi.transpose(-1, -2)) * mask
            dk_intra = torch.matmul(dqk.transpose(-1, -2), qi)
            # dki = dk_intra + dk_inter
            # print(dki.shape, ds_cumsum.shape, ki.shape, si, ei, (dki * ki).shape, dk[:, :, si:ei].shape, dk.shape)
            dki = (dk_intra + dk_inter) * gi
            dk[:, :, si:ei] = dki

            # ds
            dqi = dq[:, :, si:ei].to(torch.float32)
            ds_current = dqi * qi  # qi = qi_exact / gi_cumsum
            # print(dqi_.shape, qi.shape, ki_cumsum.shape)
            # https://github.com/pytorch/pytorch/issues/33520
            # ds_reverse_cumsum = ds + torch.flip(torch.cumsum(torch.flip(ds_current, dims=(-2,)), dim=-2), dims=(-2,))
            ds_cumsum = torch.cumsum(ds_current, dim=-2)
            ds_reverse_cumsum = ds + ds_current - ds_cumsum + ds_cumsum[:, :, -1:None]
            dg[:, :, si:ei] = dki * ki_ - gi * ds_reverse_cumsum

            dkv += torch.matmul(qi.transpose(-1, -2), doi)
            ds = ds_reverse_cumsum[:, :, :1]
            # k_sum_reverse = k_cumsum[:, :, :1]

        return dq, dk, dv, dg, None, None

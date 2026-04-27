import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    return x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist.clamp(min=1e-12).sqrt()


def hard_example_mining(dist_mat, labels):
    n = dist_mat.size(0)
    is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    is_neg = labels.expand(n, n).ne(labels.expand(n, n).t())
    eye = torch.eye(n, device=dist_mat.device, dtype=dist_mat.dtype)
    is_pos_masked = is_pos.to(dist_mat.dtype) * (1.0 - eye)
    dist_ap = torch.max(dist_mat * is_pos_masked, dim=1)[0]
    masked_neg = dist_mat * is_neg.to(dist_mat.dtype) + torch.max(
        dist_mat) * (1.0 - is_neg.to(dist_mat.dtype))
    dist_an = torch.min(masked_neg, dim=1)[0]
    return dist_ap, dist_an


class TripletLoss(object):

    def __init__(self, margin=0.15):
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin,
                                                 reduction='none')

    def __call__(self, global_feat, labels, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        y = dist_an.new_ones(dist_an.size())
        loss = self.ranking_loss(dist_an, dist_ap, y)
        is_eol_mask = (1.0 - (labels == 0).float()).view(-1)
        denom = is_eol_mask.sum().clamp_min(1.0)
        loss_masked = loss * is_eol_mask
        loss_mean = loss_masked.sum() / denom
        valid_num = (loss_masked > 0).float().sum()
        valid_sum = is_eol_mask.sum()
        return loss_mean, valid_num, valid_sum


class Bottleneck(nn.Module):

    def __init__(self, n_channels, growth_rate, use_dropout):
        super().__init__()
        inter_channels = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv1 = nn.Conv2d(n_channels,
                               inter_channels,
                               kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(inter_channels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return torch.cat((x, out), 1)


class SingleLayer(nn.Module):

    def __init__(self, n_channels, growth_rate, use_dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels,
                               growth_rate,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        return torch.cat((x, out), 1)


class Transition(nn.Module):

    def __init__(self, n_channels, out_channels, use_dropout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(n_channels,
                               out_channels,
                               kernel_size=1,
                               bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        return F.avg_pool2d(out, 2, ceil_mode=True)


class DenseNet(nn.Module):

    def __init__(self, growth_rate, reduction, bottleneck, use_dropout):
        super().__init__()
        n_dense_blocks = 22
        n_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(1,
                               n_channels,
                               kernel_size=7,
                               padding=3,
                               stride=2,
                               bias=False)
        self.dense1 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans1 = Transition(n_channels, out_channels, use_dropout)

        n_channels = out_channels
        self.dense2 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout)
        n_channels += n_dense_blocks * growth_rate
        out_channels = int(math.floor(n_channels * reduction))
        self.trans2 = Transition(n_channels, out_channels, use_dropout)

        n_channels = out_channels
        self.dense3 = self._make_dense(n_channels, growth_rate, n_dense_blocks,
                                       bottleneck, use_dropout)
        self.out_channels = n_channels + n_dense_blocks * growth_rate

    def _make_dense(self, n_channels, growth_rate, n_dense_blocks, bottleneck,
                    use_dropout):
        layers = []
        for _ in range(int(n_dense_blocks)):
            if bottleneck:
                layers.append(Bottleneck(n_channels, growth_rate, use_dropout))
            else:
                layers.append(SingleLayer(n_channels, growth_rate, use_dropout))
            n_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x, x_mask):
        out = self.conv1(x)
        out_mask = x_mask[:, 0::2, 0::2]
        out = F.relu(out, inplace=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense2(out)
        out = self.trans2(out)
        out_mask = out_mask[:, 0::2, 0::2]
        out = self.dense3(out)
        return out, out_mask


class FCLayer(nn.Module):

    def __init__(self, nin, nout):
        super().__init__()
        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class TANEmbedding(nn.Module):

    def __init__(self, vocab_size, rel_vocab_size, dim, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.re_embedding = nn.Embedding(rel_vocab_size, dim)
        pos_embedding = torch.zeros(max_len, dim)
        for pos in range(max_len):
            for i in range(dim // 2):
                scale = 10000**(2.0 * i / dim)
                pos_embedding[pos, 2 * i] = math.sin(float(pos) / scale)
                pos_embedding[pos, 2 * i + 1] = math.cos(float(pos) / scale)
        self.register_buffer('pos_embedding', pos_embedding, persistent=False)

    def word_emb(self, y):
        if y.sum() < 0:
            return self.embedding.weight.new_zeros((1, self.embedding.embedding_dim))
        return self.embedding(y)

    def re_emb(self, re):
        if re.sum() < 0:
            return self.re_embedding.weight.new_zeros(
                (1, self.re_embedding.embedding_dim))
        return self.re_embedding(re)


class GruCondLayer(nn.Module):

    def __init__(self, params):
        super().__init__()
        dim = params['m']
        hidden = params['n']
        feat = params['D']
        attn = params['dim_attention']

        self.fc_wyz0 = nn.Linear(hidden, hidden)
        self.fc_wyr0 = nn.Linear(hidden, hidden)
        self.fc_wyh0 = nn.Linear(hidden, hidden)
        self.fc_uhz0 = nn.Linear(hidden, hidden, bias=False)
        self.fc_uhr0 = nn.Linear(hidden, hidden, bias=False)
        self.fc_uhh0 = nn.Linear(hidden, hidden, bias=False)

        self.fc_wyz = nn.Linear(2 * dim, hidden)
        self.fc_wyr = nn.Linear(2 * dim, hidden)
        self.fc_wyh = nn.Linear(2 * dim, hidden)
        self.fc_uhz = nn.Linear(hidden, hidden, bias=False)
        self.fc_uhr = nn.Linear(hidden, hidden, bias=False)
        self.fc_uhh = nn.Linear(hidden, hidden, bias=False)

        self.conv_uac = nn.Conv2d(feat, attn, kernel_size=1)
        self.fc_wac = nn.Linear(hidden, attn, bias=False)
        self.conv_qc = nn.Conv2d(1, 512, kernel_size=3, bias=False, padding=1)
        self.fc_ufc = nn.Linear(512, attn)
        self.fc_vac = nn.Linear(attn, 1)

        self.fc_wcz = nn.Linear(feat, hidden, bias=False)
        self.fc_wcr = nn.Linear(feat, hidden, bias=False)
        self.fc_wch = nn.Linear(feat, hidden, bias=False)
        self.fc_uhz2 = nn.Linear(hidden, hidden)
        self.fc_uhr2 = nn.Linear(hidden, hidden)
        self.fc_uhh2 = nn.Linear(hidden, hidden)

    def _step_slice(self, ly_mask, ctx_mask, h_prev, calpha_past_prev, pctx,
                    context, state_below_z, state_below_r, state_below_h):
        z0 = torch.sigmoid(self.fc_uhz0(h_prev) + state_below_z)
        r0 = torch.sigmoid(self.fc_uhr0(h_prev) + state_below_r)
        h0_p = torch.tanh(self.fc_uhh0(h_prev) * r0 + state_below_h)
        h0 = z0 * h_prev + (1.0 - z0) * h0_p
        h0 = ly_mask[:, None] * h0 + (1.0 - ly_mask)[:, None] * h_prev

        query_child = self.fc_wac(h0)
        cover_fc = self.conv_qc(calpha_past_prev[:, None]).permute(2, 3, 0, 1)
        cover_vector = self.fc_ufc(cover_fc)
        attention_score = torch.tanh(pctx + query_child[None, None] +
                                     cover_vector)
        calpha = self.fc_vac(attention_score)
        calpha = calpha - calpha.max()
        calpha = torch.exp(calpha.view(calpha.shape[0], calpha.shape[1],
                                       calpha.shape[2]))
        if ctx_mask is not None:
            calpha = calpha * ctx_mask.permute(1, 2, 0)
        denom = calpha.sum(1).sum(0)[None, None, :].clamp_min(1e-10)
        calpha = calpha / denom
        calpha_past = calpha_past_prev + calpha.permute(2, 0, 1)
        ct_c = (context * calpha.permute(2, 0, 1)[:, None]).sum(3).sum(2)

        z2 = torch.sigmoid(self.fc_uhz2(h0) + self.fc_wcz(ct_c))
        r2 = torch.sigmoid(self.fc_uhr2(h0) + self.fc_wcr(ct_c))
        h2_p = torch.tanh(self.fc_uhh2(h0) * r2 + self.fc_wch(ct_c))
        h2 = z2 * h0 + (1.0 - z2) * h2_p
        h2 = ly_mask[:, None] * h2 + (1.0 - ly_mask)[:, None] * h0
        return h2, ct_c, calpha.permute(2, 0, 1), calpha_past

    def forward(self, params, rembedding, re_embedding, rp, ly_mask, context,
                context_mask, init_state):
        n_steps = rembedding.shape[0]
        n_samples = rembedding.shape[1]
        device = rembedding.device

        pctx = self.conv_uac(context).permute(2, 3, 0, 1)
        emb = torch.cat((rembedding, re_embedding), 2)
        state_below_z = self.fc_wyz(emb)
        state_below_r = self.fc_wyr(emb)
        state_below_h = self.fc_wyh(emb)

        calpha_past = rembedding.new_zeros((n_samples, context.shape[2],
                                            context.shape[3]))
        h2ts = rembedding.new_zeros((n_steps + 1, n_samples, params['n']))
        h2ts[0] = init_state
        h2t = init_state

        ct_cs = rembedding.new_zeros((n_steps + 1, n_samples, params['D']))
        ct_ps = rembedding.new_zeros((n_steps + 1, n_samples, params['D']))

        for i in range(n_steps):
            rpos = rp[i]
            ct_p = torch.stack(
                [ct_cs[int(rpos[j].item()), j, :] for j in range(n_samples)], 0)
            h2t, ct_c, _calpha, calpha_past = self._step_slice(
                ly_mask[i], context_mask, h2t, calpha_past, pctx, context,
                state_below_z[i], state_below_r[i], state_below_h[i])
            h2ts[i + 1] = h2t
            ct_cs[i + 1] = ct_c
            ct_ps[i + 1] = ct_p

        return h2ts[1:], ct_cs[1:], ct_ps[1:]


class GruProb(nn.Module):

    def __init__(self, params):
        super().__init__()
        self.fc_wctc = nn.Linear(params['D'], params['m'])
        self.fc_whtc = nn.Linear(params['n'], params['m'])
        self.fc_wytc = nn.Linear(params['m'], params['m'])
        self.dropout = nn.Dropout(p=0.2)
        self.fc_w0c = nn.Linear(params['m'] // 2, params['K'])
        self.triplet_loss_func = TripletLoss(margin=0.15)

    def forward(self, ct_cs, ht_cs, prev_c, y, use_dropout):
        clogit = self.fc_wctc(ct_cs) + self.fc_whtc(ht_cs) + self.fc_wytc(
            prev_c)
        cfeats = clogit.view(-1, clogit.shape[2])
        y_flatten = y.view(-1, 1)
        triplet_loss, valid_num, valid_sum = self.triplet_loss_func(
            cfeats, y_flatten)

        if clogit.dim() == 2:
            clogit = clogit.unsqueeze(0)
        cshape = clogit.shape
        clogit = clogit.view(cshape[0], cshape[1], cshape[2] // 2, 2).max(3)[0]
        if use_dropout:
            clogit = self.dropout(clogit)
        cprob = self.fc_w0c(clogit)
        return cprob, triplet_loss, valid_num, valid_sum


class TANDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 relation_dict_path,
                 n=256,
                 m=256,
                 d=936,
                 dim_attention=512,
                 max_len=35,
                 growth_rate=24,
                 reduction=0.5,
                 bottleneck=True,
                 use_dropout=True,
                 lpred_loss_weight=1.0,
                 rrepred_loss_weight=0.5,
                 triplet_loss_weight=0.05,
                 **kwargs):
        super().__init__()
        self.params = {
            'n': int(n),
            'm': int(m),
            'D': int(d),
            'dim_attention': int(dim_attention),
            'K': int(out_channels),
            'Kre': self._count_vocab(relation_dict_path),
            'rre': self._count_vocab(relation_dict_path),
            'm_re': int(m),
            'mre': int(m),
            'maxlen': int(max_len),
            'growthRate': int(growth_rate),
            'reduction': float(reduction),
            'bottleneck': bool(bottleneck),
            'use_dropout': bool(use_dropout),
            'lpred_loss': float(lpred_loss_weight),
            'rrepred_loss': float(rrepred_loss_weight),
            'triplet': float(triplet_loss_weight),
        }

        self.encoder = DenseNet(
            growth_rate=self.params['growthRate'],
            reduction=self.params['reduction'],
            bottleneck=self.params['bottleneck'],
            use_dropout=self.params['use_dropout'],
        )
        self.init_gru_model = FCLayer(self.params['D'], self.params['n'])
        self.emb_model = TANEmbedding(self.params['K'], self.params['Kre'],
                                      self.params['m'], self.params['maxlen'])
        self.gru_model = GruCondLayer(self.params)
        self.gru_prob_model = GruProb(self.params)
        self.wre = nn.Linear(2 * self.params['D'], self.params['rre'])
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.apply(self._weight_init)

    @staticmethod
    def _count_vocab(dict_path):
        path = Path(dict_path)
        if not path.exists():
            raise FileNotFoundError(f'Dictionary not found: {dict_path}')
        return sum(1 for line in path.read_text(encoding='utf-8').splitlines()
                   if line.strip())

    @staticmethod
    def _weight_init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight.data)
            if getattr(module, 'bias', None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, x, data=None):
        if data is None or len(data) < 10:
            raise ValueError(
                'TANDecoder currently requires structured training/eval labels '
                '(label, length, ly, ly_mask, ry, ry_mask, lp, rp, re, rre, rre_mask).'
            )

        if len(data) >= 11:
            _label, _length, ly, ly_mask, ry, _ry_mask, _lp, rp, re, rre, rre_mask = data[:11]
        else:
            _label, _length, ly, ly_mask, ry, _ry_mask, _lp, rp, re, rre = data[:10]
            rre_mask = ly_mask.clone()
            rre_mask[0] = 0
            rre_mask[-1] = 0

        if ly.dim() == 2 and ly.shape[0] == x.shape[0]:
            ly = ly.transpose(0, 1).contiguous()
            ly_mask = ly_mask.transpose(0, 1).contiguous()
            ry = ry.transpose(0, 1).contiguous()
            rp = rp.transpose(0, 1).contiguous()
            re = re.transpose(0, 1).contiguous()
            rre = rre.transpose(0, 1).contiguous()
            rre_mask = rre_mask.transpose(0, 1).contiguous()

        x_mask = x.new_ones((x.shape[0], x.shape[2], x.shape[3]))
        ctx, ctx_mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * ctx_mask[:, None]).sum(3).sum(2) / ctx_mask.sum(
            2).sum(1, keepdim=True).clamp_min(1e-6)
        init_state = self.init_gru_model(ctx_mean)

        remb = self.emb_model.word_emb(ry)
        re_emb = self.emb_model.re_emb(re)
        h2ts, ct_cs, ct_ps = self.gru_model(
            self.params,
            remb,
            re_emb,
            rp,
            ly_mask,
            ctx,
            ctx_mask,
            init_state,
        )
        relation_logits = self.wre(torch.cat((ct_ps, ct_cs), 2))
        char_logits, triplet_loss, valid_num, valid_sum = self.gru_prob_model(
            ct_cs, h2ts, remb, ly, use_dropout=self.params['use_dropout'])

        char_flat = char_logits.contiguous().view(-1, char_logits.shape[2])
        relation_flat = relation_logits.contiguous().view(
            -1, relation_logits.shape[2])
        ly_flat = ly.contiguous().view(-1)
        rre_flat = rre.contiguous().view(-1)

        lpred_loss = self.criterion(char_flat, ly_flat).view_as(ly)
        lpred_loss = (lpred_loss * ly_mask).sum(0) / ly_mask.sum(0).clamp_min(
            1e-10)
        lpred_loss = lpred_loss.mean()

        rre_pred_loss = self.criterion(relation_flat, rre_flat).view_as(rre)
        rre_pred_loss = (rre_pred_loss * rre_mask).sum(0) / rre_mask.sum(
            0).clamp_min(1e-10)
        rre_pred_loss = rre_pred_loss.mean()

        loss = (self.params['lpred_loss'] * lpred_loss +
                self.params['rrepred_loss'] * rre_pred_loss +
                self.params['triplet'] * triplet_loss)

        return {
            'loss': loss,
            'lpred_loss': lpred_loss,
            'rrepred_loss': rre_pred_loss,
            'triplet_loss': triplet_loss,
            'valid_num': valid_num,
            'valid_sum': valid_sum,
            'char_logits': char_logits,
            'relation_logits': relation_logits,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from .SeqContext import SeqContext
from .GNN import GNN
from .Classifier import Classifier
from .functions import batch_graphify, batch_graphify_multimodal
import gccl

log = gccl.utils.get_logger()


class GCCL(nn.Module):
    def __init__(self, args):
        super(GCCL, self).__init__()

        dataset_label_dict = {
            "iemocap": {"hap": 0, "sad": 1, "neu": 2, "ang": 3, "exc": 4, "fru": 5},
        }

        tag_size = len(dataset_label_dict[args.dataset])

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        edge_type_to_idx = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

        ### ------------------------------------add------------------------------------
        u_dim = 1380
        g_dim = args.hidden_size
        self.rnn = SeqContext(u_dim, g_dim, args)

        self.num_view = 3
        self.view_list = [100, 768, 512]
        self.feature_dim = 128
        self.num_classes = 6
        self.M = args.M
        self.token_num = 16
        self.modal_embeddings = nn.Embedding(self.num_view, self.feature_dim)

        h1_dim = args.hidden_size
        h2_dim = args.hidden_size

        # self.gcn = GNN(g_dim, h1_dim, h2_dim, args)

        self.encoders = [] # dim encoder
        for v in range(self.num_view):
            self.encoders.append(Encoder(self.view_list[v], self.feature_dim).to(self.device))
        self.encoders = nn.ModuleList(self.encoders)

        self.uni_encoders = []  # dim encoder
        for v in range(self.num_view):
            self.uni_encoders.append(Encoder(self.view_list[v], self.feature_dim).to(self.device))
        self.uni_encoders = nn.ModuleList(self.uni_encoders)

        self.modal_gcn = GNN(g_dim, h1_dim, h2_dim, args)

        self.consensus_prompts = nn.Parameter(torch.randn(self.num_view, self.token_num, self.feature_dim))

        self.aware_layers = []
        for v in range(self.num_view):
            self.aware_layer = SeqContext(self.feature_dim, self.feature_dim // 2, args)
            self.aware_layers.append(self.aware_layer.to(self.device))
        self.aware_layers = nn.ModuleList(self.aware_layers)

        self.Multi_Center_Loss = My_anchor_Loss(self.num_view, self.num_classes, self.feature_dim, self.M)

        self.clf = Classifier(g_dim + self.feature_dim // 2 * 3 + h2_dim * 3, g_dim, tag_size, args)
        
    
    def get_rep(self, data):
        # [batch_size, mx_len, D_g]
        node_features = self.rnn(data["text_len_tensor"], data["input_tensor"])
        features, edge_index, edge_type, edge_index_lengths = batch_graphify(
            node_features,
            data["text_len_tensor"],
            data["speaker_tensor"],
            self.wp,
            self.wf,
            self.edge_type_to_idx,
            self.device,
        )

        return features#, graph_out

    def get_cross(self, data, out_features):
        # ----------------------------------------------------------------------------------
        # unimodal features learning
        data_unimodal = data['unimodal_tensor']
        out_dims = [] 
        emb_idx = torch.LongTensor([0, 1, 2]).cuda()
        emb_vector = self.modal_embeddings(emb_idx)
        for v in range(self.num_view):
            out_dim = self.uni_encoders[v](data_unimodal[v])
            out_dim += emb_vector[v].reshape(1, -1).expand(out_dim.shape[0], out_dim.shape[1], out_dim.shape[2])
            out_dims.append(out_dim)

        uni_graph_outs = []
        for v in range(self.num_view):
            uni_feature, edge_index, edge_type, edge_index_lengths = batch_graphify_multimodal(
                out_dims[v],
                data["text_len_tensor"],
                data["speaker_tensor"],
                self.wp,
                self.wf,
                self.edge_type_to_idx,
                self.device,
            )
            uni_features = torch.cat([out_features, uni_feature], dim=0)

            graph_out = self.modal_gcn(uni_features, edge_index, edge_type)
            bsz_length_total = sum(data["text_len_tensor"])
            uni_graph_outs.append(graph_out[:bsz_length_total, :])
        # ----------------------------------------------------------------------------------
        return uni_graph_outs
    
    def get_aware(self, data):
        
        data_a = data['input_tensor'][:, :, :100]
        data_t = data['input_tensor'][:, :, 100:868]
        data_v = data['input_tensor'][:, :, 868:]
        data_unimodal = []
        data_unimodal.append(data_a)
        data_unimodal.append(data_t)
        data_unimodal.append(data_v)

        out_dims = []
        for v in range(self.num_view):
            out_dim = self.encoders[v](data_unimodal[v])
            out_dims.append(out_dim)
        
        out_feats = torch.cat(out_dims, dim=-1)
        data['input_tensor'] = out_feats
        batch_diag_len = data['text_len_tensor']
        
        bsz, ndiag, feat_dim = out_dims[0].size()
        out_diag_views = []
        for v in range(self.num_view):
            tmps = []
            for b in range(bsz):
                tmp = out_dims[v][b, :batch_diag_len[b], :]
                tmps.append(tmp)
            out_diag_views.append(torch.cat(tmps, dim=0))
        
        aware_features = []
        for v in range(self.num_view):
            out_diag_view = out_diag_views[v]
            out_diag_view = out_diag_view.unsqueeze(1)
            view_consensus_prompt = self.consensus_prompts[v, :, :]
            view_consensus_prompts = view_consensus_prompt.repeat(out_diag_view.size(0), 1, 1)
            view_input = torch.cat([view_consensus_prompts, out_diag_view], dim=1)
            tmp_output = self.aware_layers[v](data["text_len_tensor"], view_input)
            aware_features.append(tmp_output[:, -1, :])
        
        return data_unimodal, aware_features
        
    def get_fine_supcontrastive_loss(self, data):
        
        aware_features = data['aware_tensor']
        batch_label = data['label_tensor'] 
        diag_len = data['text_len_tensor']
        speker_seq = data['speaker_tensor']

        diag_index = []
        spk_index = []
        for ndiag in range(len(diag_len)):
            num_diag = diag_len[ndiag]
            diag_index.append(ndiag * torch.ones(num_diag))
            spk_index.append(speker_seq[ndiag][:num_diag])
        diag_index = torch.cat(diag_index).long()
        spk_index = torch.cat(spk_index)

        diag_sample = [str(i) + str(j) +str(k) for i, j, k in zip(diag_index.tolist(), spk_index.tolist(), batch_label.tolist())]

        sample_label = []
        label_dict = {}

        dlabel = 0
        for item in diag_sample:
            if item not in label_dict:
                label_dict[item] = dlabel
                dlabel += 1
            sample_label.append(label_dict[item])

        sample_label = torch.tensor(sample_label).to(self.device)

        loss = SupConLoss(features=torch.cat(aware_features, dim=0).unsqueeze(1),
                          labels=sample_label.repeat(len(aware_features)))
        return loss

    def forward(self, data):
        
        # out_features, graph_out = self.get_rep(data)
        out_features = self.get_rep(data)

        data_unimodal, aware_features = self.get_aware(data)
        aware_features = torch.cat(aware_features, dim=-1)

        data['unimodal_tensor'] = data_unimodal
        uni_graph_features = self.get_cross(data, out_features)
        unimodal_graph_features = torch.cat(uni_graph_features, dim=-1)

        out = self.clf(torch.cat([out_features, aware_features, unimodal_graph_features], dim=-1),
                       data["text_len_tensor"])

        return out

    def get_loss(self, data):
        
        # out_features, graph_out = self.get_rep(data)
        out_features = self.get_rep(data)

        data_unimodal, aware_features = self.get_aware(data)

        # ---------------------------------------------------------------------

        batch_label = data['label_tensor']
        data['aware_tensor'] = aware_features
        data['unimodal_tensor'] = data_unimodal

        uni_graph_features = self.get_cross(data, out_features)

        loss_center, sample_similarities = self.Multi_Center_Loss(aware_features, batch_label)
        loss_contrastive = self.get_fine_supcontrastive_loss(data)

        # ---------------------------------------------------------------------
       
        aware_features = torch.cat(aware_features, dim=-1)
        unimodal_graph_features = torch.cat(uni_graph_features, dim=-1)

        loss_cls = self.clf.get_loss(
            torch.cat([out_features, aware_features, unimodal_graph_features], dim=-1), data["label_tensor"],
            data["text_len_tensor"]
        )

        loss = loss_cls + loss_center + loss_contrastive

        log.info(f"loss: {loss}, loss_cls: {loss_cls}, loss_center: {loss_center}, loss_contrastive: {loss_contrastive}")

        return loss

def SupConLoss(temperature=1., contrast_mode='all', features=None, labels=None, mask=None, weights=None):

    device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                         'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)  # 1 indicates two items belong to same class
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]  # num of views
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bsz * views, dim)
    if contrast_mode == 'one':
        anchor_feature = features[:, 0]
        anchor_count = 1
    elif contrast_mode == 'all':
        anchor_feature = contrast_feature  # (bsz * views, dim)
        anchor_count = contrast_count  # num of views
    else:
        raise ValueError('Unknown mode: {}'.format(contrast_mode))

    '''compute logits'''
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature)  # (bsz, bsz)
    '''for numerical stability'''
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bsz, 1)
    logits = anchor_dot_contrast - logits_max.detach()  # (bsz, bsz) set max_value in logits to zero

    '''tile mask'''
    mask = mask.repeat(anchor_count, contrast_count)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                                0)  # (anchor_cnt * bsz, contrast_cnt * bsz)
    mask = mask * logits_mask  # 1 indicates two items belong to same class and mask-out itself
    if weights is not None:
        mask = torch.mul(mask, weights)

    '''compute log_prob'''
    exp_logits = torch.exp(logits) * logits_mask  # (anchor_cnt * bsz, contrast_cnt * bsz)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    '''compute mean of log-likelihood over positive'''
    if 0 in mask.sum(1):
        raise ValueError('Make sure there are at least two instances with the same class')
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

    # loss
    # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()
    return loss

class My_anchor_Loss(nn.Module):
    def __init__(self, num_view, num_classes, feature_dim, Margin, size_average=True):
        super(My_anchor_Loss, self).__init__()
        self.num_view = num_view
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.size_average = size_average
        self.M = Margin

        self.anchor = nn.Parameter(torch.randn(self.num_view, self.num_classes, self.feature_dim//2))

    def forward(self, feat, label):

        center_losses = []
        inter_class_losses = []
        sample_similarities = []

        batch_size = feat[0].size(0)
        batch_size_tensor = feat[0].new_empty(1).fill_(
            batch_size if self.size_average else 1)

        for v in range(self.num_view):
            z = feat[v]
            centers_batch = self.anchor[v].index_select(0, label.long())
            sample_similarity = F.cosine_similarity(z, centers_batch)
            ####### intra_loss between center and sample
            center_loss = (z - centers_batch).pow(2).sum() / 2.0 / batch_size_tensor
#----------------------------------            
            # print(center_loss)
#---------------------------------- 
            center_losses.append(center_loss)
            sample_similarities.append(sample_similarity)

        for v in range(self.num_view):
            centers_batch = self.anchor[v]

            # ######## inter_loss of centers
            inter_class_loss = torch.cuda.FloatTensor([0.])
            for i in range(self.num_classes):
                for j in range(i + 1, self.num_classes):
                    inter_class_loss += (centers_batch[i] - centers_batch[j]).pow(2).sum()/self.num_classes/(self.num_classes-1)
#--------------------------------------
            # print(inter_class_loss)
#--------------------------------------
            inter_class_losses.append(max(self.M-inter_class_loss, torch.cuda.FloatTensor([0.])))

        loss = sum(center_losses)/self.num_view + \
               sum(inter_class_losses)/self.num_view

        return loss, sample_similarities

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2 * feature_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )

    def forward(self, x):
        return self.encoder(x)
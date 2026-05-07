import torch
import torch.nn as nn

class MultiLead_MAE_Classifier(nn.Module):

    def __init__(self, pretrained_mae_model, num_leads=12,# num_classes=5, 
                 aggregation='transformer'):
        super().__init__()
        
        # 事前学習済み単誘導MAEモデル（encoder部分）
        self.single_lead_encoder = pretrained_mae_model  # またはモデル全体

        # 埋め込み次元を取得
        self.embed_dim = 128 #self.single_lead_encoder.ChunkEmbed.emb_dim  # 例: 768
        
        self.num_leads = num_leads
        self.aggregation = aggregation
        
        # 統合層の選択
        if aggregation == 'concat':
            self.aggregator = nn.Linear(self.embed_dim * num_leads, self.embed_dim)
        elif aggregation == 'mean':
            self.aggregator = None  # 平均をとるだけ
        elif aggregation == 'attention':
            self.attention = nn.MultiheadAttention(self.embed_dim, num_heads=8)
            self.query = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        elif aggregation == 'transformer':
            self.transformer_layer = nn.TransformerEncoderLayer(
                d_model=self.embed_dim, 
                nhead=8, 
                dim_feedforward=self.embed_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                self.transformer_layer, 
                num_layers=2
            )
        
        # 分類ヘッド
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(self.embed_dim),
        #     nn.Linear(self.embed_dim, num_classes)
        # )
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, num_leads, time_steps]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.shape[0]
        
        # 各誘導を独立に処理
        lead_embeddings = []
        for lead_idx in range(self.num_leads):
            lead_signal = x[:, lead_idx, :]  # [batch_size, time_steps]
            lead_signal = lead_signal.unsqueeze(1)
            
            # MAEのencoderで埋め込みを取得
            # MAEの実装によって異なるが、通常はCLSトークンや平均プーリングで表現を得る
            embedding = self.single_lead_encoder.forward_encoder(lead_signal.unsqueeze(1))  
            # embedding shape: [batch_size, embed_dim]
            
            lead_embeddings.append(embedding)
        
        # 12個の埋め込みをスタック
        lead_embeddings = torch.stack(lead_embeddings, dim=1)  
        # shape: [batch_size, num_leads, embed_dim]
        
        # 統合処理
        if self.aggregation == 'concat':
            # 連結して線形変換
            aggregated = lead_embeddings.reshape(batch_size, -1)
            aggregated = self.aggregator(aggregated)
        elif self.aggregation == 'mean':
            # 単純平均
            aggregated = lead_embeddings.mean(dim=1)
        elif self.aggregation == 'attention':
            # Attention pooling
            query = self.query.expand(batch_size, -1, -1)  
            aggregated, _ = self.attention(
                query, lead_embeddings, lead_embeddings
            )
            aggregated = aggregated.squeeze(1)
        elif self.aggregation == 'transformer':
            # Transformer で誘導間の関係を学習
            aggregated = self.transformer(lead_embeddings)
            aggregated = aggregated.mean(dim=1)  # または[:, 0, :]でCLS的な扱い
        
        # 分類
        # logits = self.classifier(aggregated)
        
        # return logits
        return aggregated
    
    def freeze_encoder(self):
        """事前学習済みエンコーダを凍結"""
        for param in self.single_lead_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """事前学習済みエンコーダを解凍"""
        for param in self.single_lead_encoder.parameters():
            param.requires_grad = True

def prepare_model_12lead_expansion(params, pretrained_model):

    aggr = getattr(params, 'lead_aggregation', 'transformer')
    model = MultiLead_MAE_Classifier(
        pretrained_mae_model=pretrained_model,
        num_leads=12,
        # num_classes=params.num_classes,
        aggregation=aggr
    )
    return model
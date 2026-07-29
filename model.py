import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


class ResBlock3D(nn.Module):
    """
    Pre-activation residual block for 3D volumes.
    BN → ReLU → Conv → BN → ReLU → Conv + skip.
    """
    def __init__(self, channels: int, drop3d: float = 0.1):
        super().__init__()
        self.bn1   = nn.BatchNorm3d(channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.drop  = nn.Dropout3d(p=drop3d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.drop(x)
        x = self.conv2(F.relu(self.bn2(x)))
        return x + residual


def _conv_block(in_ch: int, out_ch: int, drop3d: float) -> nn.Sequential:
    """Conv → BN → ReLU → MaxPool2."""
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(2),
    )


class ImagingCNNClassifier(nn.Module):
    """
    Improved 3D CNN for classification.

    Comprise
      - Channels: 1→32→64→128→256  (was 1→8→16→32)
      - ResBlock3D after every conv (stabilises gradient flow)
      - Dropout3d placed AFTER pooling, at reduced rate 0.1
      - 4th conv block added (256 ch)
      - AdaptiveAvgPool3d(1) → 256-d vector  
      - Hidden dim 256 by default

    Input:  (B, 1, D, H, W)
    Output: (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 3,
        drop3d: float = 0.1,   
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.block1 = _conv_block(1,   32,  drop3d)
        self.res1   = ResBlock3D(32,  drop3d)
        self.drop1  = nn.Dropout3d(p=drop3d)

        self.block2 = _conv_block(32,  64,  drop3d)
        self.res2   = ResBlock3D(64,  drop3d)
        self.drop2  = nn.Dropout3d(p=drop3d)

        self.block3 = _conv_block(64,  128, drop3d)
        self.res3   = ResBlock3D(128, drop3d)
        self.drop3  = nn.Dropout3d(p=drop3d)

        self.block4 = _conv_block(128, 256, drop3d)   # new
        self.res4   = ResBlock3D(256, drop3d)
        self.drop4  = nn.Dropout3d(p=drop3d)

        self.gap = nn.AdaptiveAvgPool3d(1)             # → (B, 256, 1, 1, 1)

        self.fc1        = nn.Linear(256, hidden_dim)
        self.bn_head    = nn.BatchNorm1d(hidden_dim)   # stabilises FC training
        self.dropout    = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.res1(self.block1(x)))
        x = self.drop2(self.res2(self.block2(x)))
        x = self.drop3(self.res3(self.block3(x)))
        x = self.drop4(self.res4(self.block4(x)))

        x = self.gap(x).flatten(1)          # (B, 256)

        x = self.dropout(F.relu(self.bn_head(self.fc1(x))))
        return self.classifier(x)


class Small3DCNN(nn.Module):
    def __init__(self, num_classes=3, drop3d=0.2, drop_fc=0.4):
        super(Small3DCNN, self).__init__()

        # --- Convolutional blocks (halved filter counts) ---
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm3d(8)
        self.drop1 = nn.Dropout3d(p=drop3d)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm3d(16)
        self.drop2 = nn.Dropout3d(p=drop3d)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm3d(32)
        self.drop3 = nn.Dropout3d(p=drop3d)
        self.pool3 = nn.MaxPool3d(2)

        # --- Global average pool → fixed flat size regardless of input dims ---
        # Output: (B, 32, 2, 2, 2) → flat 256
        self.gap = nn.AdaptiveAvgPool3d(2)

        # --- Classifier (smaller bottleneck + dropout) ---
        self.drop_fc = nn.Dropout(p=drop_fc)
        self.fc1 = nn.Linear(32 * 2 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim == 4:          # add channel dim if missing
            x = x.unsqueeze(1)

        x = self.pool1(self.drop1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.drop2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.drop3(F.relu(self.bn3(self.conv3(x)))))

        x = self.gap(x)                   # (B, 32, 2, 2, 2)
        x = x.view(x.size(0), -1)         # (B, 256)

        x = self.drop_fc(F.relu(self.fc1(x)))
        return self.fc2(x)





class TabularClassifier(nn.Module):
    """
    Standalone tabular classifier.

    Architecture:
        TabularBranch  →  Dropout  →  Linear(embed_dim → num_classes)

    Args:
        n_features  : Number of input tabular features.
        num_classes : Number of target classes.
        embed_dim   : Hidden embedding size (TabularBranch output width).
        dropout     : Dropout probability applied in the branch and before the head.
    """

    def __init__(
        self,
        n_features: int,
        num_classes: int,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branch = TabularBranch(
            n_features=n_features,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, n_features) float tensor
        Returns:
            logits : (B, num_classes)
        """
        embeddings = self.branch(x)   # (B, embed_dim)
        logits     = self.head(embeddings)   # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities. Useful at inference time."""
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)


class ImagingBranch(nn.Module):
    """
    3D imaging encoder backed by MONAI's DenseNet121.

    Maps a single-channel 3D MRI volume to a fixed-length embedding
    suitable for late fusion with other modalities. DenseNet121's
    final fully-connected layer is repurposed to output an
    `embed_dim`-length feature vector instead of class logits — this
    module performs representation extraction only, with no
    classification head and no output normalization. Any scale
    alignment needed for fusion (e.g. for cross-attention) is the
    responsibility of the fusion module, not this encoder.

    Parameters
    ----------
    embed_dim : int, default=128
        Dimensionality of the output embedding.
    dropout_prob : float, default=0.2
        Dropout probability applied inside the DenseNet121 backbone
        (before its final linear layer).

    Example
    -------
    >>> branch = ImagingBranch(embed_dim=128)
    >>> vol = torch.randn(4, 1, 96, 96, 96)  # (B, C, D, H, W)
    >>> emb = branch(vol)
    >>> emb.shape
    torch.Size([4, 128])
    """

    def __init__(self, embed_dim: int = 128, dropout_prob: float = 0.2) -> None:
        super().__init__()
        # in_channels=1 because ADNI volumes are single-channel (T1-weighted MRI).
        # out_channels=embed_dim repurposes DenseNet121's classification head
        # as a plain projection to the embedding space.
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=embed_dim,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of 3D MRI volumes into embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)
            Batch of normalized 3D MRI volumes.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Unnormalized imaging embeddings.
        """
        return self.backbone(x)


class TabularBranch(nn.Module):
    """
    MLP encoder for tabular (clinical/demographic) features.

    Maps a vector of preprocessed tabular features to a fixed-length
    embedding suitable for late fusion with other modalities. Like
    `ImagingBranch`, this module performs representation extraction
    only — no classification head, and no output normalization.

    Parameters
    ----------
    n_features : int
        Number of input tabular features.
    embed_dim : int, default=64
        Dimensionality of the output embedding.
    dropout : float, default=0.3
        Dropout probability applied after the hidden layer's
        activation.

    Example
    -------
    >>> branch = TabularBranch(n_features=20, embed_dim=64)
    >>> x = torch.randn(4, 20)
    >>> emb = branch(x)
    >>> emb.shape
    torch.Size([4, 64])
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),  # stabilizes the hidden layer; batch size here
                                   # is large enough (128-wide activations feeding
                                   # from a full batch) to be well-behaved, unlike
                                   # normalizing the small final embedding.
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),  # final projection to embedding space;
                                         # deliberately no activation or norm here
                                         # so the fusion stage receives an
                                         # unconstrained representation.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tabular feature vectors into embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_features)
            Batch of preprocessed (imputed, encoded, scaled) tabular
            features.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Unnormalized tabular embeddings.
        """
        return self.net(x)


import torch 
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121


class ImagingBranch(nn.Module):
    """
    3D imaging encoder backed by MONAI's DenseNet121.

    Maps a single-channel 3D MRI volume to a fixed-length embedding
    suitable for late fusion with other modalities. DenseNet121's
    final fully-connected layer is repurposed to output an
    `embed_dim`-length feature vector instead of class logits — this
    module performs representation extraction only, with no
    classification head and no output normalization. Any scale
    alignment needed for fusion (e.g. for cross-attention) is the
    responsibility of the fusion module, not this encoder.

    Parameters
    ----------
    embed_dim : int, default=128
        Dimensionality of the output embedding.
    dropout_prob : float, default=0.2
        Dropout probability applied inside the DenseNet121 backbone
        (before its final linear layer).

    Example
    -------
    >>> branch = ImagingBranch(embed_dim=128)
    >>> vol = torch.randn(4, 1, 96, 96, 96)  # (B, C, D, H, W)
    >>> emb = branch(vol)
    >>> emb.shape
    torch.Size([4, 128])
    """

    def __init__(self, embed_dim: int = 128, dropout_prob: float = 0.2) -> None:
        super().__init__()
        # in_channels=1 because ADNI volumes are single-channel (T1-weighted MRI).
        # out_channels=embed_dim repurposes DenseNet121's classification head
        # as a plain projection to the embedding space.
        self.backbone = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=embed_dim,
            dropout_prob=dropout_prob,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of 3D MRI volumes into embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 1, D, H, W)
            Batch of normalized 3D MRI volumes.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Unnormalized imaging embeddings.
        """
        return self.backbone(x)


class TabularBranch(nn.Module):
    """
    MLP encoder for tabular (clinical/demographic) features.

    Maps a vector of preprocessed tabular features to a fixed-length
    embedding suitable for late fusion with other modalities. Like
    `ImagingBranch`, this module performs representation extraction
    only — no classification head, and no output normalization.

    Parameters
    ----------
    n_features : int
        Number of input tabular features.
    embed_dim : int, default=64
        Dimensionality of the output embedding.
    dropout : float, default=0.3
        Dropout probability applied after the hidden layer's
        activation.

    Example
    -------
    >>> branch = TabularBranch(n_features=20, embed_dim=64)
    >>> x = torch.randn(4, 20)
    >>> emb = branch(x)
    >>> emb.shape
    torch.Size([4, 64])
    """

    def __init__(
        self,
        n_features: int,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),  # stabilizes the hidden layer; batch size here
                                   # is large enough (128-wide activations feeding
                                   # from a full batch) to be well-behaved, unlike
                                   # normalizing the small final embedding.
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),  # final projection to embedding space;
                                         # deliberately no activation or norm here
                                         # so the fusion stage receives an
                                         # unconstrained representation.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of tabular feature vectors into embeddings.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_features)
            Batch of preprocessed (imputed, encoded, scaled) tabular
            features.

        Returns
        -------
        torch.Tensor, shape (B, embed_dim)
            Unnormalized tabular embeddings.
        """
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where the imaging embedding attends over
    distributed tabular latent tokens.

    Inputs
    -------
    img_emb:
        (B, img_embed_dim)

    tab_emb:
        (B, tab_embed_dim)


    Processing
    ----------
    MRI:
        (B, img_embed_dim)
        ->
        (B, 1, d_model)   # Query


    Tabular:
        (B, tab_embed_dim)
        ->
        (B, n_tab_tokens, token_dim)
        ->
        (B, n_tab_tokens, d_model)  # Key / Value


    Output
    ------
    fused:
        (B, img_embed_dim + tab_embed_dim)

    Example:
        img_emb = (B,128)
        tab_emb = (B,64)

        output = (B,192)
    """

    def __init__(
        self,
        img_embed_dim: int,
        tab_embed_dim: int,
        d_model: int = 64,
        n_tab_tokens: int = 8,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % n_heads == 0, \
            "d_model must be divisible by n_heads"

        assert tab_embed_dim % n_tab_tokens == 0, \
            "tab_embed_dim must be divisible by n_tab_tokens"


        # -------------------------
        # MRI projection
        # -------------------------

        self.img_proj = nn.Linear(
            img_embed_dim,
            d_model
        )


        # -------------------------
        # Tabular latent tokenization
        # -------------------------

        self.n_tab_tokens = n_tab_tokens

        tab_token_dim = tab_embed_dim // n_tab_tokens

        self.tab_token_dim = tab_token_dim


        # Each latent tabular token is projected
        # into the attention embedding space

        self.tab_proj = nn.Linear(
            tab_token_dim,
            d_model
        )


        # -------------------------
        # Cross attention
        # -------------------------

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )


        # -------------------------
        # Normalization
        # -------------------------

        self.norm = nn.LayerNorm(
            d_model
        )


        # -------------------------
        # Restore concat dimension
        # -------------------------

        self.out_proj = nn.Linear(
            d_model,
            img_embed_dim + tab_embed_dim
        )


        self.out_dim = img_embed_dim + tab_embed_dim



    def forward(
        self,
        img_emb: torch.Tensor,
        tab_emb: torch.Tensor
    ):

        # ==================================================
        # MRI QUERY
        # ==================================================

        # (B,img_embed_dim)
        q = self.img_proj(img_emb)

        # (B,d_model)
        q = q.unsqueeze(1)

        # (B,1,d_model)



        # ==================================================
        # TABULAR KEY / VALUE TOKENS
        # ==================================================

        B = tab_emb.size(0)

        # Split tabular latent representation into tokens
        #
        # (B,64)
        #
        # ->
        #
        # (B,8,8)

        tab_tokens = tab_emb.reshape(
            B,
            self.n_tab_tokens,
            self.tab_token_dim
        )


        # Project tokens into attention space
        #
        # (B,8,8)
        #
        # ->
        #
        # (B,8,64)

        tab_tokens = self.tab_proj(
            tab_tokens
        )



        # ==================================================
        # CROSS ATTENTION
        # ==================================================

        # Query:
        #   (B,1,64)
        #
        # Key:
        #   (B,8,64)
        #
        # Value:
        #   (B,8,64)

        attn_out, attn_weights = self.attn(
            query=q,
            key=tab_tokens,
            value=tab_tokens
        )


        # attn_out:
        # (B,1,64)



        # ==================================================
        # RESIDUAL + NORMALIZATION
        # ==================================================

        out = self.norm(
            q + attn_out
        )

        # (B,1,64)


        out = out.squeeze(1)

        # (B,64)



        # ==================================================
        # PROJECT TO CONCAT DIMENSION
        # ==================================================

        out = self.out_proj(out)

        # (B,192)

        return out, attn_weights



class MultimodalADNI(nn.Module):
    """
    Multimodal ADNI classifier with selectable late-fusion strategy.

    Combines a 3D imaging embedding (`ImagingBranch`) and a tabular
    embedding (`TabularBranch`) via one of two fusion strategies, then
    classifies the fused representation into diagnostic groups
    (e.g. CN / LMCI / AD).

    Parameters
    ----------
    n_tabular_features : int
        Number of input tabular features (passed to `TabularBranch`).
    num_classes : int, default=3
        Number of diagnostic classes to predict.
    img_embed_dim : int, default=128
        Output embedding size of `ImagingBranch`.
    tab_embed_dim : int, default=64
        Output embedding size of `TabularBranch`.
    fusion : {"concat", "cross_attn"}, default="concat"
        Late-fusion strategy:
          - "concat"     — concatenate img and tab embeddings, then MLP.
          - "cross_attn" — project both to `attn_d_model`, run
                            cross-attention (img attends over tab),
                            then MLP.
    fusion_dropout : float, default=0.4
        Dropout probability applied inside the shared fusion MLP.
    attn_d_model : int, default=64
        Shared projection dimensionality for cross-attention.
        Ignored when `fusion="concat"`.
    attn_n_heads : int, default=4
        Number of attention heads for cross-attention.
        Ignored when `fusion="concat"`.
    attn_dropout : float, default=0.1
        Dropout probability inside the cross-attention module.
        Ignored when `fusion="concat"`.

    Example
    -------
    >>> model_cat  = MultimodalADNI(n_tabular_features=20, fusion="concat")
    >>> model_attn = MultimodalADNI(n_tabular_features=20, fusion="cross_attn")
    """

    #: Supported fusion strategy identifiers.
    FUSION_STRATEGIES: set[str] = {"concat", "cross_attn"}

    def __init__(
        self,
        n_tabular_features: int,
        num_classes: int = 3,
        img_embed_dim: int = 128,
        tab_embed_dim: int = 64,
        fusion: str = "concat",
        fusion_dropout: float = 0.4,
        # cross_attn-specific hyper-params (ignored when fusion="concat")
        attn_d_model: int = 64,
        attn_n_heads: int = 4,
        attn_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion not in self.FUSION_STRATEGIES:
            raise ValueError(
                f"Unknown fusion strategy '{fusion}'. "
                f"Choose from {self.FUSION_STRATEGIES}."
            )
        self.fusion_type: str = fusion

        # ── Modality branches (representation extraction only) ─────────────
        self.imaging = ImagingBranch(embed_dim=img_embed_dim)
        self.tabular = TabularBranch(
            n_features=n_tabular_features, embed_dim=tab_embed_dim
        )
        self.attn_weights = None
        # ── Fusion layer ─────────────────────────────────────────────────
        if fusion == "concat":
            # Simple concatenation: fusion_mlp's first Linear layer is
            # responsible for learning any necessary scale/weighting
            # between the two modalities.
            fusion_in = img_embed_dim + tab_embed_dim
        else:  # "cross_attn"
            # Cross-attention: image embedding attends over the tabular
            # embedding, producing a d_model-sized fused representation.
            # Any scale alignment attention needs (e.g. scaled dot-product,
            # LayerNorm on Q/K) is handled inside CrossAttentionFusion.
            self.cross_attn = CrossAttentionFusion(
                                img_embed_dim=img_embed_dim,
                                tab_embed_dim=tab_embed_dim,
                                d_model=attn_d_model,
                                n_tab_tokens=16,
                                dropout=attn_dropout,
                                n_heads=attn_n_heads
                            )
            
            fusion_in = img_embed_dim + tab_embed_dim

        # ── Shared classification head applied after fusion ────────────────
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, img: torch.Tensor, tab: torch.Tensor) -> torch.Tensor:
        """
        Run the full multimodal forward pass: encode both modalities,
        fuse them, and classify.

        Parameters
        ----------
        img : torch.Tensor, shape (B, 1, D, H, W)
            Batch of 3D MRI volumes.
        tab : torch.Tensor, shape (B, n_tabular_features)
            Batch of preprocessed tabular feature vectors.

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
            Unnormalized class logits (apply softmax/argmax externally).
        """
        img_emb = self.imaging(img)    # (B, img_embed_dim)
        tab_emb = self.tabular(tab)    # (B, tab_embed_dim)

        if self.fusion_type == "concat":
            fused = torch.cat([img_emb, tab_emb], dim=1)  # (B, img_embed_dim + tab_embed_dim)
        else:
            fused, self.attn_weights  = self.cross_attn(img_emb, tab_emb)      # (B, attn_d_model)

        return self.fusion_mlp(fused)  # (B, num_classes)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn


class UADFLiteGate(nn.Module):
    """
    UADF-lite: encoder feature level에서 audio / video를 부드럽게 섞는 게이트 블록

    - audio_features: [B, Ta, H]
    - video_features: [B, Tv, H]
    - return: fused_features: [B, T_min, H]  (T_min = min(Ta, Tv))

    특징:
    - audio를 base로 두고, video는 학습 가능한 비율로만 살짝 더함
    - 길이 안 맞으면 더 짧은 길이에 맞춰 'trim' (interpolate X)
    """

    def __init__(
        self,
        hidden_size: int,
        max_video_ratio: float = 0.3,
        per_channel: bool = False,
    ):
        """
        Args:
            hidden_size: feature dimension H
            max_video_ratio: video 비율의 상한 (예: 0.3이면 audio + 0~0.3 * video)
            per_channel: True면 채널별 gate, False면 스칼라 gate 하나
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.max_video_ratio = max_video_ratio
        self.per_channel = per_channel

        # gate 초기값 0 → sigmoid(0)=0.5 → 0.5 * max_video_ratio 정도에서 시작
        if per_channel:
            self.gate = nn.Parameter(torch.zeros(hidden_size))   # [H]
        else:
            self.gate = nn.Parameter(torch.zeros(1))             # scalar

        self.norm = nn.LayerNorm(hidden_size)

    def _align_time(self, audio_features, video_features):
        """
        단순 시간축 정렬: 더 짧은 길이에 맞게 둘 다 잘라버림
        (너무 공격적인 interpolate 대신, 안전한 trim)
        """
        Ta = audio_features.size(1)
        Tv = video_features.size(1)
        T = min(Ta, Tv)

        if Ta != T:
            audio_features = audio_features[:, :T, :]
        if Tv != T:
            video_features = video_features[:, :T, :]

        return audio_features, video_features

    def forward(self, audio_features, video_features):
        """
        audio_features: [B, Ta, H]
        video_features: [B, Tv, H]
        """
        # dtype 맞추기
        if audio_features.dtype != video_features.dtype:
            video_features = video_features.to(dtype=audio_features.dtype)

        # 길이 정렬
        audio_features, video_features = self._align_time(audio_features, video_features)

        # gate → lambda (0 ~ max_video_ratio)
        gate = torch.sigmoid(self.gate)  # [1] or [H]
        if self.per_channel:
            # [H] → [1, 1, H] → [B, T, H] 에 자동 브로드캐스트
            lam = gate.view(1, 1, -1) * self.max_video_ratio
        else:
            # scalar → [1, 1, 1]
            lam = gate.view(1, 1, 1) * self.max_video_ratio

        # audio base + video 보정
        fused = audio_features + lam * video_features
        fused = self.norm(fused)

        return fused


def create_uadf_block(hidden_size, max_video_ratio: float = 0.3, per_channel: bool = False):
    """
    AVSR_LLMs에서 기존 UADFBlock 생성하던 자리에 그대로 쓸 수 있는 helper
    """
    return UADFLiteGate(
        hidden_size=hidden_size,
        max_video_ratio=0.3,  # 비디오 최대 30%까지만 반영
        per_channel=False
    )

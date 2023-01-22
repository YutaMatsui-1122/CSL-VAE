#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright © 2022 Takafumi Horie All rights reserved.

""" heatmap

seaborn を利用して heatmap を描画するためのモジュールです.

"""
from typing import List, Set, Union

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def make_sorted_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """列の値を行を基準に並べ替え, なるべくヒートマップが対角になるよう整えたDataframeを返す

    Args:
        row_data(Dict[Any, List[Any]])

    Returns:
        pd.dataframe: 整えられたDataframe

    """

    # 列 ごとに最大の行を求め, そのインデックス順に並び替える
    df_index = list(dataframe.index)
    maxcat_id = [
        df_index.index(category_name) for category_name in list(dataframe.idxmax())
    ]

    df_columns = np.array(dataframe.columns)
    dataframe = dataframe.reindex(columns=df_columns[np.argsort(maxcat_id)])

    return dataframe


def generate_cmap(colors: List[str]) -> LinearSegmentedColormap:
    """自分で定義したカラーマップを返す

    colors で指定した色を, 前から順に線形補間していき, カラーマップを作ります.

    Args:
        colors(List[str]): 補完する色リスト.

    Returns:
        LinearSegmentedColormap: 作成したカラーマップ

    Examples:

        generate_cmap(["white","red"])          # 白 - 赤の線形補間
        generate_cmap(["white","red", "black"]) # 白 - 赤 - 黒の線形補間

    """
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for val, col in zip(values, colors):
        color_list.append((val / vmax, col))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


def diagonallike_heatmap(
    filepath:str,
    dataframe:pd.DataFrame,
    chained_frame:pd.DataFrame=None,
    each_attribute_cats:List[Set[Union[str, int, float]]]=None,
    square=False,
)->pd.DataFrame:
    """ 1-Agent 分のヒートマップを描画する

    Dataframe をなるべく対角になるよう整列して, ヒートマップを描画する.
    具体的には, いかの処理を行う.
        (i)  chain_frame が固定されていたならば, 
             縦軸を固定し, 対角になるよう横軸を整列した後,
             横軸を固定して対角になるよう縦軸を整列
        (ii) chain_frame が指定されていて each_attribute_cats is None なら,
             横軸を chain_frame と揃え, 対角になるよう縦軸を整列
        (iii)chain_frame が指定されていて each_attribute_cats is not None なら,
             横軸を chain_frame と揃え, !! 属性ごとに !! 対角になるよう縦軸を整列


    Args:
        filepath            (str)                 : ファイルへのフルパス（もちろん相対でも絶対でもよい）
        dataframe           (pd.Dataframe)        : ヒートマップを描画する対象の dataframe
        chained_frame       (pd.Dataframe)        : 横軸の基準とする dataframe
        each_attribute_cats (Dict[str, List[str]]): 属性ごとの, 「使われているカテゴリの集合」のリスト

    Returns:
        pd.Dataframe: ヒートマップを描くために各軸を整列した後の Dataframe
    """
    ## Step1. 整列処理
    if chained_frame is None: ##(i).
        dataframe = make_sorted_dataframe(dataframe)
        chained_frame = dataframe

    if each_attribute_cats is None: ##(ii)
        dataframe = make_sorted_dataframe(
            dataframe.reindex(columns=chained_frame.columns).T
        ).T
    else: ##(iii)
        dataframe = dataframe.reindex(columns=chained_frame.columns)
        end_pos = 0
        for df_id, var_z in enumerate(each_attribute_cats):
            start_pos = end_pos
            end_pos += len(var_z)
            partial_df = make_sorted_dataframe(dataframe[start_pos:end_pos].T).T

            if df_id == 0:
                ans_df = partial_df
                continue
            ans_df = pd.concat((ans_df, partial_df))

        dataframe = ans_df

    ## step2. 描画の準備
    cmap = generate_cmap(["white", "red"])
    fig, axes = plt.subplots()
    axes.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True, labelsize=4)

    ## step3. 描画する
    sns.heatmap(
        dataframe, square=square, cmap=cmap, vmax=1, vmin=0
    )

    if each_attribute_cats is None:
        fig.savefig(filepath, bbox_inches="tight")
        plt.close()
        return dataframe

    ## var_z が指定されていれば, 属性の区切りで線を引く
    line_pos = 0
    for var_z in each_attribute_cats:
        line_pos += len(var_z)
        axes.axhline(y=line_pos, linewidth=1, color="black")

    fig.savefig(filepath, bbox_inches="tight")
    plt.close()
    return dataframe

"""Render the coordinate convention and its discrepancy on source vehicle annotations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


def plot(report, filename):
    for font in (Path("/System/Library/Fonts/STHeiti Light.ttc"),
                 Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")):
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
            break
    plt.rcParams.update({"font.size":11, "axes.unicode_minus":False,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig,(axis,error) = plt.subplots(1,2,figsize=(10.4,4.9),gridspec_kw={"width_ratios":[1,1.45]})
    fig.subplots_adjust(left=.095,right=.965,wspace=.41,bottom=.25,top=.73)
    fig.text(.065,.905,"PINNS 鸟瞰图与轨迹纵轴的配准",fontsize=21,color="#163e41")
    fig.text(.065,.833,"47 段录像  ·  47,555 条车辆原始记录  ·  同一行鸟瞰中心与地面中心直接配对",color="#4b5d61")
    h=744
    v=np.arange(h)
    axis.plot(v/h,v/h,color="#087672",lw=2.4,label="发布轨迹标签")
    axis.plot(v/h,1-v/(h-1),ls="--",color="#a04350",lw=2.2,label="旧图像轮廓")
    axis.set(xlabel="鸟瞰像素纵坐标 / 图像高度",ylabel="换算后的相对地面纵坐标",
             xlim=(-.02,1.02),ylim=(-.035,1.035),xticks=[0,.5,1],yticks=[0,.5,1])
    axis.set_title("同一图像行的两种换算",loc="left",fontsize=13,pad=10)
    axis.legend(loc="upper center",bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=9)
    sites=[("america_crossroad","美国"),("italy_park","意大利"),
           ("japan_crossroad","日本"),("thailand_street","泰国")]
    labels=[]
    for i,(key,name) in enumerate(sites):
        item=report["per_site"][key]["metrics"]["old_contour_formula_minus_one_and_y_opposite_vs_parsed_trajectory"]
        y=3-i
        middle,tail=item["median_m"],item["p95_m"]
        error.plot([middle,tail],[y,y],color="#bb7a80",lw=1.9)
        error.plot(middle,y,"o",color="#9a3948",markersize=7)
        error.plot(tail,y,"|",color="#9a3948",markersize=12,markeredgewidth=1.9)
        error.annotate(f"{middle:.2f}",(middle,y),xytext=(0,7),textcoords="offset points",ha="center",fontsize=9)
        labels.append(f"{name}（{item['n']:,}）")
    error.set(yticks=[3,2,1,0],yticklabels=labels,xlim=(0,26),ylim=(-.6,3.65),xlabel="与发布地面位置的差（米）")
    error.set_title("将旧换算应用于车辆像素中心",loc="left",fontsize=13,pad=10)
    error.set_axisbelow(True);error.grid(axis="x",alpha=.22)
    fig.text(.54,.07,"圆点为中位数，短竖线为第 95 百分位。",fontsize=9,color="#455d62")
    fig.savefig(filename,dpi=165,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("aggregate",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    plot(json.loads(args.aggregate.read_bytes()),args.output)

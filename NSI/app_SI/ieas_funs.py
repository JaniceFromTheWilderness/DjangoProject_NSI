import json
import geopandas as gpd
import plotly.express as px
from numpy.linalg import inv
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output,State, ctx, callback,dash_table

from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import os
os.environ["OMP_NUM_THREADS"] = '1' #before import kmeans
from sklearn.cluster import KMeans
from pypinyin import pinyin, Style #中文排序用
from dateutil.relativedelta import relativedelta as rd
from datetime import date
from datetime import datetime
#from lib.utils import create_graph
class ieas_project:
    def __init__(self,excel_file, path, tw_datam,tw_dataq, tw_datamt, tw_dataqt,temporary_path):
        self.tw_datamtT = tw_datamt.T
        self.tw_datamtT.columns = tw_datam.columns

        self.tw_dataqtT = tw_dataqt.T
        self.tw_dataqtT.columns = tw_dataq.columns
        self.excel_file = excel_file
        self.path = path
        self.tw_datam = tw_datam
        self.tw_dataq = tw_dataq
        self.tw_datamt = tw_datamt
        self.tw_dataqt = tw_dataqt


        self.tw_datam_col = self.tw_datam.columns.tolist()
        self.tw_dataq_colmix = self.tw_dataq.columns.tolist()
        self.color_b1 = '#1e90ff' #dodgerblue
        self.color_o1 = '#ffa500' #orange
        self.color_r1 = '#fa8072' #salmon
        self.color_b2 = '#00008b'

        self.temporary_path = temporary_path
    """
    ＃此處在月資料會失敗，直接網頁上django_plotly_dash不存在，也不知道原因（但季資料就不會有問題）
    def __init__(self, path, excel_file,temporary_path):
        self.path = path
        tw_datam = pd.read_excel(
            excel_file,
            sheet_name='彙整月', header=[0], index_col=[0], na_values=['-', '－'], parse_dates=[0])
        tw_dataq = pd.read_excel(
            excel_file,
            sheet_name='彙整季', header=[0, 1], index_col=[0], na_values='-', parse_dates=[0])
        tw_datamt = pd.read_excel(
            excel_file,
            sheet_name='月資料說明', header=[0], index_col=[0], na_values=['-', '－'], parse_dates=[0])
        tw_dataqt = pd.read_excel(
            excel_file,
            sheet_name='季資料說明', header=[0], index_col=[0], na_values='-', parse_dates=[0])
        tw_dataq.index.name = ('Unnamed: 0_level_0', 'Indicator')
        tw_dataq_colori = tw_dataq.columns.tolist()
        tw_dataq_colmix = ['{}_{}'.format(col[0], col[1]) for col in tw_dataq_colori]  # 將雙層columns name降為一層
        tw_dataq.columns = tw_dataq_colmix
        # tw_dataq.loc[:, '貢獻率_國外淨需求-減：商品及服務輸入'] = tw_dataq.loc[:, '貢獻率_國外淨需求-減：商品及服務輸入'] * -1#貢獻率_貢獻度國外淨需求-減：商品及服務輸入
        tw_dataq.loc[:, '貢獻率_貢獻度國外淨需求-減：商品及服務輸入'] = tw_dataq.loc[:,
                                                                       '貢獻率_貢獻度國外淨需求-減：商品及服務輸入'] * -1  # 貢獻率_貢獻度國外淨需求-減：商品及服務輸入

        self.tw_datam = tw_datam
        self.tw_dataq = tw_dataq
        self.tw_datamt = tw_datamt
        self.tw_dataqt = tw_dataqt
        self.tw_datam_col = self.tw_datam.columns.tolist()
        self.tw_dataq_colmix = self.tw_dataq.columns.tolist()
        self.color_b1 = '#1e90ff' #dodgerblue
        self.color_o1 = '#ffa500' #orange
        self.color_r1 = '#fa8072' #salmon
        self.color_b2 = '#00008b'
        self.temporary_path = temporary_path
    """

    def bar_Chart(self,x, y, name, yaxis, marker_line_color, textfontcolor):
        chart = go.Bar(
            x=x,
            y=y,
            name=name,
            yaxis=yaxis,
            text=y,
            textposition='outside',
            marker=dict(
                color=marker_line_color
            ),
            textfont=dict(size=13,
                          color=textfontcolor),
            marker_line_color=marker_line_color,
            marker_line_width=1,
            opacity=0.8
        )
        return chart

    def Scatter_Chart(self,x, y, name, yaxis, markercolor, linecolor):
        chart = go.Scatter(
            x=x,
            y=y,
            name=name,
            yaxis=yaxis,
            mode='lines',
            # marker= dict(size=3,
            #            symbol = 'diamond',
            #            color =markercolor,
            #            line_width = 2),
            line=dict(color=linecolor, width=2)
        )
        return chart

    def produce_yaxis(self,title, txt_y, color):  # 藍 "#00008b" 紅 "#dc143c" 橘 "#d2691e"
        if txt_y == 1:
            yaxis = dict(
                title=title,
                titlefont=dict(
                    color=color
                ),
                tickfont=dict(
                    color=color
                ),
                side="left",
                position=0.05
            )
        elif txt_y == 2:
            yaxis = dict(
                title=title,
                titlefont=dict(
                    color=color
                ),
                tickfont=dict(
                    color=color
                ),
                anchor="x",
                overlaying="y",
                side="right",
            )
        else:
            yaxis = dict(
                title=title,
                titlefont=dict(
                    color=color
                ),
                tickfont=dict(
                    color=color
                ),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.94
            )
        return yaxis

    def fig1(self,Title,x1, y1, name1, chart_type1, height=750, width=1450):

        fig = go.Figure()
        fig.data = []
        if chart_type1 == 'b':
            fig.add_trace(self.bar_Chart(x1, y1, name1, 'y1', marker_line_color=self.color_b1, textfontcolor=self.color_b1))  # 藍 #1f77b4
        else:
            fig.add_trace(self.Scatter_Chart(x1, y1, name1, 'y1', markercolor=self.color_b1, linecolor=self.color_b1))  # 藍 royalblue
        fig.layout = {} #因為會不斷畫圖，若不在重置布局會造成範圍滑塊錯亂

        fig.update_layout(
            xaxis=dict(
                domain=[0.05, 0.85]
            ),

            yaxis1=self.produce_yaxis('', 1,self.color_b1)  # 藍 "#00008b" 紅 "#dc143c" 橘 "#d2691e"
        )
        fig.update_layout(
            title={
                'text': '<b>' + name1 + '<b>',
                'y': 0.94,  # 位置
                'x': 0.2,
                'xanchor': 'center',  # 相對位置
                'yanchor': 'top'
            },
            bargap=0.15,  # 相鄰座標條之間距
            bargroupgap=0.08,  # 僅在共享座標軸時起作用 # 相同位置坐標的條之間的間隙
            xaxis_title="",  # 橫軸軸標籤
            # yaxis_title=target_title,  # 縱軸軸標籤
            yaxis_title_standoff=10,  # 調整 y 軸刻度標籤與圖表之間的距離
            yaxis_showtickprefix='first',
            font=dict(size=14),  # 設置字體大小
            # template='plotly_white',  # 換默認模板
            autosize=True,
            height=height,  # 設置繪圖的高度（以像素為單位）
            width=width,
            hovermode="x",  # 以橫軸顯示每個點的細節

        )

        # Add range slider
        #(a, b, c) = (pd.Period(x1[1], freq='Q').start_time, pd.Period(x1[0], freq='Q').start_time,pd.Period(x1[-1], freq='Q').start_time)
        #months_difference = rd(a, b).months
        #if months_difference == 3:
        if Title=='季資料':
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=2,
                                 label="2y",
                                 step="year",
                                 stepmode="backward"),

                            dict(count=3,
                                 label="3y",
                                 step="year",
                                 stepmode="backward"),
                            dict(count=4,  # 設置範圍要採取的步數
                                 label="4y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=5,  # 設置範圍要採取的步數
                                 label="5y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=10,  # 設置範圍要採取的步數
                                 label="10y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=15,  # 設置範圍要採取的步數
                                 label="15y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=20,  # 設置範圍要採取的步數
                                 label="20y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=6,
                                 label="6m",
                                 step="month",
                                 stepmode="backward"),

                            dict(count=12,
                                 label="1y",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=18,  # 設置範圍要採取的步數
                                 label="1.5y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=24,  # 設置範圍要採取的步數
                                 label="2y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=36,  # 設置範圍要採取的步數
                                 label="3y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )

        data = pd.concat([y1], axis=1)

        return (fig, data)

    def fig2(self,Title,x1, y1, name1, chart_type1, x2, y2, name2, chart_type2, shared_yaxis, title, height=750, width=1200):
        fig = go.Figure()
        if chart_type1 == 'b':
            fig.add_trace(self.bar_Chart(x1, y1, name1, 'y1', marker_line_color=self.color_b1, textfontcolor=self.color_b1))  # 藍
        else:
            fig.add_trace(self.Scatter_Chart(x1, y1, name1, 'y1', markercolor=self.color_b1, linecolor=self.color_b1))  # 藍

        yaxis2 = 'y1' if shared_yaxis == 'y' else 'y2'

        if chart_type2 == 'b':
            fig.add_trace(self.bar_Chart(x2, y2, name2, yaxis2, marker_line_color=self.color_r1, textfontcolor=self.color_r1))  # 紅'#cd5c5c' '#b22222'
        else:
            fig.add_trace(self.Scatter_Chart(x2, y2, name2, yaxis2, markercolor=self.color_r1, linecolor=self.color_r1))  # 紅 "firebrick"

        (yaxis1title,color) = ('',self.color_b2) if shared_yaxis == 'y' else (name1,self.color_b1)

        fig.update_layout(
            xaxis=dict(
                domain=[0.05, 0.85]
            ),

            yaxis1=self.produce_yaxis(yaxis1title, 1, color),  # 藍 "#00008b" 紅 "#dc143c" 橘 "#d2691e"
            yaxis2=self.produce_yaxis(name2, 2, self.color_r1),

        )

        fig.update_layout(
            title={
                'text': '<b>' + title + '<b>',
                'y': 0.94,  # 位置
                'x': 0.2,
                'xanchor': 'center',  # 相對位置
                'yanchor': 'top'
            },
            bargap=0.15,  # 相鄰座標條之間距
            bargroupgap=0.08,  # 僅在共享座標軸時起作用 # 相同位置坐標的條之間的間隙
            xaxis_title="",  # 橫軸軸標籤
            # yaxis_title=target_title,  # 縱軸軸標籤
            yaxis_title_standoff=10,  # 調整 y 軸刻度標籤與圖表之間的距離
            yaxis_showtickprefix='first',
            font=dict(size=14),  # 設置字體大小
            # template='plotly_white',  # 換默認模板
            autosize=True,
            height=height,  # 設置繪圖的高度（以像素為單位）
            width=width,
            hovermode = "x",  #以橫軸顯示每個點的細節

        )
        # Add range slider
        #(a, b, c) = (pd.Period(x1[1], freq='Q').start_time, pd.Period(x1[0], freq='Q').start_time,pd.Period(x1[-1], freq='Q').start_time)
        #months_difference = rd(a, b).months
        #if months_difference == 3:
        if Title=='季資料':
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=2,
                                 label="2y",
                                 step="year",
                                 stepmode="backward"),

                            dict(count=3,
                                 label="3y",
                                 step="year",
                                 stepmode="backward"),
                            dict(count=4,  # 設置範圍要採取的步數
                                 label="4y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=5,  # 設置範圍要採取的步數
                                 label="5y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=10,  # 設置範圍要採取的步數
                                 label="10y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=15,  # 設置範圍要採取的步數
                                 label="15y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=20,  # 設置範圍要採取的步數
                                 label="20y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=6,
                                 label="6m",
                                 step="month",
                                 stepmode="backward"),

                            dict(count=12,
                                 label="1y",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=18,  # 設置範圍要採取的步數
                                 label="1.5y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=24,  # 設置範圍要採取的步數
                                 label="2y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=36,  # 設置範圍要採取的步數
                                 label="3y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )

        data = pd.concat([y1, y2], axis=1)

        return (fig, data)

    def fig3(self,Title,x1, y1, name1, chart_type1, x2, y2, name2, chart_type2, shared_yaxis, x3, y3, name3, chart_type3,
             shared_yaxis3, title, height=750, width=1300):
        fig = go.Figure()
        fig.data = []

        if chart_type1 == 'b':
            fig.add_trace(self.bar_Chart(x1, y1, name1, 'y1', marker_line_color=self.color_b1, textfontcolor=self.color_b1))  # 藍
        else:
            fig.add_trace(self.Scatter_Chart(x1, y1, name1, 'y1', markercolor=self.color_b1, linecolor=self.color_b1))  # 藍

        yaxis2 = 'y1' if shared_yaxis == 'y' else 'y2'

        if chart_type2 == 'b':
            fig.add_trace(self.bar_Chart(x2, y2, name2, yaxis2, marker_line_color=self.color_r1, textfontcolor=self.color_r1))  # 紅
        else:
            fig.add_trace(self.Scatter_Chart(x2, y2, name2, yaxis2, markercolor=self.color_r1, linecolor=self.color_r1))  # 紅

        if shared_yaxis3 == 'y':
            yaxis3 = 'y1'
        elif shared_yaxis == 'y':
            yaxis3 = 'y2'
        else:
            yaxis3 = 'y3'

        if chart_type3 == 'b':
            fig.add_trace(self.bar_Chart(x3, y3, name3, yaxis3, marker_line_color=self.color_o1, textfontcolor=self.color_o1))  # 橘色 'coral' "#ff7f50"
        else:
            fig.add_trace(self.Scatter_Chart(x3, y3, name3, yaxis3, markercolor=self.color_o1, linecolor=self.color_o1))  # 橘

        (yaxis1title,color1) = ('',self.color_b2) if (shared_yaxis == 'y' or shared_yaxis3 == 'y') else (name1,self.color_b1)

        (yaxis2title, color2) = (name3, self.color_o1) if shared_yaxis == 'y' else (name2, self.color_r1)
        fig.layout = {} #因為會不斷畫圖，若不在重置布局會造成範圍滑塊錯亂

        fig.update_layout(
            xaxis=dict(
                domain=[0.05, 0.85]
            ),
            # yaxis=dict(
            #    domain=[0, 1]
            # ),
            yaxis1=self.produce_yaxis(yaxis1title, 1, color1),  # 藍 "#00008b" 紅 "#dc143c" 橘 "#d2691e"
            yaxis2=self.produce_yaxis(yaxis2title, 2, color2),
            yaxis3=self.produce_yaxis(name3, 3, self.color_o1),

        )

        fig.update_layout(
            title={
                'text': '<b>' + title + '<b>',
                'y': 0.94,  # 位置
                'x': 0.2,
                'xanchor': 'center',  # 相對位置
                'yanchor': 'top'
            },
            bargap=0.15,  # 相鄰座標條之間距
            bargroupgap=0.08,  # 僅在共享座標軸時起作用 # 相同位置坐標的條之間的間隙
            xaxis_title="",  # 橫軸軸標籤
            # yaxis_title=target_title,  # 縱軸軸標籤
            yaxis_title_standoff=10,  # 調整 y 軸刻度標籤與圖表之間的距離
            yaxis_showtickprefix='first',
            font=dict(size=14),  # 設置字體大小
            # template='plotly_white',  # 換默認模板
            autosize=True,
            height=height,  # 設置繪圖的高度（以像素為單位）
            width=width,
            hovermode="x",  # 以橫軸顯示每個點的細節

        )
        # Add range slider
        #(a, b, c) = (pd.Period(x1[1], freq='Q').start_time, pd.Period(x1[0], freq='Q').start_time,pd.Period(x1[-1], freq='Q').start_time)
        #months_difference = rd(a, b).months
        #if months_difference == 3:
        if Title=='季資料':
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=2,
                                 label="2y",
                                 step="year",
                                 stepmode="backward"),

                            dict(count=3,
                                 label="3y",
                                 step="year",
                                 stepmode="backward"),
                            dict(count=4,  # 設置範圍要採取的步數
                                 label="4y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=5,  # 設置範圍要採取的步數
                                 label="5y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=10,  # 設置範圍要採取的步數
                                 label="10y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=15,  # 設置範圍要採取的步數
                                 label="15y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=20,  # 設置範圍要採取的步數
                                 label="20y",
                                 step="year",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )
        else:
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([

                            dict(count=6,
                                 label="6m",
                                 step="month",
                                 stepmode="backward"),

                            dict(count=12,
                                 label="1y",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=18,  # 設置範圍要採取的步數
                                 label="1.5y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=24,  # 設置範圍要採取的步數
                                 label="2y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(count=36,  # 設置範圍要採取的步數
                                 label="3y",
                                 step="month",  # count值設置範圍所依據的度量單位
                                 stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(
                        visible=False,#True,
                        autorange=False,  # 確定是否根據輸入數據計算範圍滑塊範圍。如果提供了 `range`，則 `autorange` 設置為“False”。
                        # range=date_list[date_range[0]:date_range[1]], 錯誤

                        bgcolor='rgb(255,228,225)',  # 設置範圍滑塊的背景顏色 rgb(255,250,250) rgb(255,228,225)
                        thickness=0.09,  # 範圍滑塊的高度佔繪圖區域總高度的%
                    ),
                    range=[x1[0], x1[-1] + rd(months=1)],  # 設置範圍滑塊的範圍。如果未設置，則默認為完整的 x 軸範圍。#+rd(months=1)
                    rangeslider_range=[x1[0], x1[-1] + rd(months=1)],  # +rd(months=1)
                    type="date",
                )
            )

        data = pd.concat([y1, y2, y3], axis=1)

        return (fig, data)

    def Dash_Chart_n(self, data0, title, dashname):
        # --------
        data0 = data0.dropna(how='all')
        date_list = pd.date_range(start=data0.index[0], end=(data0.index[-1]), freq='MS')  # .strftime('%y-%m-%d')
        T = len(date_list)

        # --------
        now = datetime.now()
        file_time = datetime.strftime(now, '%Y-%m-%d')

        if title == '月資料':
            col_list = self.tw_datam_col.copy()
            datat = self.tw_datamtT
            d_shift = 12
            d_freq = 'MS'

        elif title == '季資料':
            col_list = self.tw_dataqtT.columns.tolist()
            datat = self.tw_dataqtT
            d_shift = 4
            d_freq = '3MS'

        else :
            col_list = ["待放入"]
            datat = pd.DataFrame()
            d_shift = 0
            d_freq = 'D'

        col_list2 = col_list.copy()
        col_list2.insert(0, '(不選擇)')
        # ----------------------------------------------------------------------

        app = DjangoDash(dashname, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.css.append_css({'external_url': '/static/css/freestyle.css'})
        app.layout = dbc.Container([
            # html.Br(),
            # html.Br(),
            # html.Center(html.H4(title,className="f02 fw-light")),
            html.Br(),
            html.P(
                html.A(
                    html.Button("下載互動式圖檔", className='Button3'),
                    id="download",
                    style={"text-align": "right"}), style={"text-align": "right"}),
                dcc.Download(id='download_1'),

            html.P("( 圖形右上方隱藏選單內有 PNG 檔載點與重置圖形紐 )",
                   style={"text-align": "right", "font-weight": "bold"}),  #
            html.Div([
                dcc.Loading(dcc.Graph(id="graph",
                                      config={"displaylogo": False,'scrollZoom': False,  # 'scrollZoom': True 滑鼠滾輪縮放開起
                                              'toImageButtonOptions': {'filename': "{}_{}".format(title, file_time)}}),

                            type="graph"),  # 改預設"cube" 為 "graph",
            ], style={'height': '800px'}),

            html.Div(
                [
                    dbc.Button(
                        "動態調整圖形大小",
                        id="collapse-button_fig",
                        className="Button1",
                        color="primary",
                        n_clicks=0,

                        # 可設定預設值 is_open=True顯示內容和is_open=False隱藏內容
                    ),

                    dbc.Collapse(
                        html.Div([
                            html.P(),
                            html.Div([
                                html.P('長度 '),
                                dcc.Slider(id='SliderHeight', min=400, max=800, step=25, value=750,
                                           marks={x: str(x) for x in [400, 500, 600, 700, 800]}),
                            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
                            html.Div([
                                html.P('寬度 '),
                                dcc.Slider(id='SliderWidth', min=800, max=1200, step=25, value=1000,
                                           marks={x: str(x) for x in
                                                  [800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200]}),
                            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
                        ]),
                        id="collapse_fig",
                        is_open=False,
                    ),
                ],style={"text-align": "right"}),
            html.Div(id='pandas-output-container-1', style={"paddingTop": 10, 'color': '#178CA4'}),
            dbc.Alert(id='pandas-output-container-1', style={"paddingTop": 10}, color="primary"),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P('欲呈現的變數 1 '),
                        dcc.Dropdown(  # 欲選擇的
                            id='industry1',
                            options=col_list,  # tw_datam_col2 = self.tw_datam_col.insert(0, None)
                            value=col_list[0],
                            searchable=True,  # 啟用搜索功能 允許搜索關鍵字
                            clearable=False,  # 是否顯示清除選項的按鈕
                            optionHeight=70,  # 各別選項高度
                            maxHeight=200,  # 下拉高度，展開的下拉菜單的高度默認為 200px
                            # inline=True
                        ),
                    ]),
                ], width=4),
                dbc.Col([
                    html.P("資料形式:"),
                    dbc.RadioItems(
                        id='selection1_0',
                        options=[{'label': 'level 值', 'value': 'level'}, {'label': 'yoy 值', 'value': 'yoy'}],
                        value="level",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([
                    html.P("圖之呈現方式:"),
                    dbc.RadioItems(
                        id='selection1',
                        options=[{'label': '折線圖', 'value': 's'}, {'label': '直方圖', 'value': 'b'}],
                        value="s",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([]),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P('欲呈現的變數 2 '),
                        dcc.Dropdown(  # 欲選擇的
                            id='industry2',
                            options=col_list2,
                            # value=self.tw_datam_col[0],
                            searchable=True,  # 啟用搜索功能 允許搜索關鍵字
                            clearable=False,  # 是否顯示清除選項的按鈕
                            optionHeight=70,  # 各別選項高度
                            maxHeight=200,  # 下拉高度，展開的下拉菜單的高度默認為 200px
                            # inline=True
                        ),
                    ]),
                ], width=4),
                dbc.Col([
                    html.P("資料形式:"),
                    dbc.RadioItems(
                        id='selection2_0',
                        options=[{'label': 'level 值', 'value': 'level'}, {'label': 'yoy 值', 'value': 'yoy'}],
                        value="level",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([
                    html.P("圖之呈現方式:"),
                    dbc.RadioItems(
                        id='selection2',
                        options=[{'label': '折線圖', 'value': 's'}, {'label': '直方圖', 'value': 'b'}],
                        value="s",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([
                    html.P("是否與第1個變數共用縱座標軸:"),

                    dbc.RadioItems(
                        id='selection2_1',
                        options=[{'label': '是', 'value': 'y'}, {'label': '否', 'value': 'n'}],
                        value="n",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    ),

                ]),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.P('欲呈現的變數 3 '),
                        dcc.Dropdown(  # 欲選擇的
                            id='industry3',
                            options=col_list2,
                            # value=self.tw_datam_col[0],
                            searchable=True,  # 啟用搜索功能 允許搜索關鍵字
                            clearable=False,  # 是否顯示清除選項的按鈕
                            optionHeight=70,  # 各別選項高度
                            maxHeight=200,  # 下拉高度，展開的下拉菜單的高度默認為 200px
                            # inline=True
                        ),
                    ]),
                ], width=4),
                dbc.Col([
                    html.P("資料形式:"),
                    dbc.RadioItems(
                        id='selection3_0',
                        options=[{'label': 'level 值', 'value': 'level'}, {'label': 'yoy 值', 'value': 'yoy'}],
                        value="level",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([
                    html.P("圖之呈現方式:"),
                    dbc.RadioItems(
                        id='selection3',
                        options=[{'label': '折線圖', 'value': 's'}, {'label': '直方圖', 'value': 'b'}],
                        value="s",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    )
                ]),
                dbc.Col([
                    html.P("是否與第1個變數共用縱座標軸:"),

                    dbc.RadioItems(
                        id='selection3_1',
                        options=[{'label': '是', 'value': 'y'}, {'label': '否', 'value': 'n'}],
                        value="n",  # 預設選項
                        switch=True,
                        labelCheckedStyle={"color": "rgb(255,99,71)"},
                        #inline=False,#True,
                        inline=True,
                    ),

                ]),
                html.P('(直方圖選「是」才會並排，否則重疊)', style={'color': '#178CA4', "text-align": "right"}),
            ]),

                    html.P([
                        html.A(
                            html.Button("下載圖形資料", className='Button3 me-md-2'),
                            id="download_excel",
                            style={"text-align": "legft"}),
                        dcc.Download(id='download_excel_o'),

                        html.A(
                            html.Button("下載所有資料", className='Button3  me-md-2'),
                            id="download_excelall",
                            style={"text-align": "legft"}),
                        dcc.Download(id='download_excelall_o'),



                    dbc.Button(
                        "打開圖形資料",
                        id="collapse-button",
                        className="Button1",
                        color="primary",
                        n_clicks=0,
                        # 可設定預設值 is_open=True顯示內容和is_open=False隱藏內容
                    ),


                    ], style={"text-align": "legft"}),
            dbc.Collapse(
                html.Div(

                    [html.P(
                        '(可以透過點選欄位名稱最左邊的上下箭頭按鈕, 來達到升降冪排序)',
                        style={'color': '#178CA4'}),
                        html.P(
                            '(可以在 filter data... 部份直接輸入條件式做即時搜尋,舉例：在 Date 欄位輸入 >=2010-03)',
                            style={'color': '#178CA4'}),
                        html.P('(資料表中,橘色底色與紫色底色分別代表該欄位前、後百分之時十的數值)',
                               style={'color': '#178CA4'}),
                        dash_table.DataTable(
                            id='table-editing-simple',
                            editable=False,  # 是否可以編輯表格
                            filter_action="native",
                            export_format='xlsx',
                            export_headers='display',
                            merge_duplicate_headers=True,
                            page_action='native',
                            page_current=0,
                            page_size=24,
                            sort_action='native',
                            # sort_action='custom'適用於表頭固定名稱~並在回調中定義應該如何進行排序（sort_by輸入和data輸出在哪裡）
                            sort_mode='multi',  # 對於多列排序 ;  默認'single'按單個列執行排序
                            sort_by=[],
                            style_cell={'textAlign': 'left'},  # 向左對齊 (補充:“cell”是整個表格，“header”只是標題行，“data”只是數據行)
                            style_as_list_view=True,  # 將表格樣式化為列表
                            style_header={  # 標題的列表的 CSS 樣式
                                'backgroundColor': '#178CA4',
                                'color': 'rgb(255, 255, 255)',
                                'fontWeight': 'bold',
                                # 'border': '1px solid pink' #邊框
                            },
                            style_data={  # 數據行的 CSS 樣式
                                'backgroundColor': 'rgb(240,248,255)',
                                'border': '1px solid pink'
                            },
                            style_filter={  # 過濾器單元格的 CSS 樣式
                                'backgroundColor': 'rgb(255,240,245)',
                                'border': '1px solid pink'
                            }
                        ),
                    ]),
                id="collapse",
                is_open=False,
            ),

            html.Br(),

        ], style={'height': '800px',"margin-right": "2%", "margin-left": "2%"})

        @app.callback(
            Output('pandas-output-container-1', 'children'),
            Input("industry1", "value"),
            Input("industry2", "value"),
            Input("industry3", "value"))
        def render_content(name1,name2,name3):
            my_list = [name1,name2,name3]# 合成一個列表
            unique_list = []# 新的列表，用於存儲不重複的值
            for item in my_list:# 歷原始列表，將不重複的值添加到新列表中
                if item not in unique_list:
                    unique_list.append(item)
            unique_list = list(filter(lambda x: x not in [None,'(不選擇)'], unique_list)) #排除未點選之None與不選擇
            n = len(unique_list)

            if n == 1:
                name1 = unique_list[0]
                return [dmc.Text(f"資料說明："),dmc.Text(f"【{name1}】 1.資料來源為{datat.loc['資料來源',name1]} 2.單位或說明 {datat.loc['單位或說明',name1]}")]
            elif   n == 2:
                name1 = unique_list[0]
                name2 = unique_list[1]
                return [dmc.Text(f"資料說明："),dmc.Text(f"【{name1}】 1.資料來源為{datat.loc['資料來源',name1]} 2.單位或說明 {datat.loc['單位或說明',name1]}"),dmc.Text(f"【{name2}】 1.資料來源為{datat.loc['資料來源',name2]} 2.單位或說明 {datat.loc['單位或說明',name2]}")]

            else :
                name1 = unique_list[0]
                name2 = unique_list[1]
                name3 = unique_list[2]
                return [dmc.Text(f"資料說明："),dmc.Text(f"【{name1}】 1.資料來源為{datat.loc['資料來源',name1]} 2.單位或說明 {datat.loc['單位或說明',name1]}"),dmc.Text(f"【{name2} 】 1.資料來源為{datat.loc['資料來源',name2]} 2.單位或說明 {datat.loc['單位或說明',name2]}"),dmc.Text(f"【{name3}】 1.資料來源為{datat.loc['資料來源',name3]} 2.單位或說明 {datat.loc['單位或說明',name3]}")]

        @app.callback(
            Output("collapse", "is_open"),
            [Input("collapse-button", "n_clicks")],
            [State("collapse", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("collapse_fig", "is_open"),
            [Input("collapse-button_fig", "n_clicks")],
            [State("collapse_fig", "is_open")],
        )
        def toggle_collapse(n, is_open):
            if n:
                return not is_open
            return is_open

        @app.callback(
            Output("graph", "figure"),
            Output('table-editing-simple', 'data'),
            Output('table-editing-simple', 'columns'),
            Output('table-editing-simple', 'style_data_conditional'),
            Input("industry1", "value"),
            Input("industry2", "value"),
            Input("industry3", "value"),
            Input("selection1_0", "value"),
            Input("selection2_0", "value"),
            Input("selection3_0", "value"),
            Input("selection1", "value"),
            Input("selection2", "value"),
            Input("selection3", "value"),
            Input("selection2_1", "value"),
            Input("selection3_1", "value"),
            # Input('date_range_slider', 'value'),
            Input('SliderHeight', 'value'),
            Input('SliderWidth', 'value'),
        )
        def display_barcahrt(name1, name2, name3, data_type1, data_type2, data_type3, chart_type1, chart_type2, chart_type3,
                             shared_yaxis, shared_yaxis3, height, width):  # ,date_range
            # data1 = data0[name1][date_range[0]: date_range[1]].dropna(how='all')


            my_list = [name1, data_type1, name2, data_type2, name3, data_type3]
            indices_nan = [i for i, x in enumerate(my_list) if x is None or x == '(不選擇)']  # 找出所有 None 與 '(不選擇)' 的位置
            indices_to_remove = [x for x in indices_nan] + [x + 1 for x in indices_nan]  # 組合出真正要刪除的元素位置
            new_list = [x for i, x in enumerate(my_list) if
                        i not in indices_to_remove]  # 刪除指定位置的元素(None 與 '(不選擇)' 及對應的資料型態)
            ### 要解決重複值
            pairs = [(new_list[i], new_list[i + 1]) for i in range(0, len(new_list), 2)]  # 將列表按順序兩兩一組
            unique_list = list(set(pairs))  # 排除掉重複的組
            for i in range(len(unique_list)):  # 要先解決，不然遇到同樣的欄位名稱後面才yoy會出錯
                item = unique_list[i]
                if item[1] == 'yoy':
                    data0[f'{item[0]}_YoY'] = data0[item[0]] / data0[item[0]].shift(d_shift) - 1
                    # item[0] = item[0]+'_YoY' 會出錯所以，創建一個包含修改過的 item 的新元組，然後替換掉列表中的舊元組
                    unique_list[i] = (item[0] + '_YoY', item[1])
                else:
                    pass

            n = len(unique_list)

            if n == 1:
                # 所以生成連續的日期範圍，確保沒有遺
                name1 = unique_list[0][0]
                data1 = pd.concat([data0[name1]], axis=1).dropna(how='all')  # 刪除所有全為nan的列，但會有在中間缺某時間點的情形出現
                # 沒有刪除nan則作圖會有很多空白處漏時間，頭尾也不會有nan(上一行刪除)
                date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq=d_freq)
                data1 = data1.reindex(date_range)  # 重新索引DataFrame


                (x1, y1) = (data1.index, data1[name1])
                (fig, data) = self.fig1(title, x1, y1, name1, chart_type1, height, width)
            elif n == 2:
                name1 = unique_list[0][0]
                name2 = unique_list[1][0]

                data1 = pd.concat([data0[name1], data0[name2]], axis=1).dropna(how='all')  # 沒有刪除nan則作圖會有很多空白處
                # 所以生成連續的日期範圍，確保沒有遺漏時間，頭尾也不會有nan(上一行刪除)
                date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq=d_freq)
                data1 = data1.reindex(date_range)  # 重新索引DataFrame


                (x1, y1) = (data1.index, data1[name1])
                (x2, y2) = (data1.index, data1[name2])
                (fig, data) = self.fig2(title, x1, y1, name1, chart_type1, x2, y2, name2, chart_type2, shared_yaxis, '',
                                        height, width)
            else:
                name1 = unique_list[0][0]
                name2 = unique_list[1][0]
                name3 = unique_list[2][0]

                data1 = pd.concat([data0[name1], data0[name2], data0[name3]], axis=1).dropna(how='all')  # 沒有刪除nan則作圖會有很多空白處
                # 所以生成連續的日期範圍，確保沒有遺漏時間，頭尾也不會有nan(上一行刪除)
                date_range = pd.date_range(start=data1.index.min(), end=data1.index.max(), freq=d_freq)
                data1 = data1.reindex(date_range)  # 重新索引DataFrame

                (x1, y1) = (data1.index, data1[name1])
                (x2, y2) = (data1.index, data1[name2])
                (x3, y3) = (data1.index, data1[name3])
                (fig, data) = self.fig3(title, x1, y1, name1, chart_type1, x2, y2, name2, chart_type2, shared_yaxis, x3,
                                        y3,
                                        name3, chart_type3, shared_yaxis3, '', height, width)

            fig.write_html(self.temporary_path + "/{}_{}.html".format(title, unique_list), config= {'displaylogo': False})
            data1.to_csv(self.temporary_path + "/{}_{}.csv".format(title, unique_list), index=True,
                         encoding='utf-8-sig')

            data = data.rename_axis('Date').reset_index()
            import datetime
            # 將 'Date' 欄位轉換為 datetime
            data['Date'] = pd.to_datetime(data['Date'])  # datetime64[ns]
            data['Date'] = data['Date'].dt.strftime("%Y-%m")  # object

            data_col_name = data.columns.values.tolist()
            table_columns = [{'id': col, 'name': col, 'deletable': True, 'renamable': True} for col in data_col_name]

            table_data = [dict(Model=i, **{col: data.iloc[i][col] for col in data_col_name}) for i in range(len(data))]

            style_data_conditional = (
                    [
                        {
                            'if': {'row_index': 'odd'},  # 數據行樣式 且控制偶數行
                            'backgroundColor': 'rgb(176,224,230)',
                        }
                    ] +
                    [
                        {
                            'if': {
                                'filter_query': '{{{}}} >= {}'.format(col, value),
                                'column_id': col
                            },
                            'backgroundColor': 'rgb(255,127,80)',
                            'color': 'white'
                        } for (col, value) in data.loc[:, data.columns[1:]].quantile(0.9).items()
                    ] +
                    [
                        {
                            'if': {
                                'filter_query': '{{{}}} <= {}'.format(col, value),
                                'column_id': col
                            },
                            'backgroundColor': 'rgb(218,112,214)',
                            'color': 'white'
                        } for (col, value) in data.loc[:, data.columns[1:]].quantile(0.1).items()
                    ]
            )

            return (fig, table_data, table_columns, style_data_conditional)

        @app.callback(
            Output('download_1', 'data'),
            Input('download', 'n_clicks'),
            [State("industry1", "value"),
            State("industry2", "value"),
            State("industry3", "value"),
            State("selection1_0", "value"),
            State("selection2_0", "value"),
            State("selection3_0", "value")],
            prevent_initial_call=True)
        def download_html(n_clicks ,name1, name2, name3, data_type1, data_type2, data_type3):
            if n_clicks != None:
                my_list = [name1, data_type1, name2, data_type2, name3, data_type3]
                indices_nan = [i for i, x in enumerate(my_list) if x is None or x == '(不選擇)']  # 找出所有 None 與 '(不選擇)' 的位置
                indices_to_remove = [x for x in indices_nan] + [x + 1 for x in indices_nan]  # 組合出真正要刪除的元素位置
                new_list = [x for i, x in enumerate(my_list) if
                            i not in indices_to_remove]  # 刪除指定位置的元素(None 與 '(不選擇)' 及對應的資料型態)
                ### 要解決重複值
                pairs = [(new_list[i], new_list[i + 1]) for i in range(0, len(new_list), 2)]  # 將列表按順序兩兩一組
                unique_list = list(set(pairs))  # 排除掉重複的組
                for i in range(len(unique_list)):  # 要先解決，不然遇到同樣的欄位名稱後面才yoy會出錯
                    item = unique_list[i]
                    if item[1] == 'yoy':
                        data0[f'{item[0]}_YoY'] = data0[item[0]] / data0[item[0]].shift(d_shift) - 1
                        # item[0] = item[0]+'_YoY' 會出錯所以，創建一個包含修改過的 item 的新元組，然後替換掉列表中的舊元組filename
                        unique_list[i] = (item[0] + '_YoY', item[1])
                    else:
                        pass
                file = self.temporary_path + "/{}_{}.html".format(title, unique_list)
                print(f"88888888'{file}'")
                return dcc.send_file(file)

        @app.callback(
            Output('download_excel_o', 'data'),
            Input('download_excel', 'n_clicks'),
            [State("industry1", "value"),
            State("industry2", "value"),
            State("industry3", "value"),
            State("selection1_0", "value"),
            State("selection2_0", "value"),
            State("selection3_0", "value")],
            prevent_initial_call=True)
        def download_csv(n_clicks,name1, name2, name3, data_type1, data_type2, data_type3):
            if n_clicks != None:
                my_list = [name1, data_type1, name2, data_type2, name3, data_type3]
                indices_nan = [i for i, x in enumerate(my_list) if x is None or x == '(不選擇)']  # 找出所有 None 與 '(不選擇)' 的位置
                indices_to_remove = [x for x in indices_nan] + [x + 1 for x in indices_nan]  # 組合出真正要刪除的元素位置
                new_list = [x for i, x in enumerate(my_list) if
                            i not in indices_to_remove]  # 刪除指定位置的元素(None 與 '(不選擇)' 及對應的資料型態)
                ### 要解決重複值
                pairs = [(new_list[i], new_list[i + 1]) for i in range(0, len(new_list), 2)]  # 將列表按順序兩兩一組
                unique_list = list(set(pairs))  # 排除掉重複的組
                for i in range(len(unique_list)):  # 要先解決，不然遇到同樣的欄位名稱後面才yoy會出錯
                    item = unique_list[i]
                    if item[1] == 'yoy':
                        data0[f'{item[0]}_YoY'] = data0[item[0]] / data0[item[0]].shift(d_shift) - 1
                        # item[0] = item[0]+'_YoY' 會出錯所以，創建一個包含修改過的 item 的新元組，然後替換掉列表中的舊元組filename
                        unique_list[i] = (item[0] + '_YoY', item[1])
                    else:
                        pass

                file = self.temporary_path + "/{}_{}.csv".format(title, unique_list)
                print(f"88888888'{file}'")
                return dcc.send_file(file)

        @app.callback(
            Output('download_excelall_o', 'data'),
            Input('download_excelall', 'n_clicks'), prevent_initial_call=True)
        def download_excel(n_clicks ):
            if n_clicks != None:

                return dcc.send_file(self.excel_file, filename="歷史資料.xlsx")
        return app

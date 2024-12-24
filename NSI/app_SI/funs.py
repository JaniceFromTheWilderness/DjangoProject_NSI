import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from dash import Dash, dcc, html, Input, Output,State, ctx, callback,dash_table
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import re
from plotly.subplots import make_subplots
import io
import base64
import json
import itertools
import torch
from torch.utils.data import DataLoader, Dataset
from multiprocessing import Pool
import warnings
import time  # 用於計算處理時間
warnings.simplefilter(action='ignore', category=FutureWarning)
# 所有的函數
def file_list(path):
    files = os.listdir(path)# 取得資料夾中的所有檔案
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True) # 將檔案按照修改時間排序
    return files

def json_file(dicts,model,file_path,file_name):
    file_name = file_name if file_name.endswith('.json') else file_name+'.json'
    if model=='write':
        # 將字典列表存為 JSON 檔案
        with open(file_path+'/'+file_name, 'w', encoding='utf-8') as file:
            json.dump(dicts, file, indent=4)
        print(f'Successfully saved dictionary list as JSON file （{file_name}.json）')
    elif model=='read':
        # 從 JSON 檔案讀取字典列表
        with open(file_path+'/'+file_name, 'r', encoding='utf-8') as file:
            loaded_dicts = json.load(file)
        return loaded_dicts

def extract_keywords(sentence):
    match = re.search(r'([\u4e00-\u9fff]+關鍵字)', sentence)
    if match:
        return match.group(1)
    return None

def generate_table(self,df, **kwargs): #將dataframe生成可以呈現於dash的表
    #for col, col_new in zip(['Industry','COUNTYNAME'],['產業',  '縣市']):
    #    if col in list(df.columns):
    #    df = df.rename(columns={col: col_new})
    #max_rows = 10
    if ('contains_index' in kwargs) == True:
        df.reset_index(inplace=True)
    return (
        # Header
        html.Table([html.Tr([html.Th(col) for col in df.columns])] + # self.Ind_list
                   # Body
                   [html.Tr([
                       html.Td(df.iloc[i][col]) for col in df.columns
                   ]) for i in range(len(df))] #range(min(len(data), max_rows)) 可以有所限制
                   ))
# -------------------------------------------------------------------------------------

def plotly_line_chart(df, x_column, y_columns, title, x_title, y_title, output_file=None, colors=None,
                      range_slider=None, height=None, width=None, bar_columns=None, bar_title=None, subplots=None,
                      bar_colors=None):
    """
    參數:
    - df: DataFrame, 資料來源
    - x_column: str, 橫軸對應的資料欄位
    - y_columns: list, 縱軸對應的多個資料欄位列表
    - title: str, 圖表標題
    - x_title: str, 橫軸標題
    - y_title: str, 縱軸標題
    - output_file: str, 選填，儲存圖檔的檔案路徑，副檔名可為 `.png`, `.jpeg`, `.webp`, `.html`
    - bar_columns: list, 用於柱狀圖的資料欄位
    - bar_title: str, 柱狀圖的縱軸標題

    回傳:
    - None
    """
    # 判斷 x_column 是否在索引中，如果是則使用索引作為 x 軸
    if x_column in df.index.names:
        x_data = df.index
    else:
        x_data = df[x_column]

    if subplots:  # [0.7, 0.3]
        # 創建子圖結構，上方折線圖，下方柱狀圖
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,  # 兩行一列共用x軸
                            vertical_spacing=0.08,  # 調整垂直間距，默認是 0.1
                            row_heights=subplots,  # 上下方比例
                            # subplot_titles=(y_title, bar_title)
                            )
    else:
        fig = go.Figure()

    if bar_columns:
        # 添加直條圖（柱狀圖），設置它在後面
        # 動態添加多組直條圖（柱狀圖），每個對應 bar_columns 中的欄位
        colorsb = bar_colors if bar_colors else ['lightblue', 'orange', 'lightgreen', 'purple', 'red']  # 預設顏色列表
        for i, bar_col in enumerate(bar_columns):
            # for bar_col ,color in zip(bar_columns,bar_colors):
            if bar_col in df.columns:
                if subplots:
                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=df[bar_col],
                        name=bar_col,
                        marker_color=colorsb[i % len(colorsb)],  # 依序分配不同顏色
                        # marker_color=color,
                        yaxis='y2',

                    ), row=2, col=1)
                    fig.update_layout(
                        yaxis2=dict(
                            title=bar_title,
                            ###overlaying='y',  # 疊加在主 y 軸上
                            side='right',  # 第二個 y 軸顯示在右邊
                            showgrid=False  # 隱藏第二個 Y 軸的網格線
                        ),
                        legend=dict(
                            x=1.07,  # 將圖例移動到圖表外部，避免覆蓋 Y 軸
                            y=1,
                            traceorder="normal",  # 按繪製的順序顯示圖例  #"reversed"：按相反的順序顯示圖例，最後繪製的圖例顯示在最上面
                        )
                    )
                else:
                    fig.add_trace(go.Bar(
                        x=x_data,
                        y=df[bar_col],
                        name=bar_col,
                        marker_color=colorsb[i % len(colorsb)],  # 依序分配不同顏色
                        opacity=0.25,  # 調整透明度讓折線圖更顯眼
                        yaxis='y2',

                    ))
                    fig.update_layout(
                        yaxis2=dict(
                            title=bar_title,
                            overlaying='y',  # 疊加在主 y 軸上
                            side='right',  # 第二個 y 軸顯示在右邊
                            showgrid=False  # 隱藏第二個 Y 軸的網格線
                        ),
                        legend=dict(
                            x=1.07,  # 將圖例移動到圖表外部，避免覆蓋 Y 軸
                            y=1,
                            traceorder="normal",  # 按繪製的順序顯示圖例  #"reversed"：按相反的順序顯示圖例，最後繪製的圖例顯示在最上面
                        )
                    )

            else:
                print(f"Warning: bar_column '{bar_col}' 不存在於 DataFrame 中，該柱狀圖不會繪製。")

    # 設定圖表標題與軸標題
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis=dict(
            title=y_title,
            title_standoff=2,  # 調整與軸線的距離
            side='left',
            automargin=True,
            tickangle=0,  # 保持刻度水平顯示
            rangemode='normal',  # 允許縮放
            fixedrange=False  # Y 軸可滾動
        ),
        template='plotly_white',

        hovermode="x",
        hoverlabel=dict(namelength=-1),  # 顯示完整的懸浮標籤名稱
        font=dict(size=16),  # 設置字體大小

    )
    if colors:
        for col, color in zip(y_columns, colors):
            if col in df.columns:
                if subplots:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=df[col],
                        name=col,
                        line=dict(color=color, width=2)
                    ), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=df[col],
                        name=col,
                        line=dict(color=color, width=2)
                    ))
            else:
                print(f"Warning: y_column '{col}' 不存在於 DataFrame 中，該線不繪製。")



    else:
        # 為每個指定的縱軸欄位繪製線條
        for col in y_columns:
            if col in df.columns:
                if subplots:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=df[col],
                        line=dict(width=2),
                        name=col
                    ), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(
                        x=x_data,
                        y=df[col],
                        line=dict(width=2),
                        name=col
                    ))
            else:
                print(f"Warning: y_column '{col}' 不存在於 DataFrame 中，該線不繪製。")

    # 添加水平線 y=0
    if subplots:
        fig.add_shape(
            type="line",
            x0=x_data.min(),  # 水平線起點的 x 值
            x1=x_data.max(),  # 水平線終點的 x 值
            y0=0,  # 水平線起點的 y 值
            y1=0,  # 水平線終點的 y 值
            line=dict(
                color="lightgray",  # 線的顏色
                width=3,  # 線的寬度
                dash="solid"  # 線的樣式 ('solid', 'dot', 'dash' etc.)
            ),
            layer="below",  # 將水平線置於所有數據線之下
            xref="x1",  # 指定應用於第一個子圖的x軸
            yref="y1",  # 指定應用於第一個子圖的y軸
        )
    else:
        fig.add_shape(
            type="line",
            x0=x_data.min(),  # 水平線起點的 x 值
            x1=x_data.max(),  # 水平線終點的 x 值
            y0=0,  # 水平線起點的 y 值
            y1=0,  # 水平線終點的 y 值
            line=dict(
                color="lightgray",  # 線的顏色
                width=3,  # 線的寬度
                dash="solid"  # 線的樣式 ('solid', 'dot', 'dash' etc.)
            ),
            layer="below"  # 將水平線置於所有數據線之下
        )

    if height:
        # 設置繪圖的高度（以像素為單位）
        fig.update_layout(height=height)
    if width:
        # 設置繪圖的寬度（以像素為單位）
        fig.update_layout(width=width)

    if range_slider:
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeslider=dict(
                    visible=True,
                    thickness=0.05  # 減少範圍滑塊的厚度，避免遮擋刻度標籤
                ),
                type="date",
            )
        )

    # 顯示圖表 fig.show(config={'displaylogo': False})

    # 儲存圖檔
    if output_file:
        if output_file.endswith('.html'):
            fig.write_html(output_file, config={'displaylogo': False})

        else:
            # 可選的格式包括: 'png', 'jpeg', 'webp', 'svg'
            fig.write_image(output_file, engine="kaleido")
        print(f'Successfully saved （ {output_file} ）')
    return fig


def Dash_Chart(merged_df,title,dashname):
    merged_df.columns = [item.replace("sentiment score_", "") for item in list(merged_df.columns)]  # 修改所有欄位名稱
    bar_col_list = [col for col in list(merged_df.columns) if '每日新聞數量' in col]

    si_dict = {'[dict1] Bian, S., Jia, D., Li, F., & Yan, Z. (2021). A new Chinese financial sentiment dictionary for textual analysis in accounting and finance.':'dict1',
                    '[dict2] Du, Z., Huang, A. G., Wermers, R., & Wu, W. (2022). Language and domain specificity: A Chinese financial sentiment dictionary.':'dict2',
                    '[intersection] The intersection of D1 and D2':'intersection',
                    '[union] Union of D1 and D2':'union'}

    key_dict = {'總體經濟面關鍵字': '總體經濟面關鍵字',
                 '產業面關鍵字': '產業面關鍵字',
                 '不分類關鍵字': '不分類關鍵字'}
    method_dict = {'方法1':'方法1',
                    '方法2':' _with',
                    '方法3':' d_by ws_n',
                    '方法4':'_sentences_m',
                    '方法5':['_sentences_m','d_by ws_n']}


    # --------
    app = DjangoDash(dashname, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.css.append_css({'external_url': '/static/css/freestyle.css'})

    app.layout = dbc.Container([
        html.Br(),
        html.Br(),
        #html.Center(html.H4(title, className="f02 fw-light")),
        html.Br(),
        html.P('選擇想呈現哪些[情緒]字典之結果:'),
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="si_all",
                    n_clicks=0,
                )
            ), style={'display': 'inline-table'}),  # 呈現並排
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="si_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        dbc.Checklist(  # 欲選擇的
            id='si',
            options=list(si_dict.keys()),
            value=list(si_dict.keys())[3],
            labelCheckedStyle={'color': 'red'},
            labelStyle={'display': 'block'},
            inline=True,
        ),
        html.Br(),
        html.P('選擇想呈現哪些[關鍵字]字典之結果:'),
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="key_all",
                    n_clicks=0,
                )
            ), style={'display': 'inline-table'}),  # 呈現並排
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="key_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        dbc.Checklist(  # 欲選擇的
            id='key',
            options=list(key_dict.keys()),
            value=key_dict[list(key_dict.keys())[-1]],
            labelCheckedStyle={'color': 'red'},
            labelStyle={'display': 'block'},
            inline=True,
        ),
        html.Br(),
        html.Br(),
        html.P('選擇想呈現哪些[方法]之結果:'),
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="m_all",
                    n_clicks=0,
                )), style={'display': 'inline-table'}),
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="m_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        dbc.Checklist(  # 欲選擇的
            id='m',
            options=list(method_dict.keys()),
            value=list(method_dict.keys())[-1],
            labelCheckedStyle={'color': 'red'},
            labelStyle={'display': 'block', 'margin': '0'},
            # inline=True
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(4, 1fr)',
            },
        ),
        html.Br(),
        html.Div(
            [
                dbc.Button(
                    "打開隱藏資料",
                    id="collapse-button",
                    className="mb-3 Button1",
                    color="primary",
                    n_clicks=0,
                    # 可設定預設值 is_open=True顯示內容和is_open=False隱藏內容
                ),
                dbc.Collapse(
                    dash_table.DataTable(

                        id='table-editing-simple',
                        editable=False,  # 是否可以編輯表格
                        filter_action="native",
                        export_format='xlsx',
                        export_headers='display',
                        merge_duplicate_headers=True,
                        page_action='native',
                        page_current=0,
                        page_size=22,
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
                    id="collapse",
                    is_open=False,
                ),
            ]
        ),
        html.Br(),
        html.Div([
            html.P('動態調整圖形大小:'),
            html.Div([
                html.P('長度 '),
                dcc.Slider(id='SliderHeight', min=600, max=1000, step=25, value=750,
                           marks={x: str(x) for x in [600, 700, 800, 900, 1000]}),
            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
            html.Div([
                html.P('寬度 '),
                dcc.Slider(id='SliderWidth', min=1000, max=2000, step=25, value=1400,
                           marks={x: str(x) for x in [1000, 1200, 1400, 1600, 1800, 2000]}),
            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
        ]),
        html.Br(),
        html.P(
            html.A(
                html.Button("按一下將圖形載為 HTML 檔", className='Button1'),
                id="download",
                style={"text-align": "right"}), style={"text-align": "right"}),
        dcc.Download(id='download_1'),
        html.P("( 圖形右上方隱藏選單內有 PNG 檔載點與重置圖形紐 )",
               style={"text-align": "right", "font-weight": "bold"}),  #
        html.Div([
            dcc.Loading(dcc.Graph(id="graph",
                                  config={'scrollZoom': False,  # 'scrollZoom': True 滑鼠滾輪縮放開起
                                          'displaylogo': False,
                                          'toImageButtonOptions': {'filename': "圖表_「{}」".format(title)}}),
                        type="graph"),  # 改預設"cube" 為 "graph",
        ], style={'height': '1100px', 'width': '1500px'}),

    ], style={"margin-right": "2%", "margin-left": "2%"})

    @app.callback(
        Output("si", "value"),
        Input("si_all", "n_clicks"),
        Input("si_notall", "n_clicks"),
        State('si', 'options'),
        State("si_all", "n_clicks_timestamp"),
        State("si_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0
        print(n, '***', timestamp1)
        print('******************')
        print(n2, '***', timestamp2)
        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]
        if last_n_id == 'n':
            # 如果選中了“全選” 獲取所有選項的值
            selected_keys = list(si_dict.keys())
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            selected_keys = list(si_dict.keys())[-1]
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys

    @app.callback(
        Output("key", "value"),
        Input("key_all", "n_clicks"),
        Input("key_notall", "n_clicks"),
        State('key', 'options'),
        State("key_all", "n_clicks_timestamp"),
        State("key_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0
        print(n, '***', timestamp1)
        print('******************')
        print(n2, '***', timestamp2)

        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]
        if last_n_id == 'n':
            # 如果選中了“全選” 獲取所有選項的值
            selected_keys = list(key_dict.keys())
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            selected_keys = list(key_dict.keys())[-1]
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys


    @app.callback(
        Output("m", "value"),
        Input("m_all", "n_clicks"),
        Input("m_notall", "n_clicks"),
        State('m', 'options'),
        State("m_all", "n_clicks_timestamp"),
        State("m_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0
        print(n, '***', timestamp1)
        print('******************')
        print(n2, '***', timestamp2)

        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]
        if last_n_id == 'n':
            # 獲取所有選項的值
            all_values = [option for option in options]
            print('**** all_values :',all_values)
            # 如果選中了“全選” 獲取所有選項的值
            selected_keys = all_values
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            selected_keys = list(method_dict.keys())[-1]
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys

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
    Output("graph", "figure"),
        Output('table-editing-simple', 'data'),
        Output('table-editing-simple', 'columns'),
        Output('table-editing-simple', 'style_data_conditional'),
        Input("si", "value"),
        Input('key', "value"),
        Input('m', "value"),
        Input('SliderHeight', 'value'),
        Input('SliderWidth', 'value')
    )
    def display_barcahrt(si, keys, m, height, width):
        print('*** 原始變數 si, keys, m :', si, keys, m)
        # 檢查原始資料，並以防範措施確保其不會被意外拆解
        def clean_input(input_value):
            if input_value is None:
                return []  # 若為 None，轉為空列表
            elif isinstance(input_value, list):
                # 若為列表，確認每個元素為字串，不會拆成單字元
                return [item for item in input_value if len(item) > 1] #[item for item in m if len(item) > 1]
            elif isinstance(input_value, str):
                # 若為單一字串，包裝成單元素列表，避免被拆解成單字元
                return [input_value]
            else:
                # 若有其他情況，強制轉換為列表
                return [str(input_value)]

        # 使用 clean_input 函數來處理每個變數
        si = clean_input(si)
        keys = clean_input(keys)
        m = clean_input(m)
        print('*** 確認後的 si, keys, m :', si, keys, m)

        si_select =  [si_dict[key] for key in si]
        key_select =  [key_dict[key] for key in keys]
        m_select =  [method_dict[key] for key in m]
        m_select = list(
            itertools.chain.from_iterable([item] if not isinstance(item, list) else item for item in m_select))

        print('@@@@',si_select, key_select, m_select)
        filtered_columns =list(merged_df.columns)
        for select in [si_select, key_select, m_select]:
              filtered_columns = [item for item in filtered_columns if any(keyword in item for keyword in select)]
        print('@@@@~',filtered_columns)

        # 遍歷原始列表並提取符合條件的子字串
        key_name = []
        for item in filtered_columns:
            matches = re.search(r'[\u4e00-\u9fff]+關鍵字', item).group(0)
            key_name.append(matches)

        # 去除重複項目（如果需要）
        key_name = list(set(key_name))
        print('1263456~~',key_name)
        bar_columns = [bar for bar in bar_col_list if any(keyword in bar for keyword in key_name)]
        print('＠00＠~~', bar_columns)



        fig = plotly_line_chart(
                     merged_df,
                     x_column='新聞發布時間',
                     y_columns=filtered_columns,
                     title='每日情感分數變化',
                     x_title='',
                     y_title='分數',
                     #output_file=path+'/每日情感分數變化圖/sentiment_scores_all_withbar.html',
                     #colors=colors,
                     bar_columns=bar_columns,
                     bar_title='新聞數量',
                     range_slider=True,
                     subplots=[0.8,0.2],
                     #bar_colors=['dodgerblue','darkorange'],
                    height=height,
                    width=width
                   )
        fig.write_html("sentiment_scores_all_withbar.html")
       # fig.write_html("{}_「{}」.html".format(chart, title))
        # ----
        data = merged_df[bar_columns+filtered_columns]

        data_col_name = data.columns.values.tolist()
        table_columns = [{'id': col, 'name': col, 'deletable': True, 'renamable': True} for col in
                         data_col_name]

        table_data = [dict(Model=i, **{col: data.iloc[i][col] for col in data_col_name})
                      for i in range(len(data))]
        # 創建一個過濾條件，排除指定的欄位
        columns_to_keep = [col for col in data.columns if col not in ['產業', '縣市']]
        df_modified = data[columns_to_keep]
        print("*********************")
        print(df_modified)
        print("*********************")
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
                    } for (col, value) in df_modified.quantile(0.9).items()
                ] +
                [
                    {
                        'if': {
                            'filter_query': '{{{}}} <= {}'.format(col, value),
                            'column_id': col
                        },
                        'backgroundColor': 'rgb(218,112,214)',
                        'color': 'white'
                    } for (col, value) in df_modified.quantile(0.1).items()
                ]
        )

        return (fig, table_data, table_columns, style_data_conditional)

    @app.callback(
        Output('download_1', 'data'),
        Input('download', 'n_clicks'), prevent_initial_call=True)
    def download_html(n):
        return dcc.send_file("{}_「{}」.html".format(chart, title))

    return app


def load_stopwords(file_path): # 將停用詞字典讀成一個集合
    with open(file_path, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file])
    return stopwords
"""
        html.Div([
            dbc.Checklist(  # 欲選擇的
                id='key',
                options=list(key_dict.keys()),
                value=key_dict[list(key_dict.keys())[-1]],
                labelCheckedStyle={'color': 'red'},
                labelStyle={'display': 'block', 'whiteSpace': 'nowrap'},# 設置為 inline-block 並防止換行
                inline=True,
            ),
            dcc.Upload(
                id='upload-data',
                children=html.Button('上傳 csv 檔案', style={
                    'backgroundColor': '#ADD8E6',  # 淺藍色背景
                    'color': 'black',  # 黑色文字
                    'border': 'none',
                    'padding': '6px 10px', #按鈕內部的邊距設置
                    'fontSize': '16px',
                    'cursor': 'pointer',
                    'borderRadius': '8px',  # 圓角
                    #'boxShadow': '2px 2px 5px rgba(0, 0, 0, 0.2)',  # 立體效果
                    'transition': 'box-shadow 0.8s',  # 增加小動畫
                    'whiteSpace': 'nowrap'  # 防止按鈕文字換行
                }),
                multiple=False  # 僅允許一次上傳一個檔案
            ), # 使用 dcc.Loading 組件顯示加載動畫或訊息
            dcc.Loading(
                id="loading",
                type="default",  # 或者使用 "circle" 或 "dot" 來更改樣式
                children=html.Div(id='output', style={'marginTop': '10px'}),
                fullscreen=False  # 設置為 True 可使 loading 全螢幕顯示
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1px'}),  # 使用 flex 使它們水平排列，並設置間距),
"""
def Dash_key_heatmap(data_path,data,keywords_dict,title,dashname):
    key_dict = {'總體經濟面關鍵字': '總體經濟面關鍵字',
                 '產業面關鍵字': '產業面關鍵字',
                 '不分類關鍵字': '不分類關鍵字'}
    # --------
    app = DjangoDash(dashname, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.css.append_css({'external_url': '/static/css/freestyle.css'})

    app.layout = dbc.Container([
        html.Br(),
        html.Br(),
        html.P('選擇想呈現哪一個【關鍵字】字典之熱力圖結果:'),
        html.Div([
            dbc.RadioItems(  # 欲選擇的
                id='key',
                options=[{'label': key, 'value': key} for key in key_dict.keys()],
                value=key_dict[list(key_dict.keys())[0]],
                labelCheckedStyle={'color': 'red'},
                labelStyle={'display': 'block', 'whiteSpace': 'nowrap'},# 設置為 inline-block 並防止換行
                inline=True,
            ),
            dcc.Loading(  # 將 dcc.Loading 放在按鈕外層，並包含 output
                id="loading",
                type="default",
                children=html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('上傳 csv 檔案', style={
                            'backgroundColor': '#ADD8E6',
                            'color': 'black',
                            'border': 'none',
                            'padding': '6px 10px',
                            'fontSize': '16px',
                            'cursor': 'pointer',
                            'borderRadius': '8px',
                            'transition': 'box-shadow 0.8s',
                            'whiteSpace': 'nowrap'  # 防止按鈕文字換行
                        }),
                        multiple=False
                    ),
                    html.Div(id='output', style={'marginTop': '10px'})  # 保留 output 區域
                ]),
                fullscreen=False
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1px'}),  # 使用 flex 使它們水平排列，並設置間距),
        html.Div(id='pandas-output-container-1', style={"paddingTop": 10, 'color': '#178CA4'}),
        dbc.Alert(id='pandas-output-container-1', style={"paddingTop": 10}, color="primary"),
        html.Br(),
        ])
    
    @app.callback(
        Output('key', 'options'),
        Input('upload-data', 'contents'),
        State('key', 'options')
    )
    def update_checklist_options(contents, current_options):
        # 若有檔案上傳，將 "自行設計的關鍵字" 選項加入列表
        custom_option = '自行設計的關鍵字'
        if contents is not None and custom_option not in current_options:
            current_options += [custom_option]  # 使用正確格式添加選項

        return current_options
    
    @app.callback(
        Output('output', 'children'),
        Input('upload-data', 'contents')
    )
    def upload_and_set_global_df(csv_contents):
        global keywords_dict_3,key_dict_3,data_heatmap  # 聲明使用全域變數
        keywords_dict_3 = keywords_dict
        key_dict_3 = key_dict
        if csv_contents:
            start_time = time.time()  # 開始時間
            try:
                content_type, content_string = csv_contents.split(',')
                decoded = base64.b64decode(content_string)
                keywords_list0 = pd.read_csv(io.BytesIO(decoded), header=None).squeeze().tolist()
            except:
                return " !! 選取檔案格式非 csv, 請重新選取 !!"
            key = '自行設計的關鍵字'
            keywords_dict_3[key] = keywords_list0
            key_dict_3[key] = key
            print('@ 自行設計的關鍵字:     ', keywords_dict_3[key])
            
            def calculate_keywords_tf(ws):
                # 建立字詞到索引的映射字典
                word_to_index = {word: i for i, word in enumerate(keywords_list)}
                keywords_indices = torch.tensor([word_to_index[word] for word in keywords_list], device='cuda')
                
                 # 將 ws 中的詞轉換為索引，如果詞不存在於字典中則忽略
                ws_indices = [word_to_index[word] for word in ws if word in word_to_index]
                
                # 將索引轉換為 GPU 張量
                ws_tensor = torch.tensor(ws_indices, device='cuda')
                
                # 使用 GPU 計算關鍵字出現的次數
                keywords_tf = torch.sum(torch.isin(ws_tensor, keywords_indices)).cpu().item()
                # 釋放 GPU 記憶體
                torch.cuda.empty_cache()
                return float(keywords_tf)
            
            print('# 計算關鍵字出現次數  ==================================================================')
            data_n = data
            for keyword in keywords_list0:
                keywords_list = [keyword]
                data[keyword+'_tf'] = data['ws'].apply(calculate_keywords_tf)

            data_heatmap = data.drop(columns=['新聞發布時間', '新聞標題', '超連結', '新聞內文', 'ws', 'pos'])
             
            for keyword in keywords_list0:
                data_heatmap[keyword+'_是否出現在該篇新聞'] = np.where(data_heatmap[keyword+'_tf']>0 , 1,0)
            


            
            
            # 計算分鐘和秒
            minutes = int(processing_time // 60)
            seconds = int(processing_time % 60)
            return f"(成功上傳並運算完成,耗時{minutes}分{seconds}秒)"
        return "(無上傳檔案)"
    return app
    
def Dash_Chart2(data_path,data,stopwords,merged_df,keywords_dict,title,dashname):



    si_dict = {'[dict1] Bian, S., Jia, D., Li, F., & Yan, Z. (2021). A new Chinese financial sentiment dictionary for textual analysis in accounting and finance.':'dict1',
                    '[dict2] Du, Z., Huang, A. G., Wermers, R., & Wu, W. (2022). Language and domain specificity: A Chinese financial sentiment dictionary.':'dict2',
                    '[intersection] The intersection of D1 and D2':'intersection',
                    '[union] Union of D1 and D2':'union'}

    key_dict = {'不使用關鍵字': '不使用關鍵字',
                '總體經濟面關鍵字': '總體經濟面關鍵字',
                 '產業面關鍵字': '產業面關鍵字',
                 '不分類關鍵字': '不分類關鍵字'}

    method_dict = {'方法1':'',
                    '方法2':' _with',
                    '方法3':' d_by ws_n',
                    '方法4':'_sentences_m',
                    '方法5':['_sentences_m','d_by ws_n']}




    # --------
    app = DjangoDash(dashname, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.css.append_css({'external_url': '/static/css/freestyle.css'})

    app.layout = dbc.Container([
        html.Br(),
        html.Br(),
        #html.Center(html.H4(title, className="f02 fw-light")),
        html.Br(),
        html.P('1. 選擇想呈現哪些【情緒】字典之結果:'), # 【情緒】
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="si_all",
                    n_clicks=0,
                )
            ), style={'display': 'inline-table'}),  # 呈現並排
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="si_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        dbc.Checklist(  # 欲選擇的
            id='si',
            options=list(si_dict.keys()),
            value=list(si_dict.keys())[3],
            labelCheckedStyle={'color': 'red'},
            labelStyle={'display': 'block'},
            inline=True,
        ),
        html.Br(),
        html.P('2. 選擇想呈現哪些【關鍵字】字典之結果:'),# 【關鍵字】
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="key_all",
                    n_clicks=0,
                )
            ), style={'display': 'inline-table'}),  # 呈現並排
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="key_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        html.Div([
            dbc.Checklist(  # 欲選擇的
                id='key',
                options=list(key_dict.keys()),
                value=key_dict[list(key_dict.keys())[-1]],
                labelCheckedStyle={'color': 'red'},
                labelStyle={'display': 'block', 'whiteSpace': 'nowrap'},# 設置為 inline-block 並防止換行
                inline=True,
            ),
            dcc.Loading(  # 將 dcc.Loading 放在按鈕外層，並包含 output
                id="loading",
                type="default",
                children=html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('上傳 csv 檔案', style={
                            'backgroundColor': '#ADD8E6',
                            'color': 'black',
                            'border': 'none',
                            'padding': '6px 10px',
                            'fontSize': '16px',
                            'cursor': 'pointer',
                            'borderRadius': '8px',
                            'transition': 'box-shadow 0.8s',
                            'whiteSpace': 'nowrap'  # 防止按鈕文字換行
                        }),
                        multiple=False
                    ),
                    html.Div(id='output', style={'marginTop': '10px'})  # 保留 output 區域
                ]),
                fullscreen=False
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '1px'}),  # 使用 flex 使它們水平排列，並設置間距),
        html.Div(id='pandas-output-container-1', style={"paddingTop": 10, 'color': '#178CA4'}),
        dbc.Alert(id='pandas-output-container-1', style={"paddingTop": 10}, color="primary"),
        html.Br(),
        html.P('3. 選擇想呈現哪些【方法】之結果:'),# 【方法】
        html.Div([
            html.Div(html.P(
                html.A(
                    html.Button('全選', className='Button2'),
                    id="m_all",
                    n_clicks=0,
                )), style={'display': 'inline-table'}),
            html.Div(html.P(
                html.A(
                    html.Button('重置', className='Button2'),
                    id="m_notall",
                    n_clicks=0,
                ))
                , style={'display': 'inline-table'})]),
        dbc.Checklist(  # 欲選擇的
            id='m',
            options=list(method_dict.keys()),
            value=list(method_dict.keys())[-1],
            labelCheckedStyle={'color': 'red'},
            labelStyle={'display': 'block', 'margin': '0'},
            # inline=True
            style={
                'display': 'grid',
                'grid-template-columns': 'repeat(5, minmax(0, 100px))', #5col 指定每列的最小寬度為 0，最大寬度為 80px
                #'column-gap': '5px', #進一步減少了柱之間的空間

            },
        ),
        html.Br(),
        html.Div(
            [
                dbc.Button(
                    "打開隱藏資料",
                    id="collapse-button",
                    className="mb-3 Button1",
                    color="primary",
                    n_clicks=0,
                    # 可設定預設值 is_open=True顯示內容和is_open=False隱藏內容
                ),
                dbc.Collapse(
                    dash_table.DataTable(

                        id='table-editing-simple',
                        editable=False,  # 是否可以編輯表格
                        filter_action="native",
                        export_format='xlsx',
                        export_headers='display',
                        merge_duplicate_headers=True,
                        page_action='native',
                        page_current=0,
                        page_size=22,
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
                    id="collapse",
                    is_open=False,
                ),
            ]
        ),
        html.Br(),
        html.Div([
            html.P('動態調整圖形大小:'),
            html.Div([
                html.P('長度 '),
                dcc.Slider(id='SliderHeight', min=600, max=2200, step=25, value=750,
                           marks={x: str(x) for x in [600, 800, 1000, 1200, 1400, 1600, 1800, 2000,2200]}),
            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
            html.Div([
                html.P('寬度 '),
                dcc.Slider(id='SliderWidth', min=1000, max=2600, step=25, value=1900,
                           marks={x: str(x) for x in [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600]}), #numbers = list(range(1000, 2601, 200))
            ], style={"display": "grid", "grid-template-columns": "5% 90%"}),
        ]),
        html.Br(),
        html.P(
            html.A(
                html.Button("按一下將圖形載為 HTML 檔", className='Button1'),
                id="download",
                style={"text-align": "right"}), style={"text-align": "right"}),
        dcc.Download(id='download_1'),
        html.P("( 圖形右上方隱藏選單內有 PNG 檔載點與重置圖形紐 )",
               style={"text-align": "right", "font-weight": "bold"}),  #
        html.Div([
            dcc.Loading(dcc.Graph(id="graph",
                                  config={'scrollZoom': False,  # 'scrollZoom': True 滑鼠滾輪縮放開起
                                          'displaylogo': False,
                                          'toImageButtonOptions': {'filename': "圖表_「{}」".format(title)}}),
                        type="graph"),  # 改預設"cube" 為 "graph",
        ], style={'height': '1100px', 'width': '1500px'}),

    ], style={"margin-right": "2%", "margin-left": "2%"})

    # 回調函數：上傳檔案並更新全域變數
    @app.callback(
        Output('output', 'children'),
        Input('upload-data', 'contents')
    )
    def upload_and_set_global_df(csv_contents):
        global keywords_dict_2,key_dict_2,daily_mean_scores_with_keywords_renamed  # 聲明使用全域變數
        keywords_dict_2 = keywords_dict
        key_dict_2 = key_dict
        if csv_contents:
            start_time = time.time()  # 開始時間
            try:
                content_type, content_string = csv_contents.split(',')
                decoded = base64.b64decode(content_string)
                keywords_list = pd.read_csv(io.BytesIO(decoded), header=None).squeeze().tolist()
            except:
                return " !! 選取檔案格式非 csv, 請重新選取 !!"
            key = '自行設計的關鍵字'
            keywords_dict_2[key] = keywords_list
            key_dict_2[key] = key
            print('@ 自行設計的關鍵字:     ', keywords_dict_2[key])
            # ------------------------------
            # 檢查是否有可用的 GPU
            def get_device():
                if torch.cuda.is_available():
                    print("## CUDA is available. Using GPU.")
                    return torch.device('cuda')
                else:
                    print("## CUDA is not available. Using CPU.")
                    return torch.device('cpu')

            # 彈性化平行處理 DataFrame 的功能
            def parallelize_dataframe(df, func, col, n_cores=8, batch_size=len(data)):  # n_cores 處理器核心數量除以2
                device = get_device()  # 決定使用 GPU 或 CPU
                if device.type == 'cuda':
                    return parallelize_dataframe_on_gpu(df, func, col, batch_size, device)
                else:
                    return parallelize_dataframe_on_cpu(df, func, col, n_cores)

            # CPU 平行運算
            def parallelize_dataframe_on_cpu(df, func, col, n_cores=8):
                df_split = np.array_split(df, n_cores)  # 將資料分成幾個小的部分，這樣每個部分可以被不同的核心獨立處理
                pool = Pool(n_cores)
                df = pd.concat(pool.map(wrapper_process_dataframe, [(d, func, col) for d in df_split]))
                pool.close()
                pool.join()
                return df

            # GPU 批次處理
            def parallelize_dataframe_on_gpu(df, func, col, batch_size=len(data), device='cuda'):
                df_split = np.array_split(df, np.ceil(len(df) / batch_size).astype(int))
                results = []
                for d in df_split:
                    if d[col].dtype == 'object':
                        d[col] = np.nan  # Replace text data with np.nan

                    d_gpu = torch.tensor(d[col].values, device=device)  # 將資料轉換到 GPU
                    result = process_dataframe_on_gpu(d, func, col, d_gpu, device)
                    results.append(result)

                df = pd.concat(results)
                return df

            # CPU 子處理 DataFrame 的包裝函數
            def wrapper_process_dataframe(args):
                df, func, col = args
                return process_dataframe(df, func, col)

            # 對每個子 DataFrame 應用函數 (CPU)
            def process_dataframe(df, func, col):
                df[col] = df.apply(func, axis=1)
                return df

            # 對每個子 DataFrame 應用函數 (GPU)
            def process_dataframe_on_gpu(df, func, col, data_on_gpu, device):
                # GPU 上進行批次處理的邏輯
                df[col] = df.apply(func, axis=1)  # 可以根據需求進一步優化
                return df

            # 算法 2 （只看提及到關鍵字的新聞,然後一樣該新聞斷詞中有出現對應於字典的詞就計算,正向一詞紀錄加一分,負向詞一詞紀錄減一分)
            """
            def calculate_keywords_tf(news):  # 判斷提及關鍵字次數 # news=data_n.iloc[0]
                tokens = pd.Series(news['ws'])
                if torch.cuda.is_available():
                    keywords_tf = torch.tensor(tokens.isin(keywords_list).sum(), device='cuda').cpu().item()
                else:
                    keywords_tf = tokens.isin(keywords_list).sum()
                return keywords_tf
            
            """
            def calculate_keywords_tf(ws):
                # 建立字詞到索引的映射字典
                word_to_index = {word: i for i, word in enumerate(keywords_list)}
                keywords_indices = torch.tensor([word_to_index[word] for word in keywords_list], device='cuda')
                
                 # 將 ws 中的詞轉換為索引，如果詞不存在於字典中則忽略
                ws_indices = [word_to_index[word] for word in ws if word in word_to_index]
                
                # 將索引轉換為 GPU 張量
                ws_tensor = torch.tensor(ws_indices, device='cuda')
                
                # 使用 GPU 計算關鍵字出現的次數
                keywords_tf = torch.sum(torch.isin(ws_tensor, keywords_indices)).cpu().item()
                # 釋放 GPU 記憶體
                torch.cuda.empty_cache()
                return float(keywords_tf)
            
            # 算法 3 （每篇的正、負面詞彙分數加總並除以詞彙數)
            def calculate_word(news):  # 判斷新聞詞彙數量
                tokens = pd.Series(news['ws'])
                selected_words = []
                filtered_words = [
                    word.replace(" ", "") for word in tokens  # 刪除單詞中的空格
                    # 僅考慮長度大於1的單詞
                    # 並排除包含數字的詞（只要參雜就都會被過濾掉）
                    # 也排除停用詞庫內的詞
                    if len(word) > 1 and not re.search(r'\d', word) and word not in stopwords
                    # 其他參考：保留詞性為名詞或動詞(pos.startswith('N') or pos.startswith('V'))
                ]
                selected_words.extend(filtered_words)
                ws_n = len(selected_words)
                return ws_n

            # 算法 4 「得到含關鍵字的句子斷詞」--> 分數
            # 算法 4-1 得到每篇文章篩選過之句子斷詞的函數
            """
            def split_ws_and_filter(news, delimiters=['。', '!', '；', '?']):  # news=data_n.iloc[0]
                total_words_to_keep = 20
                tokens = pd.Series(news['ws'])
                result = []  # 擺該篇文章篩選過之句子斷詞

                current_sentence = []  # 擺當前句子
                for word in tokens:  # 迴圈每筆資料的斷詞
                    current_sentence.append(word)  # 填斷詞進入當前句子
                    if word in delimiters:  # 遇到句子結尾
                        if any(keyword in current_sentence for keyword in keywords_list):  # 檢查目前句子是否包含關鍵字
                            filtered_words = [
                                word.replace(" ", "") for word in current_sentence  # 刪除單詞中的空格
                                # 僅考慮長度大於1的單詞
                                # 並排除包含數字的詞（只要參雜就都會被過濾掉）
                                # 也排除停用詞庫內的詞
                                if len(word) > 1 and not re.search(r'\d', word) and word not in stopwords
                                # 其他參考：保留詞性為名詞或動詞(pos.startswith('N') or pos.startswith('V'))
                            ]
                            # 斷詞被重新整理了,所以關鍵字位置要必須在這裡才能獲取
                            keyword_indices = [i for i, w in enumerate(filtered_words) if w in keywords_list]

                            if len(filtered_words) > total_words_to_keep:  # 如果句子的長度超過n
                                trimmed_sentence = []  # 用於儲存最終的斷詞列表

                                # 遍歷關鍵字的位置，確保前後總共加起來n個斷詞
                                for i, keyword_index in enumerate(keyword_indices):
                                    # 計算保留的範圍，關鍵字前後確定要保留的總共 total_words_to_keep 個斷詞，前後的加總
                                    start = max(0, keyword_index - (total_words_to_keep // 2))  # 儘量從前面取一半的詞
                                    words_taken_before = keyword_index - start  # 前面取了幾個
                                    words_to_take_after = total_words_to_keep - words_taken_before  # 後面可以取的詞數
                                    end = min(len(current_sentence), keyword_index + words_to_take_after)  # 後面取足夠的詞

                                    # 如果這不是第一個關鍵字，檢查是否有重疊部分避免重複添加
                                    if i > 0:
                                        # 只添加前一個結束到當前開始之間的部分
                                        trimmed_sentence.extend(
                                            filtered_words[keyword_indices[i - 1]:start])  # len(trimmed_sentence)

                                    # 添加當前關鍵字前後的斷詞
                                    trimmed_sentence.extend(filtered_words[start:end])

                                result.append(trimmed_sentence)
                            else:
                                result.append(filtered_words)  # 如果斷詞不超過total_words_to_keep，直接加入結果
                        current_sentence = []  # 重置當前句子

                return result
            """
            def batch_process_split_ws_and_filter(data_n,col, delimiters=['。', '!', '；', '?']):
                results = []  # 用於存儲所有批次的結果

                # 按照 batch_size 將資料分批次處理
                for start_idx in range(0, len(data_n), batch_size):
                    # 取得當前批次的範圍
                    end_idx = min(start_idx + batch_size, len(data_n))
                    batch_data = data_n[start_idx:end_idx]

                    batch_results = []  # 用於存儲當前批次的結果

                    # 對每一個 row 的 `ws` 欄位進行處理
                    for ws in batch_data['ws']:
                        # 檢查 ws 是否為列表型態，並使用 CPU 進行字串處理
                        if isinstance(ws, list):
                            result = split_ws_and_filter(ws, delimiters)  # 在 CPU 上處理字串
                            batch_results.append(result)
                        else:
                            batch_results.append([])  # 若 ws 欄位不是列表型態，則回傳空結果

                    # 將當前批次的結果添加到總結果中
                    results.extend(batch_results)

                # 將結果加回 DataFrame 的新欄位 'col'
                data_n[col] = results
                # 釋放 GPU 記憶體
                torch.cuda.empty_cache()
                return data_n
            # 修改後的split_ws_and_filter函式，不再嘗試轉換字串為tensor
            def split_ws_and_filter(ws, delimiters=['。', '!', '；', '?']):
                result = []  # 儲存過濾後的句子
                current_sentence = []  # 暫存當前句子
                total_words_to_keep = 20
                for word in ws:  # 對每個詞進行處理
                    current_sentence.append(word)  # 加入到當前句子
                    if word in delimiters:  # 如果詞為句子分隔符號
                        if any(keyword in current_sentence for keyword in keywords_list):  # 檢查關鍵字
                            # 過濾符合條件的詞
                            filtered_words = [
                                word.replace(" ", "") for word in current_sentence
                                if len(word) > 1 and not re.search(r'\d', word) and word not in stopwords
                            ]
                            
                            # 找出關鍵字索引位置
                            keyword_indices = [i for i, w in enumerate(filtered_words) if w in keywords_list]
                            
                            # 如果句子長度超過指定的 `total_words_to_keep`
                            if len(filtered_words) > total_words_to_keep:
                                trimmed_sentence = []  # 用於存儲篩選後的斷詞句子
                                
                                # 確保前後各取 `total_words_to_keep` 個斷詞
                                for i, keyword_index in enumerate(keyword_indices):
                                    start = max(0, keyword_index - (total_words_to_keep // 2))
                                    words_taken_before = keyword_index - start
                                    words_to_take_after = total_words_to_keep - words_taken_before
                                    end = min(len(current_sentence), keyword_index + words_to_take_after)
                                    
                                    if i > 0:
                                        trimmed_sentence.extend(filtered_words[keyword_indices[i-1]:start])
                                    
                                    trimmed_sentence.extend(filtered_words[start:end])
                                
                                result.append(trimmed_sentence)
                            else:
                                result.append(filtered_words)
                        current_sentence = []  # 清空當前句子

                return result


            # 算法 4-2 計算每篇新聞含關鍵字的句子斷詞之情感分數的函數
            """
            def calculate_sentiment_score_sentences(news):  # news=data_n.iloc[0]
                sentences_wsdf = pd.DataFrame(news['sentences_m' + key])
                if torch.cuda.is_available():
                    positive_score = torch.tensor(sentences_wsdf.isin(positive_word_list).sum().sum(),
                                                  device='cuda').cpu().item()
                    negative_score = torch.tensor(sentences_wsdf.isin(negative_word_list).sum().sum(),
                                                  device='cuda').cpu().item()
                else:
                    positive_score = sentences_wsdf.isin(positive_word_list).sum().sum()
                    negative_score = sentences_wsdf.isin(negative_word_list).sum().sum()
                return positive_score - negative_score
            """
            # 修改 calculate_sentiment_score 函數來處理嵌套列表
            def calculate_sentiment_score(ws):
                # 展開嵌套列表，並將詞彙轉換為對應的索引
                ws_indices = [word_to_index[word] for sublist in ws for word in sublist if word in word_to_index]
                tokens = torch.tensor(ws_indices, device='cuda')

                # 使用 GPU 計算正面和負面詞彙的分數
                positive_score = torch.sum(torch.isin(tokens, positive_indices))
                negative_score = torch.sum(torch.isin(tokens, negative_indices))

                # 將結果轉回 CPU，並轉為 float
                result = float((positive_score - negative_score).cpu().item())
                
                # 釋放 GPU 記憶體
                torch.cuda.empty_cache()
                
                return result

            # 算法 5 基於算法4除上對應的斷詞數量 計算每篇新聞含關鍵字的句子斷詞之 數量 的函數
            # 5-1 計算每篇新聞含關鍵字的句子斷詞之 數量 的函數
            """
            def calculate_sentiment_ws_n(news):  # news=data_n.iloc[0]
                total_ws_count = sum(len(sentence) for sentence in news['sentences_m' + key])
                return total_ws_count
            """
            def batch_calculate_sentiment_ws_n(sentences_m_list):
                    # 計算每個句子的詞數，將結果轉換為 GPU 張量
                    lengths_list = [[len(sentence) for sentence in sentences] for sentences in sentences_m_list]
                    total_ws_count = torch.tensor([sum(lengths) for lengths in lengths_list], device='cuda').cpu().tolist()
                    
                    # 釋放 GPU 記憶體
                    torch.cuda.empty_cache()
                    return total_ws_count

            # ------------------------------


            list_cols = ['sentiment score_dict1', 'sentiment score_dict2', 'sentiment score_intersection',
                         'sentiment score_union']
            modified_list0 = [item + ' d_by ws_n' for item in list_cols]
            modified_list = [item + '_sentences_m' + key for item in list_cols]
            modified_list1 = [item + '_sentences_m' + key + ' d_by ws_n' for item in
                              list_cols]  # modified_list1 = [item + ' d_by ws_n' for item in modified_list]

            data_n = data
            print('# 算法 2  (1次gpu運算)  ==================================================================')
            data_n[key + '_tf'] = data_n['ws'].apply(calculate_keywords_tf)# 算法2
            print('# 算法 3   ==================================================================')
            for col in list_cols:
                data_n[f'{col} d_by ws_n'] = data_n[col] / data['ws_n']  # 算法3
            print('# 算法 4  (5次gpu運算) ==================================================================')
            batch_size = len(data)  # 可依據 GPU 記憶體大小調整
            data_n['sentences_m' + key] = None
            data_n = batch_process_split_ws_and_filter(data_n,col='sentences_m' + key)  # 算法4-1 得到出現 keyword 的斷詞句子
            # Sentiment Dictionary
            file_dict1_path = data_path + '/字典/' + "A New Chinese Financial Sentiment Dictionary for Textual Analysis in Accounting and Finance_Full_Traditional.xlsx"
            file_dict2_path = data_path + '/字典/' + "ChineseSentimentDictionary_Traditional.xlsx"
            intersection_file_path = data_path + '/字典/' + "Sentiment_Dictionary_Intersection.xlsx"
            union_file_path = data_path + '/字典/' + "Sentiment_Dictionary_Union.xlsx"
            for col, file_path in zip(modified_list,
                                      [file_dict1_path, file_dict2_path, intersection_file_path, union_file_path]):
                positive_words = pd.read_excel(file_path, sheet_name='Positive', names=['Positive'])
                negative_words = pd.read_excel(file_path, sheet_name='Negative', names=['Negative'])
                positive_word_list = positive_words['Positive'].tolist()
                negative_word_list = negative_words['Negative'].tolist()
                # 建立字詞到索引的映射字典
                word_to_index = {word: i for i, word in enumerate(set(positive_word_list + negative_word_list))}
                positive_indices = torch.tensor([word_to_index[word] for word in positive_word_list if word in word_to_index], device='cuda')
                negative_indices = torch.tensor([word_to_index[word] for word in negative_word_list if word in word_to_index], device='cuda')
                data_n[col] = data_n['sentences_m' + key].apply(calculate_sentiment_score)# 算法4-2 計算每篇新聞含關鍵字的句子斷詞之情感分數的函數
                
            print('# 算法 5   (1次gpu運算) ==================================================================')
            # data_n['sentences_m' + key + ' ws_n'] = None
            # data_n = parallelize_dataframe(data_n, calculate_sentiment_ws_n, col='sentences_m' + key + ' ws_n')
            # 將批量處理函數應用到整個 DataFrame
            data_n['sentences_m' + key + ' ws_n'] = batch_calculate_sentiment_ws_n(data_n['sentences_m' + key].tolist())

            for col in modified_list:
                data_n[f'{col} d_by ws_n'] = data_n[col] / data_n['sentences_m' + key + ' ws_n']  # 算法5

            data_n["新聞發布日期"] = pd.to_datetime(data_n["新聞發布時間"]).dt.date
            cut_col = "新聞發布日期"
            daily_mean_scores_with_keywords = data_n[data_n[key + '_tf'] > 0].groupby(cut_col).agg(
                每日新聞數量=(cut_col, 'size'),
                **{col: (col, 'mean') for col in list_cols + modified_list0 + modified_list + modified_list1}
            )


            daily_mean_scores_with_keywords_renamed = daily_mean_scores_with_keywords.rename(
                columns={col: col + ' _with ' + key for col in list(daily_mean_scores_with_keywords.columns) if key not in col}
            )
       
            end_time = time.time()  # 結束時間
            processing_time = end_time - start_time  # 計算耗時

            # 計算分鐘和秒
            minutes = int(processing_time // 60)
            seconds = int(processing_time % 60)
            return f"(成功上傳並運算完所有涉及到之情感分數,耗時{minutes}分{seconds}秒)"
        return "(無上傳檔案)"

    @app.callback(
        Output('key', 'options'),
        Input('upload-data', 'contents'),
        State('key', 'options')
    )
    def update_checklist_options(contents, current_options):
        # 若有檔案上傳，將 "自行設計的關鍵字" 選項加入列表
        custom_option = '自行設計的關鍵字'
        if contents is not None and custom_option not in current_options:
            current_options += [custom_option]  # 使用正確格式添加選項

        return current_options



    @app.callback(
        Output("si", "value"),

        Input("si_all", "n_clicks"),
        Input("si_notall", "n_clicks"),
        State('si', 'options'),
        State("si_all", "n_clicks_timestamp"),
        State("si_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0
        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]
        if last_n_id == 'n':
            # 如果選中了“全選” 獲取所有選項的值
            selected_keys = list(si_dict.keys())
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            selected_keys = list(si_dict.keys())[-1]
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys

    @app.callback(
        Output("key", "value"),

        Input("key_all", "n_clicks"),
        Input("key_notall", "n_clicks"),
        State('key', 'options'),
        State("key_all", "n_clicks_timestamp"),
        State("key_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0

        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]
        if last_n_id == 'n':
            # 如果選中了“全選” 獲取所有選項的值
            all_values = [option for option in options]
            selected_keys = all_values
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            #selected_keys = list(key_dict.keys())[-1]
            selected_keys = ['不分類關鍵字']
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys


    @app.callback(
        Output("m", "value"),
        Input("m_all", "n_clicks"),
        Input("m_notall", "n_clicks"),
        State('m', 'options'),
        State("m_all", "n_clicks_timestamp"),
        State("m_notall", "n_clicks_timestamp"),
        prevent_initial_call=True)  # prevent_initial_call=True 使不被預設先按了按鈕
    def update_dropdowns(n, n2, options, timestamp1, timestamp2):
        if timestamp1 is None:
            timestamp1 = 0
        if timestamp2 is None:
            timestamp2 = 0

        last_n_id = max([(timestamp1, 'n'), (timestamp2, 'n2')])[1]

        if last_n_id == 'n':
            # 如果選中了“全選” 獲取所有選項的值
            all_values = [option for option in options]
            selected_keys = all_values
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        elif last_n_id == "n2":
            selected_keys = list(method_dict.keys())[-1]
            #values =  [si_dict[key] for key in selected_keys]  # 預設
        return selected_keys

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
    Output("graph", "figure"),

        Output('table-editing-simple', 'data'),
        Output('table-editing-simple', 'columns'),
        Output('table-editing-simple', 'style_data_conditional'),
        Output('pandas-output-container-1', 'children'),
        Input("si", "value"),
        Input('key', "value"),
        Input('m', "value"),
        Input('SliderHeight', 'value'),
        Input('SliderWidth', 'value'),

    )
    def display_barcahrt(si, keys, m, height, width):

        if 'daily_mean_scores_with_keywords_renamed'  in globals():

            # 合併兩個 DataFrame
            merged_df_new = pd.merge(merged_df, daily_mean_scores_with_keywords_renamed,
                                         how='outer', left_index=True, right_index=True)

            keywords_dict_new = keywords_dict_2
            key_dict_new = key_dict_2
        else:
            merged_df_new = merged_df
            keywords_dict_new = keywords_dict
            key_dict_new = key_dict
        merged_df_new.columns = [item.replace("sentiment score_", "") for item in list(merged_df_new.columns)]  # 修改所有欄位名稱
        try:
            print('＠3＠',merged_df_new[['每日新聞數量 _with 自行設計的關鍵字',
       'dict1 _with 自行設計的關鍵字', 'dict2 _with 自行設計的關鍵字',
       'intersection _with 自行設計的關鍵字', 'union _with 自行設計的關鍵字',
       'dict1 d_by ws_n _with 自行設計的關鍵字', 'dict2 d_by ws_n _with 自行設計的關鍵字',
       'intersection d_by ws_n _with 自行設計的關鍵字',
       'union d_by ws_n _with 自行設計的關鍵字', 'dict1_sentences_m自行設計的關鍵字',
       'dict2_sentences_m自行設計的關鍵字', 'intersection_sentences_m自行設計的關鍵字',
       'union_sentences_m自行設計的關鍵字', 'dict1_sentences_m自行設計的關鍵字 d_by ws_n',
       'dict2_sentences_m自行設計的關鍵字 d_by ws_n',
       'intersection_sentences_m自行設計的關鍵字 d_by ws_n',
       'union_sentences_m自行設計的關鍵字 d_by ws_n']])
        except:
            print('*********** 尚無 自行設計的關鍵字   ***********')
        # print('*** 選取 si, keys, m :', si, keys, m)
        # 檢查原始資料，並以防範措施確保其不會被意外拆解
        def clean_input(input_value):
            if input_value is None:
                return []  # 若為 None，轉為空列表
            elif isinstance(input_value, list):
                # 若為列表，確認每個元素為字串，不會拆成單字元
                return [item for item in input_value if len(item) > 1] #[item for item in m if len(item) > 1]
            elif isinstance(input_value, str):
                # 若為單一字串，包裝成單元素列表，避免被拆解成單字元
                return [input_value]
            else:
                # 若有其他情況，強制轉換為列表
                return [str(input_value)]

        # 使用 clean_input 函數來處理每個變數
        si = clean_input(si)
        keys = clean_input(keys)
        m = clean_input(m)
        print('*** 選取 si, keys, m :', si, keys, m)

        si_select =  [si_dict[key] for key in si]
        key_select =  [key_dict[key] for key in keys]
            
        """
            method_dict = {'方法1':'方法1',
                    '方法2':' _with',
                    '方法3':' d_by ws_n',
                    '方法4':'_sentences_m',
                    '方法5':['_sentences_m','d_by ws_n']}
        """
        filtered_columns = []
        # 預先過濾 `key_select`，去除不需要的關鍵字
        filtered_keys = [key for key in key_select if key != '不使用關鍵字']

        if '方法1' in m and '不使用關鍵字' in key_select:
                 filtered_columns.extend(si_select) #透過 extend() 和生成器表達式，減少內存開銷
        if '方法2' in m:
            filtered_columns.extend(f"{col} _with {key}" for col in si_select for key in filtered_keys)

        if '方法3' in m:
            if '不使用關鍵字' in key_select:
                filtered_columns.extend(f"{col} d_by ws_n" for col in si_select)
            filtered_columns.extend(f"{col} d_by ws_n _with {key}" for col in si_select for key in filtered_keys)

        if '方法4' in m:
            m4_list_tem = [item for item in merged_df_new.columns if '_sentences_m' in item and 'd_by ws_n' not in item]
            filtered_columns.extend(col for col in m4_list_tem if
                                    any(si in col for si in si_select) and any(key in col for key in filtered_keys))

        if '方法5' in m:
            m5_list_tem = [item for item in merged_df_new.columns if '_sentences_m' in item and 'd_by ws_n' in item]
            filtered_columns.extend(col for col in m5_list_tem if
                                    any(si in col for si in si_select) and any(key in col for key in filtered_keys))


        print('*** filtered_columns ：',filtered_columns)
        bar_columns = []
        if '方法1' in m and '不使用關鍵字' in key_select:
            bar_columns.append('每日新聞數量')
        for key in key_select:
            key_list0 = [col for col in list(merged_df_new.columns) if key in col and '每日新聞數量'  in col]
            bar_columns += key_list0  # 不使用關鍵字



        fig = plotly_line_chart(
                     merged_df_new,
                     x_column='新聞發布日期',
                     y_columns=filtered_columns,
                     title='每日情感分數變化',
                     x_title='',
                     y_title='分數',
                     #output_file=path+'/每日情感分數變化圖/sentiment_scores_all_withbar.html',
                     #colors=colors,
                     bar_columns=bar_columns,
                     bar_title='新聞數量',
                     range_slider=True,
                     subplots=[0.8,0.2],
                     #bar_colors=['dodgerblue','darkorange'],
                    height=height,
                    width=width
                   )
        fig.write_html("sentiment_scores_all_withbar.html")
       # fig.write_html("{}_「{}」.html".format(chart, title))
        # ----

        data_t = merged_df_new[bar_columns+filtered_columns].copy()
        data_t.reset_index(inplace=True)# 將 index 轉為第一欄位
        data_t['新聞發布日期'] = data_t['新聞發布日期'].dt.strftime("%Y-%m-%d")  # object

        data_col_name = data_t.columns.values.tolist()
        table_columns = [{'id': col, 'name': col, 'deletable': True, 'renamable': True} for col in
                         data_col_name]

        table_data = [dict(Model=i, **{col: data_t.iloc[i][col] for col in data_col_name})
                      for i in range(len(data_t))]


        # 創建一個過濾條件，排除指定的欄位
        columns_to_keep = [col for col in data_t.columns if col not in ['新聞發布日期']]
        df_modified = data_t[columns_to_keep]
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
                    } for (col, value) in df_modified.quantile(0.9).items()
                ] +
                [
                    {
                        'if': {
                            'filter_query': '{{{}}} <= {}'.format(col, value),
                            'column_id': col
                        },
                        'backgroundColor': 'rgb(218,112,214)',
                        'color': 'white'
                    } for (col, value) in df_modified.quantile(0.1).items()
                ]
        )


        # 生成 dmc.Text 的項目
        dmc_text = [dmc.Text("資料說明：")]
        dmc_text.append(dmc.Text(""))
        for i, key in enumerate(key_select):
            result = ', '.join(keywords_dict_new.get(key, []))
            dmc_text.append(dmc.Space(h=8))
            dmc_text.append(dmc.Text(f"【{key}】")) # 【】
            dmc_text.append(dmc.Text(f"{result}"))
            if i < len(key_select) - 1:  # 在每個項目之間插入，但不在最後一個後面插入
                dmc_text.append(dmc.Space(h=8))
                dmc_text.append(dmc.Divider(style={"border-top": "1px dashed #000"}))


        return (fig, table_data, table_columns, style_data_conditional,dmc_text)

    @app.callback(
        Output('download_1', 'data'),
        Input('download', 'n_clicks'), prevent_initial_call=True)
    def download_html(n):
        return dcc.send_file("{}_「{}」.html".format(chart, title))

    return app
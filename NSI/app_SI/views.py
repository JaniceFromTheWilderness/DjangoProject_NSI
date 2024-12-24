from django.shortcuts import render
from .funs import *
from dash import Dash, dcc, html, Input, Output,State, ctx, callback,dash_table
#import dash_core_components as dcc
from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from django.conf import settings
# Create your views here.


# 找到 static 資料夾中的檔案路徑
data_path = os.path.join(settings.BASE_DIR, 'app_SI/static')
print('data_path:',data_path)
#已存檔之最新資料檔名： ====================================================================================

for file in file_list(data_path+'/data'):
        if  'words_result_新聞內文等完整資訊_與斷詞與詞性_無排除重複值與過濾停用詞_資料時間' in file and file.endswith('.json'):
            if 'file_name1' not in globals():
                file_name1 = file
                print(file_name1)
        if  '新聞內文與新聞情感分數_資料時間' in file and file.endswith('.json'):
            if 'file_name2' not in globals():
                file_name2 = file
                print(file_name2)
        if '新聞情感分數 daily_mix_資料時間' in file and file.endswith('.csv'):
            if 'file_name3' not in globals():
                file_name3 = file
                print(file_name3)
        if 'file_name1' in globals() and 'file_name2' in globals() and 'file_name3' in globals():
            break

stopwords = load_stopwords(os.path.join(data_path+'/data/字典/', 'stopwords.txt'))
# 讀檔
data_dict = json_file({},'read',data_path+'/data',file_name1)
data = pd.DataFrame(data_dict)# 將字典轉換為 DataFrame
data_dict = json_file({}, 'read', data_path+'/data', file_name2)
data2 = pd.DataFrame(data_dict)
data = pd.merge(data, data2.drop(columns=['新聞發布時間']), on='新聞內文', how='left')
del data_dict
data["新聞發布日期"] = pd.to_datetime(data["新聞發布時間"]).dt.date

merged_df=pd.read_csv( data_path+f'/data/{file_name3}', index_col='新聞發布日期', parse_dates=['新聞發布日期'])

keywords_dict = {}
for file in file_list(data_path+'/字典/'):
    if 'Keywords' in file and file.endswith('.csv'):
        filename = data_path+'/字典/' + file
        if file != 'Keywords.csv':
            name = file.split('_')[1].split('.csv')[0] + '關鍵字'
            keywords_dict[name] = pd.read_csv(filename, header=None).squeeze().tolist()
        else:
            name = '不分類關鍵字'
            keywords_dict[name] = pd.read_csv(filename, header=None).squeeze().tolist()



def sentiment(request):
    text_title='新聞情緒指標'
    app_si = Dash_Chart(data,merged_df,keywords_dict,text_title,'app_si')

    return render(request, 'SI_web.html', locals())

def sentiment_key(request): 
    text_title='新聞情緒指標'
    app_si = Dash_Chart2(data_path,data,stopwords,merged_df,keywords_dict,text_title,'app_si')

    return render(request, 'SI_web.html', locals())

def key_heatmap(request):
    text_title='新聞關鍵字字典熱力圖'
    app_key_heatmap = Dash_key_heatmap(data_path,data,keywords_dict,text_title,'app_key_heatmap')

    return render(request, 'key_heatmap_web.html', locals())
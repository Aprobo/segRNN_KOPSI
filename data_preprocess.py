import pandas as pd
import glob
import os

# 1) CSV 파일 목록 가져오기
file_list = glob.glob("./data/raw/*.csv")   # 현재 폴더의 모든 CSV

# 2) CSV 파일 모두 읽어서 리스트에 저장
df_list = []
for file in file_list:
    df = pd.read_csv(file, thousands=',')
    df_list.append(df)

# 3) 데이터프레임 모두 concat
data = pd.concat(df_list, ignore_index=True)

data.rename(columns={"날짜": "Date",
                     "종가":"Price",
                     "거래량":"Volume",
                     "변동 %":"Fluct"}, inplace=True)

# 거래량 변환
def convert_volume(v):
    if isinstance(v, str):
        v = v.replace(",", "").strip()

        if v.endswith("M"):
            return float(v[:-1]) * 1e6
        elif v.endswith("K"):
            return float(v[:-1]) * 1e3
        elif v.endswith("B"):
            return float(v[:-1]) * 1e9
        else:
            return float(v)
    return v

# 변동률 변환
def convert_percent(p):
    if isinstance(p, str):
        p = p.replace("%", "").replace(",", "").strip()
        return float(p) / 100.0
    return p

data["Volume"] = data["Volume"].apply(convert_volume)
data['Fluct'] = data['Fluct'].apply(convert_percent)

# 4) 날짜 컬럼 datetime 변환
data['Date'] = pd.to_datetime(data['Date'])


# 5) 날짜 기준으로 정렬 (필수)
data = data.sort_values('Date').reset_index(drop=True)

# 6) 원하는 열만 남긴다
cols_to_use = ["Date", "Price", "Volume","Fluct"]
pv = data[cols_to_use]

p = pd.read_csv("./data/kospi_p.csv")

p['Date'] = pd.to_datetime(p['Date'])
pv['Date'] = pd.to_datetime(pv['Date'])

dates_to_remove = pv[~pv['Date'].isin(p['Date'])]['Date']

print("삭제할 날짜 목록:")
print(dates_to_remove)

# 4) 해당 날짜 행 전체 삭제
pv_clean = pv[~pv['Date'].isin(dates_to_remove)]

# 5) 삭제 후 결과 확인
print("삭제 전 pv 길이:", len(pv))
print("삭제 후 pv 길이:", len(pv_clean))
print("p 길이:", len(p))

pv.set_index('Date', inplace=True)

# 6) 저장
csv_path = "./data/kospi_pvf.csv"

if not os.path.exists(csv_path):
    with open(csv_path, 'w', encoding='utf-8'):
        pv_clean.to_csv("./data/kospi_pvf.csv", index=False)
else:
    print("csv file exis")

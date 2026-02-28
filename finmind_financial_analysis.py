"""
AI 台股財報分析系統 (FinMind)
基於 Streamlit 框架的台股財務分析應用程式
使用 FinMind API 獲取財務數據，並透過 OpenAI 進行 AI 分析
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from openai import OpenAI

# ============================================================
# 頁面基本配置
# ============================================================
st.set_page_config(
    page_title="AI 台股財報分析系統",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 工具函數模組
# ============================================================

def format_large_number(value):
    """將大數字格式化為易讀的中文單位格式"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        value = float(value)
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1e12:
            return f"{sign}{abs_value/1e12:.2f}兆"
        elif abs_value >= 1e8:
            return f"{sign}{abs_value/1e8:.2f}億"
        elif abs_value >= 1e6:
            return f"{sign}{abs_value/1e6:.2f}百萬"
        else:
            return f"{sign}{abs_value:,.0f}"
    except (TypeError, ValueError):
        return "N/A"


def format_percentage(value, decimals=2):
    """格式化百分比顯示"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    try:
        return f"{float(value)*100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "N/A"


def validate_stock_code(code):
    """驗證台股代碼是否為四位數字格式"""
    if not code:
        return False, "請輸入股票代碼"
    code = code.strip()
    if not code.isdigit():
        return False, f"股票代碼必須為數字，您輸入了：{code}（範例：2330、2454、2317、2412）"
    if len(code) != 4:
        return False, f"台股代碼必須為四位數字，您輸入了 {len(code)} 位（範例：2330、2454）"
    return True, "格式正確"


def safe_divide(numerator, denominator, default=0.0):
    """安全除法，避免除以零錯誤"""
    try:
        if denominator == 0 or denominator is None:
            return default
        result = float(numerator) / float(denominator)
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (TypeError, ValueError, ZeroDivisionError):
        return default


# ============================================================
# FinMind API 整合模組
# ============================================================

# FinMind API 統一端點
FINMIND_API_URL = "https://api.finmindtrade.com/api/v4/data"

# 損益表欄位對應（FinMind type → 內部標準名稱）
INCOME_STATEMENT_MAPPING = {
    "Revenue": "revenues",
    "GrossProfit": "grossprofit",
    "OperatingIncome": "operatingincomeloss",
    "IncomeAfterTaxes": "netincomeloss",
    "PreTaxIncome": "incomelossfromcontinuingoperationsbeforeincometaxes",
    "EPS": "eps_basic",
    "TotalNonoperatingIncomeAndExpense": "total_nonoperating",
}

# 資產負債表欄位對應
BALANCE_SHEET_MAPPING = {
    "TotalAssets": "assets",
    "Liabilities": "liabilities",
    "Equity": "stockholdersequity",
    "CurrentAssets": "assetscurrent",
    "CurrentLiabilities": "liabilitiescurrent",
    "RetainedEarnings": "retainedearningsaccumulateddeficit",
    "NoncurrentLiabilities": "longtermdebtnoncurrent",
}

# 現金流量表欄位對應
CASHFLOW_MAPPING = {
    "CashFlowsFromOperatingActivities": "netcashprovidedbyusedinoperatingactivities",
    "CashProvidedByInvestingActivities": "netcashprovidedbyusedininvestingactivities",
    "CashFlowsProvidedFromFinancingActivities": "netcashprovidedbyusedinfinancingactivities",
    "PropertyAndPlantAndEquipment": "paymentstoacquireproductiveassets",
}


def fetch_finmind_data(dataset, stock_id, start_date, token):
    """
    從 FinMind API 獲取指定 dataset 的數據
    
    Parameters:
        dataset: FinMind dataset 名稱
        stock_id: 股票代碼
        start_date: 資料起始日期
        token: FinMind API Token
    
    Returns:
        DataFrame 或 None（發生錯誤時）
    """
    try:
        params = {
            "dataset": dataset,
            "data_id": stock_id,
            "start_date": start_date,
            "token": token,
        }
        response = requests.get(FINMIND_API_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != 200:
            msg = data.get("msg", "未知錯誤")
            st.warning(f"FinMind API 警告（{dataset}）：{msg}")
            return None
        
        records = data.get("data", [])
        if not records:
            return None
        
        return pd.DataFrame(records)
    
    except requests.exceptions.ConnectionError:
        st.error("無法連接 FinMind API，請確認網路連線後重試。")
        return None
    except requests.exceptions.Timeout:
        st.error("FinMind API 請求逾時，請稍後重試。")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"FinMind API HTTP 錯誤：{e}")
        return None
    except Exception as e:
        st.error(f"FinMind API 發生未知錯誤（{dataset}）：{e}")
        return None


def standardize_financial_statement(df, mapping, date_col="date"):
    """
    將 FinMind API 回傳的 type 欄位值轉換為程式內部標準欄位名稱
    
    Parameters:
        df: FinMind API 回傳的 DataFrame
        mapping: 欄位對應字典
        date_col: 日期欄位名稱
    
    Returns:
        以日期為索引、各財務指標為欄位的 DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    try:
        # 取出需要的欄位：日期、type、value
        if not all(col in df.columns for col in [date_col, "type", "value"]):
            return pd.DataFrame()
        
        # 篩選對應的 type 欄位
        filtered = df[df["type"].isin(mapping.keys())].copy()
        if filtered.empty:
            return pd.DataFrame()
        
        # 轉換 type 名稱為內部標準名稱
        filtered["internal_key"] = filtered["type"].map(mapping)
        
        # Pivot：日期為索引，各指標為欄位
        pivot = filtered.pivot_table(
            index=date_col,
            columns="internal_key",
            values="value",
            aggfunc="first"
        )
        pivot.index = pd.to_datetime(pivot.index)
        pivot = pivot.sort_index(ascending=False)
        
        # 確保數值欄位為 float
        for col in pivot.columns:
            pivot[col] = pd.to_numeric(pivot[col], errors="coerce")
        
        return pivot
    
    except Exception as e:
        st.warning(f"欄位標準化處理發生錯誤：{e}")
        return pd.DataFrame()


def fetch_all_financial_data(stock_id, start_date, token):
    """
    從 FinMind API 獲取所有財務報表數據並整合
    
    Returns:
        dict 包含各報表標準化後的 DataFrame 及公司基本資訊
    """
    result = {
        "income_statement": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "cash_flow": pd.DataFrame(),
        "stock_price": pd.DataFrame(),
        "company_info": {},
    }
    
    progress = st.progress(0, text="正在獲取損益表數據...")
    
    # 1. 損益表
    income_raw = fetch_finmind_data(
        "TaiwanStockFinancialStatements", stock_id, start_date, token
    )
    result["income_statement"] = standardize_financial_statement(
        income_raw, INCOME_STATEMENT_MAPPING
    )
    progress.progress(20, text="正在獲取資產負債表數據...")
    
    # 2. 資產負債表
    balance_raw = fetch_finmind_data(
        "TaiwanStockBalanceSheet", stock_id, start_date, token
    )
    result["balance_sheet"] = standardize_financial_statement(
        balance_raw, BALANCE_SHEET_MAPPING
    )
    progress.progress(40, text="正在獲取現金流量表數據...")
    
    # 3. 現金流量表
    cashflow_raw = fetch_finmind_data(
        "TaiwanStockCashFlowsStatement", stock_id, start_date, token
    )
    result["cash_flow"] = standardize_financial_statement(
        cashflow_raw, CASHFLOW_MAPPING
    )
    progress.progress(60, text="正在獲取股價數據...")
    
    # 4. 股價
    price_raw = fetch_finmind_data(
        "TaiwanStockPrice", stock_id, start_date, token
    )
    if price_raw is not None and not price_raw.empty:
        result["stock_price"] = price_raw
    progress.progress(80, text="正在獲取公司基本資料...")
    
    # 5. 公司基本資料
    info_raw = fetch_finmind_data(
        "TaiwanStockInfo", stock_id, "2010-01-01", token
    )
    if info_raw is not None and not info_raw.empty:
        latest = info_raw[info_raw["stock_id"] == stock_id].iloc[0] if "stock_id" in info_raw.columns else info_raw.iloc[0]
        result["company_info"] = latest.to_dict()
    
    progress.progress(100, text="數據獲取完成！")
    progress.empty()
    
    return result


def compute_derived_fields(income_df, balance_df, cash_flow_df, price_df, company_info):
    """
    計算缺失欄位補償機制：
    - 加權平均股數 = 淨利潤 ÷ EPS
    - 利息費用推估
    - 現金流量表資本支出取絕對值
    """
    # 加權平均股數計算
    if not income_df.empty:
        if "netincomeloss" in income_df.columns and "eps_basic" in income_df.columns:
            mask = (income_df["eps_basic"] != 0) & (~income_df["eps_basic"].isna())
            income_df["weightedaveragenumberofsharesoutstandingbasic"] = np.nan
            income_df.loc[mask, "weightedaveragenumberofsharesoutstandingbasic"] = (
                income_df.loc[mask, "netincomeloss"] / income_df.loc[mask, "eps_basic"]
            ) * 1000  # EPS 通常以元計，轉換為股數（千股）
        
        # 利息費用推估
        if "total_nonoperating" in income_df.columns:
            income_df["interestexpensenonoperating"] = income_df["total_nonoperating"].apply(
                lambda x: abs(x) if (not pd.isna(x) and x < 0) else 0
            )
    
    # 現金流量表：資本支出取絕對值
    if not cash_flow_df.empty and "paymentstoacquireproductiveassets" in cash_flow_df.columns:
        cash_flow_df["paymentstoacquireproductiveassets"] = (
            cash_flow_df["paymentstoacquireproductiveassets"].abs()
        )
    
    # 市值計算
    market_cap = None
    latest_price = None
    shares_outstanding = None
    
    if price_df is not None and not price_df.empty and "close" in price_df.columns:
        price_df["date"] = pd.to_datetime(price_df["date"])
        latest_price_row = price_df.sort_values("date", ascending=False).iloc[0]
        latest_price = float(latest_price_row["close"]) if "close" in latest_price_row else None
    
    if company_info and "stock_id" in company_info:
        shares_str = company_info.get("shares", None)
        if shares_str:
            try:
                shares_outstanding = float(str(shares_str).replace(",", ""))
            except (ValueError, TypeError):
                pass
    
    if latest_price and shares_outstanding:
        market_cap = latest_price * shares_outstanding * 1000  # 千股轉換
    
    return income_df, balance_df, cash_flow_df, market_cap, latest_price, shares_outstanding


def merge_financial_data(income_df, balance_df, cash_flow_df):
    """
    將三個財務報表按日期合併為統一格式的數據列表
    
    Returns:
        list of dict，每個元素代表一個財報年度的完整數據
    """
    if income_df.empty and balance_df.empty and cash_flow_df.empty:
        return []
    
    # 取得所有出現的日期
    all_dates = set()
    for df in [income_df, balance_df, cash_flow_df]:
        if not df.empty:
            all_dates.update(df.index.tolist())
    
    if not all_dates:
        return []
    
    all_dates = sorted(all_dates, reverse=True)
    merged_data = []
    
    for d in all_dates:
        record = {"date": d}
        
        # 損益表欄位
        if not income_df.empty and d in income_df.index:
            for col in income_df.columns:
                record[col] = income_df.loc[d, col]
        
        # 資產負債表欄位
        if not balance_df.empty and d in balance_df.index:
            for col in balance_df.columns:
                record[col] = balance_df.loc[d, col]
        
        # 現金流量表欄位
        if not cash_flow_df.empty and d in cash_flow_df.index:
            for col in cash_flow_df.columns:
                record[col] = cash_flow_df.loc[d, col]
        
        merged_data.append(record)
    
    return merged_data


# ============================================================
# 數據驗證模組
# ============================================================

def validate_financial_data(financial_data):
    """
    驗證財務數據完整性和合理性
    
    Returns:
        (is_valid, warnings, errors) 三元組
    """
    warnings_list = []
    errors_list = []
    
    if not financial_data:
        errors_list.append("無法獲取任何財務數據，請確認股票代碼和 API Token 是否正確。")
        return False, warnings_list, errors_list
    
    if len(financial_data) < 2:
        warnings_list.append("財務數據少於 2 年，部分年度比較分析將無法進行。")
    
    # 必要欄位檢查
    required_fields = ["netincomeloss", "assets", "revenues", "stockholdersequity"]
    latest = financial_data[0]
    
    missing_required = [f for f in required_fields if latest.get(f) is None or (isinstance(latest.get(f), float) and np.isnan(latest.get(f)))]
    if missing_required:
        field_names = {
            "netincomeloss": "淨利潤",
            "assets": "總資產",
            "revenues": "營收",
            "stockholdersequity": "股東權益",
        }
        missing_names = [field_names.get(f, f) for f in missing_required]
        errors_list.append(f"缺少關鍵財務指標：{', '.join(missing_names)}，分析結果可能不準確。")
    
    # 合理性檢查
    assets = latest.get("assets")
    if assets is not None and not np.isnan(float(assets if assets else 0)) and float(assets if assets else 0) <= 0:
        warnings_list.append("總資產數值異常（≤0），請確認數據來源。")
    
    revenues = latest.get("revenues")
    if revenues is not None and not np.isnan(float(revenues if revenues else 0)) and float(revenues if revenues else 0) < 0:
        warnings_list.append("營收出現負值，可能為特殊情況，請注意分析結果的解讀。")
    
    return len(errors_list) == 0, warnings_list, errors_list


def generate_data_quality_report(financial_data, income_df, balance_df, cash_flow_df):
    """
    生成財務數據品質報告
    
    Returns:
        dict 包含品質等級、年份統計、缺失欄位等資訊
    """
    report = {
        "quality_level": "良好",
        "years_count": len(financial_data),
        "missing_fields": [],
        "computed_fields": [],
        "limitations": [],
    }
    
    if not financial_data:
        report["quality_level"] = "嚴重不足"
        return report
    
    # 重要欄位清單
    important_fields = {
        "revenues": "營收",
        "grossprofit": "毛利",
        "operatingincomeloss": "營業利潤",
        "netincomeloss": "淨利潤",
        "assets": "總資產",
        "liabilities": "總負債",
        "stockholdersequity": "股東權益",
        "assetscurrent": "流動資產",
        "liabilitiescurrent": "流動負債",
        "netcashprovidedbyusedinoperatingactivities": "營運現金流",
        "paymentstoacquireproductiveassets": "資本支出",
    }
    
    latest = financial_data[0]
    missing_count = 0
    
    for field, name in important_fields.items():
        val = latest.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            report["missing_fields"].append(f"{name}（{field}）")
            missing_count += 1
    
    # 計算欄位說明
    report["computed_fields"] = [
        "加權平均股數：由「淨利潤 ÷ EPS」計算而得，可能因 EPS 精度產生誤差",
        "利息費用：由「營業外收入及支出」推估，負值取絕對值，可能低估實際利息費用",
        "市值：由「最新收盤價 × 公司發行股數（千股）」計算，僅供參考",
    ]
    
    # 品質等級評估
    total_fields = len(important_fields)
    if missing_count == 0:
        report["quality_level"] = "良好"
    elif missing_count <= total_fields * 0.3:
        report["quality_level"] = "部分缺失"
    else:
        report["quality_level"] = "嚴重不足"
    
    if report["years_count"] < 2:
        report["quality_level"] = "部分缺失"
        report["limitations"].append("財務數據年份不足 2 年，無法進行年度比較分析")
    
    return report


# ============================================================
# 財務計算模組
# ============================================================

def calculate_piotroski_fscore(financial_data):
    """
    計算 Piotroski F-Score（9項指標，各 1 分）
    
    使用最新年度 vs 前一年度進行比較
    """
    if len(financial_data) < 2:
        return None
    
    curr = financial_data[0]  # 最新年度
    prev = financial_data[1]  # 前一年度
    
    def get_val(record, key, default=0.0):
        v = record.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    
    results = {
        "total_score": 0,
        "profitability_scores": [],
        "leverage_scores": [],
        "efficiency_scores": [],
    }
    
    # ---- 獲利能力指標（4項）----
    
    # 1. ROA 正值
    curr_net = get_val(curr, "netincomeloss")
    curr_assets = get_val(curr, "assets", 1)
    curr_roa = safe_divide(curr_net, curr_assets)
    score1 = 1 if curr_roa > 0 else 0
    results["profitability_scores"].append({
        "description": "ROA 正值（淨利潤 / 總資產 > 0）",
        "current_value": f"{curr_roa*100:.2f}%",
        "previous_value": "-",
        "score": score1,
        "passed": score1 == 1,
    })
    
    # 2. 營運現金流正值
    curr_ocf = get_val(curr, "netcashprovidedbyusedinoperatingactivities")
    score2 = 1 if curr_ocf > 0 else 0
    results["profitability_scores"].append({
        "description": "營運現金流 > 0",
        "current_value": format_large_number(curr_ocf),
        "previous_value": "-",
        "score": score2,
        "passed": score2 == 1,
    })
    
    # 3. ROA 年增率
    prev_net = get_val(prev, "netincomeloss")
    prev_assets = get_val(prev, "assets", 1)
    prev_roa = safe_divide(prev_net, prev_assets)
    score3 = 1 if curr_roa > prev_roa else 0
    results["profitability_scores"].append({
        "description": "ROA 年增（最新年度 ROA > 前一年度）",
        "current_value": f"{curr_roa*100:.2f}%",
        "previous_value": f"{prev_roa*100:.2f}%",
        "score": score3,
        "passed": score3 == 1,
    })
    
    # 4. 營運現金流品質（OCF > 淨利潤）
    score4 = 1 if curr_ocf > curr_net else 0
    results["profitability_scores"].append({
        "description": "現金流品質（營運現金流 > 淨利潤）",
        "current_value": f"OCF={format_large_number(curr_ocf)}, NI={format_large_number(curr_net)}",
        "previous_value": "-",
        "score": score4,
        "passed": score4 == 1,
    })
    
    # ---- 槓桿與流動性指標（3項）----
    
    # 5. 長期負債比率改善
    curr_ltd = get_val(curr, "longtermdebtnoncurrent")
    curr_assets_v = get_val(curr, "assets", 1)
    prev_ltd = get_val(prev, "longtermdebtnoncurrent")
    prev_assets_v = get_val(prev, "assets", 1)
    curr_ltd_ratio = safe_divide(curr_ltd, curr_assets_v)
    prev_ltd_ratio = safe_divide(prev_ltd, prev_assets_v)
    score5 = 1 if curr_ltd_ratio < prev_ltd_ratio else 0
    results["leverage_scores"].append({
        "description": "長期負債比率改善（最新 < 前期）",
        "current_value": f"{curr_ltd_ratio*100:.2f}%",
        "previous_value": f"{prev_ltd_ratio*100:.2f}%",
        "score": score5,
        "passed": score5 == 1,
    })
    
    # 6. 流動比率改善
    curr_ca = get_val(curr, "assetscurrent", 1)
    curr_cl = get_val(curr, "liabilitiescurrent", 1)
    prev_ca = get_val(prev, "assetscurrent", 1)
    prev_cl = get_val(prev, "liabilitiescurrent", 1)
    curr_current_ratio = safe_divide(curr_ca, curr_cl)
    prev_current_ratio = safe_divide(prev_ca, prev_cl)
    score6 = 1 if curr_current_ratio > prev_current_ratio else 0
    results["leverage_scores"].append({
        "description": "流動比率改善（最新 > 前期）",
        "current_value": f"{curr_current_ratio:.2f}",
        "previous_value": f"{prev_current_ratio:.2f}",
        "score": score6,
        "passed": score6 == 1,
    })
    
    # 7. 股份未稀釋
    curr_shares = get_val(curr, "weightedaveragenumberofsharesoutstandingbasic")
    prev_shares = get_val(prev, "weightedaveragenumberofsharesoutstandingbasic")
    score7 = 1 if (curr_shares <= prev_shares and curr_shares > 0 and prev_shares > 0) else 0
    results["leverage_scores"].append({
        "description": "股份未稀釋（流通股數未增加）",
        "current_value": format_large_number(curr_shares),
        "previous_value": format_large_number(prev_shares),
        "score": score7,
        "passed": score7 == 1,
    })
    
    # ---- 營運效率指標（2項）----
    
    # 8. 毛利率改善
    curr_gp = get_val(curr, "grossprofit")
    curr_rev = get_val(curr, "revenues", 1)
    prev_gp = get_val(prev, "grossprofit")
    prev_rev = get_val(prev, "revenues", 1)
    curr_gpm = safe_divide(curr_gp, curr_rev)
    prev_gpm = safe_divide(prev_gp, prev_rev)
    score8 = 1 if curr_gpm > prev_gpm else 0
    results["efficiency_scores"].append({
        "description": "毛利率改善（最新 > 前期）",
        "current_value": f"{curr_gpm*100:.2f}%",
        "previous_value": f"{prev_gpm*100:.2f}%",
        "score": score8,
        "passed": score8 == 1,
    })
    
    # 9. 資產周轉率改善
    curr_ato = safe_divide(curr_rev, curr_assets_v)
    prev_ato = safe_divide(prev_rev, prev_assets_v)
    score9 = 1 if curr_ato > prev_ato else 0
    results["efficiency_scores"].append({
        "description": "資產周轉率改善（最新 > 前期）",
        "current_value": f"{curr_ato:.3f}",
        "previous_value": f"{prev_ato:.3f}",
        "score": score9,
        "passed": score9 == 1,
    })
    
    results["total_score"] = score1 + score2 + score3 + score4 + score5 + score6 + score7 + score8 + score9
    return results


def calculate_dupont_analysis(financial_data, max_years=3):
    """
    計算杜邦分析（ROE 三因子分解）
    
    Returns:
        list of dict，每個元素為一個年度的杜邦分析結果
    """
    results = []
    data_slice = financial_data[:max_years]
    
    for i, record in enumerate(data_slice):
        def get_val(key, default=0.0):
            v = record.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)
        
        net_income = get_val("netincomeloss")
        revenues = get_val("revenues", 1)
        assets = get_val("assets", 1)
        equity = get_val("stockholdersequity", 1)
        
        # 三因子計算
        net_margin = safe_divide(net_income, revenues)        # 淨利率
        asset_turnover = safe_divide(revenues, assets)        # 資產周轉率
        equity_multiplier = safe_divide(assets, equity)       # 權益乘數
        
        # ROE：三因子乘積
        roe_dupont = net_margin * asset_turnover * equity_multiplier
        # 直接計算 ROE
        roe_direct = safe_divide(net_income, equity)
        
        entry = {
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "net_margin": net_margin,
            "asset_turnover": asset_turnover,
            "equity_multiplier": equity_multiplier,
            "roe_dupont": roe_dupont,
            "roe_direct": roe_direct,
        }
        
        # 計算與前一年度的變化
        if results:
            prev_entry = results[-1]
            entry["net_margin_change"] = net_margin - prev_entry["net_margin"]
            entry["asset_turnover_change"] = asset_turnover - prev_entry["asset_turnover"]
            entry["equity_multiplier_change"] = equity_multiplier - prev_entry["equity_multiplier"]
            entry["roe_change"] = roe_dupont - prev_entry["roe_dupont"]
        else:
            entry["net_margin_change"] = None
            entry["asset_turnover_change"] = None
            entry["equity_multiplier_change"] = None
            entry["roe_change"] = None
        
        results.append(entry)
    
    return results


def calculate_cashflow_analysis(financial_data, max_years=5):
    """
    計算現金流分析指標
    
    Returns:
        list of dict，每個元素為一個年度的現金流分析結果
    """
    results = []
    data_slice = financial_data[:max_years]
    
    for record in data_slice:
        def get_val(key, default=0.0):
            v = record.get(key)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)
        
        ocf = get_val("netcashprovidedbyusedinoperatingactivities")
        icf = get_val("netcashprovidedbyusedininvestingactivities")
        fcf_financing = get_val("netcashprovidedbyusedinfinancingactivities")
        net_income = get_val("netincomeloss", 1)
        capex = abs(get_val("paymentstoacquireproductiveassets"))  # 資本支出使用絕對值
        
        # 現金流品質比率
        ocf_quality = safe_divide(ocf, net_income) if net_income != 0 else 0
        
        # 自由現金流：營運現金流 - 資本支出（資本支出已是絕對值）
        free_cash_flow = ocf - capex
        
        # 品質評估
        if ocf_quality >= 1.2:
            quality_rating = "優秀 🌟"
        elif ocf_quality >= 1.0:
            quality_rating = "良好 ✅"
        elif ocf_quality >= 0.8:
            quality_rating = "尚可 ⚠️"
        else:
            quality_rating = "需關注 🔴"
        
        results.append({
            "date": record["date"].strftime("%Y-%m-%d") if hasattr(record["date"], "strftime") else str(record["date"]),
            "operating_cash_flow": ocf,
            "investing_cash_flow": icf,
            "financing_cash_flow": fcf_financing,
            "net_income": net_income,
            "capex": capex,
            "free_cash_flow": free_cash_flow,
            "ocf_quality_ratio": ocf_quality,
            "quality_rating": quality_rating,
        })
    
    return results


# ============================================================
# 視覺化模組
# ============================================================

CHART_COLORS = {
    "dark_green": "#1B5E20",
    "dark_red": "#B71C1C",
    "steel_blue": "#1565C0",
    "gold": "#F57F17",
    "purple": "#4A148C",
    "light_green": "#4CAF50",
    "light_red": "#EF5350",
    "light_blue": "#42A5F5",
    "orange": "#FF8F00",
    "teal": "#00695C",
}


def create_bar_chart(x_data, y_data, title, x_label, y_label, color=None, height=400):
    """創建專業柱狀圖"""
    colors = [CHART_COLORS["steel_blue"] if v >= 0 else CHART_COLORS["dark_red"] for v in y_data]
    if color:
        colors = [color] * len(y_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_data,
        y=y_data,
        marker_color=colors,
        text=[format_large_number(v) for v in y_data],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(title=x_label, tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white",
        height=height,
        margin=dict(t=60, b=80, l=60, r=20),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def create_multi_bar_chart(dates, series_data, title, y_label, height=400):
    """
    創建多系列柱狀圖
    series_data: list of (name, values, color)
    """
    fig = go.Figure()
    for name, values, color in series_data:
        fig.add_trace(go.Bar(
            name=name,
            x=dates,
            y=values,
            marker_color=color,
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:,.0f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        barmode="group",
        xaxis=dict(title="日期", tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white",
        height=height,
        margin=dict(t=60, b=80, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_line_chart(x_data, y_series, title, y_label, height=400):
    """
    創建折線圖
    y_series: list of (name, values, color)
    """
    fig = go.Figure()
    for name, values, color in y_series:
        fig.add_trace(go.Scatter(
            x=x_data,
            y=values,
            mode="lines+markers",
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=8),
            hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        xaxis=dict(title="日期", tickangle=-45),
        yaxis=dict(title=y_label),
        template="plotly_white",
        height=height,
        margin=dict(t=60, b=80, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def create_pie_chart(labels, values, title, colors, height=350):
    """創建圓餅圖"""
    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} 項<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#1a1a2e")),
        template="plotly_white",
        height=height,
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def display_fscore_tables(fscore_result):
    """顯示 F-Score 各項指標表格"""
    
    def make_score_df(scores):
        rows = []
        for item in scores:
            rows.append({
                "指標說明": item["description"],
                "當前值": item["current_value"],
                "前期值": item["previous_value"],
                "得分": item["score"],
                "狀態": "✅" if item["passed"] else "❌",
            })
        return pd.DataFrame(rows)
    
    st.markdown("#### 🏆 獲利能力指標（4項）")
    st.dataframe(make_score_df(fscore_result["profitability_scores"]), use_container_width=True, hide_index=True)
    
    st.markdown("#### 🏦 槓桿與流動性指標（3項）")
    st.dataframe(make_score_df(fscore_result["leverage_scores"]), use_container_width=True, hide_index=True)
    
    st.markdown("#### ⚙️ 營運效率指標（2項）")
    st.dataframe(make_score_df(fscore_result["efficiency_scores"]), use_container_width=True, hide_index=True)


# ============================================================
# AI 分析模組
# ============================================================

def prepare_ai_analysis_data(financial_data, fscore_result, dupont_result, cashflow_result, stock_id, company_info, market_cap, latest_price):
    """整合三階段分析結果和財務數據，準備 AI 分析所需的提示語內容"""
    
    # 公司基本資訊
    company_name = company_info.get("stock_name", stock_id)
    industry = company_info.get("industry_category", "未知")
    
    # F-Score 摘要
    fscore_summary = ""
    if fscore_result:
        fscore_summary = f"""
【Piotroski F-Score】
總分：{fscore_result['total_score']} / 9 分

獲利能力指標（{sum(i['score'] for i in fscore_result['profitability_scores'])} / 4）：
"""
        for item in fscore_result["profitability_scores"]:
            status = "✅" if item["passed"] else "❌"
            fscore_summary += f"  {status} {item['description']}：{item['current_value']}（前期：{item['previous_value']}）\n"
        
        fscore_summary += f"\n槓桿與流動性指標（{sum(i['score'] for i in fscore_result['leverage_scores'])} / 3）：\n"
        for item in fscore_result["leverage_scores"]:
            status = "✅" if item["passed"] else "❌"
            fscore_summary += f"  {status} {item['description']}：{item['current_value']}（前期：{item['previous_value']}）\n"
        
        fscore_summary += f"\n營運效率指標（{sum(i['score'] for i in fscore_result['efficiency_scores'])} / 2）：\n"
        for item in fscore_result["efficiency_scores"]:
            status = "✅" if item["passed"] else "❌"
            fscore_summary += f"  {status} {item['description']}：{item['current_value']}（前期：{item['previous_value']}）\n"
    
    # 杜邦分析摘要
    dupont_summary = "\n【杜邦分析（最近3年）】\n"
    if dupont_result:
        for entry in dupont_result:
            dupont_summary += (
                f"  {entry['date']}：淨利率={entry['net_margin']*100:.2f}%，"
                f"資產周轉率={entry['asset_turnover']:.3f}，"
                f"權益乘數={entry['equity_multiplier']:.2f}，"
                f"ROE={entry['roe_dupont']*100:.2f}%\n"
            )
    
    # 現金流分析摘要
    cashflow_summary = "\n【現金流分析（最近3年）】\n"
    if cashflow_result:
        for entry in cashflow_result[:3]:
            cashflow_summary += (
                f"  {entry['date']}：OCF={format_large_number(entry['operating_cash_flow'])}，"
                f"FCF={format_large_number(entry['free_cash_flow'])}，"
                f"品質比率={entry['ocf_quality_ratio']:.2f}（{entry['quality_rating']}）\n"
            )
    
    # 最新財務數據
    latest = financial_data[0] if financial_data else {}
    
    def fv(key):
        v = latest.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return format_large_number(float(v))
    
    financial_summary = f"""
【最新年度財務數據（{latest.get('date', 'N/A')}）】
營收：{fv('revenues')}
毛利：{fv('grossprofit')}
營業利潤：{fv('operatingincomeloss')}
淨利潤：{fv('netincomeloss')}
總資產：{fv('assets')}
股東權益：{fv('stockholdersequity')}
市值（估算）：{format_large_number(market_cap) if market_cap else 'N/A'}
最新股價：{f'{latest_price:.2f} 元' if latest_price else 'N/A'}
產業別：{industry}
"""
    
    return {
        "company_name": company_name,
        "stock_id": stock_id,
        "industry": industry,
        "fscore_summary": fscore_summary,
        "dupont_summary": dupont_summary,
        "cashflow_summary": cashflow_summary,
        "financial_summary": financial_summary,
    }


def run_ai_analysis(openai_api_key, analysis_data, model="o4-mini"):
    """
    使用 OpenAI 新版 API 進行 AI 財務分析
    注意：必須使用 client.chat.completions.create 方法，不可使用舊版 ChatCompletion.create
    """
    try:
        # 使用 OpenAI 客戶端初始化方式（新版 API 格式）
        client = OpenAI(api_key=openai_api_key)
        
        system_message = """你是一位專精台股財務分析和台灣會計準則的資深分析師，
熟悉台灣上市公司財報規範（IFRS 台版）、FinMind 開源財務資料的特性和限制、
以及台股市場的投資環境（法規、產業政策、兩岸關係等影響因素）。
請用繁體中文提供客觀、專業且負責任的財務分析，避免過度承諾或產生誤導性內容。"""
        
        user_prompt = f"""
請根據以下已完成的三階段財務分析結果，對台股 {analysis_data['stock_id']}（{analysis_data['company_name']}）進行深度財務分析。

**請基於以下已計算完成的數據進行解讀，而非重新計算。**

{analysis_data['fscore_summary']}
{analysis_data['dupont_summary']}
{analysis_data['cashflow_summary']}
{analysis_data['financial_summary']}

---

請依以下結構提供完整分析報告：

## 一、三階段評分總結

請輸出以下表格（Markdown 格式）：

| 分析階段 | 評分狀態 | 評價 | 主要發現 |
|---------|---------|------|---------|
| Piotroski F-Score | ... | ... | ... |
| 杜邦分析 | ... | ... | ... |
| 現金流分析 | ... | ... | ... |

## 二、Piotroski F-Score 解讀
根據 F-Score 得分，解讀各項指標對投資判斷的意義和公司業務狀況。

## 三、杜邦分析趨勢洞察
分析 ROE 三因子（淨利率、資產周轉率、權益乘數）的趨勢變化，找出主要驅動力和財務效率變化。

## 四、現金流結構深度分析
分析現金流品質、自由現金流趨勢、資本支出模式和獲利品質一致性。

## 五、台股市場特性分析
分析該公司在台股市場的定位、競爭優勢，以及台灣法規、產業政策、兩岸關係等對投資的影響。

## 六、台灣會計準則與資料來源說明

### FinMind 資料特點與限制
說明 FinMind 開源資料的特性和限制。

### 計算欄位說明
- **加權平均股數**：由「淨利潤 ÷ EPS」計算（非直接申報數據），可能存在精度誤差
- **利息費用**：由「營業外收入及支出」推估，可能低估實際利息費用
- **市值**：由最新收盤價 × 發行股數估算，僅供參考

## 七、綜合財務健康診斷

### 主要優勢（3-5點）
列出關鍵財務優勢。

### 風險因素
列出需要關注的財務和市場風險。

### 後續追蹤重點
投資後需要監控的關鍵指標。

### 財報綜合評比

| 評估面向 | 評分 | 說明 |
|---------|------|------|
| 營運績效 | ... | ... |
| 財務結構 | ... | ... |
| 現金流量 | ... | ... |
| 總結 | ... | ... |

---
*本分析僅供教育和研究用途，不構成投資建議。投資決策請自行評估風險。*
"""
        
        # 使用新版 API 格式：client.chat.completions.create
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=4000,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            return f"⚠️ OpenAI API 金鑰錯誤：請確認您的 API 金鑰是否正確。\n\n技術錯誤：{error_msg}"
        elif "model" in error_msg.lower():
            return f"⚠️ 模型不可用：{model} 可能不在您的 API 授權範圍內。請嘗試使用 gpt-4o-mini。\n\n技術錯誤：{error_msg}"
        elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
            return f"⚠️ API 使用量超限或請求頻率過高，請稍後再試。\n\n技術錯誤：{error_msg}"
        else:
            return f"⚠️ AI 分析發生錯誤，請確認 API 金鑰和網路連線後重試。\n\n技術錯誤：{error_msg}"


# ============================================================
# 主程式
# ============================================================

def main():
    # ---- 頁面標題 ----
    st.title("📊 AI 台股財報分析系統")
    st.markdown(
        "<hr style='border: 2px solid #1a237e; margin: 0 0 1rem 0;'>",
        unsafe_allow_html=True
    )
    
    # ---- 側邊欄 ----
    with st.sidebar:
        st.markdown("## 📈 AI 財報分析")
        st.markdown("<hr style='border: 2px solid #1a237e;'>", unsafe_allow_html=True)
        
        stock_id = st.text_input(
            "🏷️ 股票代碼",
            placeholder="例：2330、2454、2317、2412",
            help="請輸入四位數字的台股代碼"
        )
        
        finmind_token = st.text_input(
            "🔑 FinMind API Token",
            type="password",
            help="請至 FinMind 官網申請免費 API Token：https://finmindtrade.com"
        )
        
        openai_key = st.text_input(
            "🤖 OpenAI API 金鑰",
            type="password",
            help="請至 OpenAI 官網申請 API 金鑰：https://platform.openai.com"
        )
        
        start_date = st.text_input(
            "📅 起始日期",
            value="2019-01-01",
            help="財務數據起始日期（格式：YYYY-MM-DD）"
        )
        
        analyze_btn = st.button("🔍 分析股票", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
**⚠️ 免責聲明**

本系統僅供**教育和研究用途**，分析結果不構成投資建議。投資有風險，請自行評估並承擔風險。

財務數據來源：[FinMind 開源平台](https://finmindtrade.com)
        """)
        
        st.markdown("---")
        st.markdown("""
**使用說明**
1. 輸入台股四位數代碼
2. 填入 FinMind API Token
3. 填入 OpenAI API 金鑰（AI分析功能需要）
4. 點擊「分析股票」開始分析
        """)
    
    # ---- 主要內容區 ----
    if not analyze_btn:
        # 首頁介紹
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("""
**📊 三大財務報表分析**
- 損益表趨勢分析
- 資產負債表結構分析
- 現金流量表品質分析
            """)
        with col2:
            st.info("""
**🎯 三階段專業分析**
- Piotroski F-Score（9項指標）
- 杜邦分析（ROE三因子）
- 現金流品質評估
            """)
        with col3:
            st.info("""
**🤖 AI 深度分析**
- 台股市場特性解讀
- 財務健康綜合診斷
- 投資風險評估報告
            """)
        
        st.markdown("""
### 如何開始？

請在左側側邊欄輸入：
1. **股票代碼**：例如 2330（台積電）、2454（聯發科）、2317（鴻海）、2412（中華電）
2. **FinMind API Token**：前往 [FinMind 官網](https://finmindtrade.com) 免費申請
3. **OpenAI API 金鑰**：前往 [OpenAI Platform](https://platform.openai.com) 申請（AI分析功能必填）
4. 點擊「分析股票」按鈕開始分析

---
        """)
        return
    
    # ---- 輸入驗證 ----
    if not finmind_token:
        st.error("❌ 請填入 FinMind API Token 才能獲取財務數據。")
        return
    
    valid, msg = validate_stock_code(stock_id)
    if not valid:
        st.error(f"❌ 股票代碼格式錯誤：{msg}")
        return
    
    # ---- 數據獲取 ----
    st.info(f"⏳ 正在獲取 {stock_id} 的財務數據，請稍候...")
    
    raw_data = fetch_all_financial_data(stock_id, start_date, finmind_token)
    
    income_df = raw_data["income_statement"]
    balance_df = raw_data["balance_sheet"]
    cashflow_df = raw_data["cash_flow"]
    price_df = raw_data["stock_price"]
    company_info = raw_data["company_info"]
    
    # 計算衍生欄位
    income_df, balance_df, cashflow_df, market_cap, latest_price, shares_outstanding = compute_derived_fields(
        income_df, balance_df, cashflow_df, price_df, company_info
    )
    
    # 合併財務數據
    financial_data = merge_financial_data(income_df, balance_df, cashflow_df)
    
    # 驗證數據品質
    is_valid, warnings_list, errors_list = validate_financial_data(financial_data)
    
    for err in errors_list:
        st.error(f"⚠️ {err}")
    for warn in warnings_list:
        st.warning(f"⚠️ {warn}")
    
    if not financial_data:
        st.error("❌ 無法獲取財務數據，請確認股票代碼和 API Token 是否正確，以及 FinMind 服務是否正常。")
        return
    
    # ---- 公司基本資訊 ----
    st.success(f"✅ 成功獲取 {stock_id} 的財務數據（共 {len(financial_data)} 個年度）")
    st.markdown("---")
    
    company_name = company_info.get("stock_name", stock_id)
    industry = company_info.get("industry_category", "未知")
    sector = company_info.get("type", "")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"🏢 {company_name}")
        st.write(f"**產業類別**：{industry}")
        if sector:
            st.write(f"**行業分類**：{sector}")
    
    with col2:
        if latest_price:
            st.metric(label="💹 最新收盤價", value=f"NT$ {latest_price:,.2f}")
        else:
            st.metric(label="💹 最新收盤價", value="N/A")
    
    with col3:
        mc_display = format_large_number(market_cap) if market_cap else "N/A"
        st.write(f"**市值（估算）**：NT$ {mc_display}")
        
        # 本益比計算
        latest = financial_data[0] if financial_data else {}
        net_income = latest.get("netincomeloss")
        if market_cap and net_income and not np.isnan(float(net_income)) and float(net_income) > 0:
            pe_ratio = market_cap / float(net_income)
            st.write(f"**本益比（P/E）**：{pe_ratio:.2f}x")
        else:
            st.write("**本益比（P/E）**：N/A")
    
    st.markdown("---")
    
    # ---- 財務計算 ----
    with st.spinner("🧮 正在進行三階段財務分析計算..."):
        fscore_result = calculate_piotroski_fscore(financial_data)
        dupont_result = calculate_dupont_analysis(financial_data)
        cashflow_result = calculate_cashflow_analysis(financial_data)
    
    # ---- 頁籤結構 ----
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 損益表分析",
        "🏦 資產負債表分析",
        "💰 現金流量表分析",
        "🎯 三階段財報分析",
        "🤖 AI 分析",
    ])
    
    # ============================================================
    # 頁籤 1：損益表分析
    # ============================================================
    with tab1:
        st.subheader("📈 損益表關鍵指標")
        
        if income_df.empty:
            st.warning("無法獲取損益表數據，請確認 API Token 和股票代碼。")
        else:
            # 準備圖表數據
            dates = [d.strftime("%Y-%m") for d in income_df.index]
            
            # 營收與毛利柱狀圖
            if "revenues" in income_df.columns and "grossprofit" in income_df.columns:
                fig = create_multi_bar_chart(
                    dates,
                    [
                        ("營收", income_df["revenues"].tolist(), CHART_COLORS["steel_blue"]),
                        ("毛利", income_df["grossprofit"].tolist(), CHART_COLORS["dark_green"]),
                    ],
                    "營收與毛利趨勢",
                    "金額（元）",
                )
                st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # 淨利潤趨勢
                if "netincomeloss" in income_df.columns:
                    fig2 = create_bar_chart(
                        dates,
                        income_df["netincomeloss"].tolist(),
                        "淨利潤趨勢",
                        "日期", "金額（元）",
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with col_b:
                # 毛利率趨勢
                if "revenues" in income_df.columns and "grossprofit" in income_df.columns:
                    gpm = [safe_divide(g, r) * 100 for g, r in zip(
                        income_df["grossprofit"].fillna(0),
                        income_df["revenues"].replace(0, np.nan).fillna(1)
                    )]
                    fig3 = create_bar_chart(
                        dates, gpm, "毛利率趨勢（%）", "日期", "毛利率（%）",
                        color=CHART_COLORS["gold"]
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            # 完整損益表數據表格
            st.markdown("#### 完整損益表數據")
            display_cols = {
                "revenues": "營收",
                "grossprofit": "毛利",
                "operatingincomeloss": "營業利潤",
                "netincomeloss": "淨利潤",
                "eps_basic": "EPS（基本）",
            }
            display_df = pd.DataFrame()
            display_df.index = income_df.index.strftime("%Y-%m-%d") if hasattr(income_df.index, 'strftime') else income_df.index
            for col, name in display_cols.items():
                if col in income_df.columns:
                    display_df[name] = income_df[col].apply(
                        lambda x: format_large_number(x) if col != "eps_basic" else (f"{x:.2f}" if not pd.isna(x) else "N/A")
                    )
            
            st.dataframe(display_df, use_container_width=True)
    
    # ============================================================
    # 頁籤 2：資產負債表分析
    # ============================================================
    with tab2:
        st.subheader("🏦 資產負債表關鍵指標")
        
        if balance_df.empty:
            st.warning("無法獲取資產負債表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in balance_df.index]
            
            # 資產、負債、股東權益趨勢
            series = []
            if "assets" in balance_df.columns:
                series.append(("總資產", balance_df["assets"].tolist(), CHART_COLORS["steel_blue"]))
            if "liabilities" in balance_df.columns:
                series.append(("總負債", balance_df["liabilities"].tolist(), CHART_COLORS["dark_red"]))
            if "stockholdersequity" in balance_df.columns:
                series.append(("股東權益", balance_df["stockholdersequity"].tolist(), CHART_COLORS["dark_green"]))
            
            if series:
                fig = create_multi_bar_chart(dates, series, "資產負債結構趨勢", "金額（元）")
                st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                # 流動比率趨勢
                if "assetscurrent" in balance_df.columns and "liabilitiescurrent" in balance_df.columns:
                    current_ratios = [
                        safe_divide(ca, cl) for ca, cl in zip(
                            balance_df["assetscurrent"].fillna(0),
                            balance_df["liabilitiescurrent"].replace(0, np.nan).fillna(1)
                        )
                    ]
                    fig2 = create_bar_chart(
                        dates, current_ratios, "流動比率趨勢",
                        "日期", "流動比率",
                        color=CHART_COLORS["teal"]
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            with col_b:
                # 負債比率趨勢
                if "liabilities" in balance_df.columns and "assets" in balance_df.columns:
                    debt_ratios = [
                        safe_divide(d, a) * 100 for d, a in zip(
                            balance_df["liabilities"].fillna(0),
                            balance_df["assets"].replace(0, np.nan).fillna(1)
                        )
                    ]
                    fig3 = create_bar_chart(
                        dates, debt_ratios, "負債比率趨勢（%）",
                        "日期", "負債比率（%）",
                        color=CHART_COLORS["purple"]
                    )
                    st.plotly_chart(fig3, use_container_width=True)
            
            # 財務比率表格
            st.markdown("#### 財務比率計算")
            ratio_rows = []
            for i, (d, row) in enumerate(balance_df.iterrows()):
                assets = row.get("assets", np.nan)
                liabilities = row.get("liabilities", np.nan)
                equity = row.get("stockholdersequity", np.nan)
                ca = row.get("assetscurrent", np.nan)
                cl = row.get("liabilitiescurrent", np.nan)
                
                ratio_rows.append({
                    "日期": d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d),
                    "負債比率": f"{safe_divide(liabilities, assets)*100:.2f}%" if not pd.isna(assets) else "N/A",
                    "流動比率": f"{safe_divide(ca, cl):.2f}" if not (pd.isna(ca) or pd.isna(cl)) else "N/A",
                    "股東權益": format_large_number(equity),
                    "總資產": format_large_number(assets),
                })
            
            st.dataframe(pd.DataFrame(ratio_rows), use_container_width=True, hide_index=True)
    
    # ============================================================
    # 頁籤 3：現金流量表分析
    # ============================================================
    with tab3:
        st.subheader("💰 現金流量表關鍵指標")
        
        if cashflow_df.empty:
            st.warning("無法獲取現金流量表數據。")
        else:
            dates = [d.strftime("%Y-%m") for d in cashflow_df.index]
            
            # 三大現金流趨勢
            series = []
            if "netcashprovidedbyusedinoperatingactivities" in cashflow_df.columns:
                series.append(("營運現金流", cashflow_df["netcashprovidedbyusedinoperatingactivities"].tolist(), CHART_COLORS["dark_green"]))
            if "netcashprovidedbyusedininvestingactivities" in cashflow_df.columns:
                series.append(("投資現金流", cashflow_df["netcashprovidedbyusedininvestingactivities"].tolist(), CHART_COLORS["dark_red"]))
            if "netcashprovidedbyusedinfinancingactivities" in cashflow_df.columns:
                series.append(("融資現金流", cashflow_df["netcashprovidedbyusedinfinancingactivities"].tolist(), CHART_COLORS["steel_blue"]))
            
            if series:
                fig = create_multi_bar_chart(dates, series, "三大現金流趨勢", "金額（元）")
                st.plotly_chart(fig, use_container_width=True)
            
            # 自由現金流趨勢
            if cashflow_result:
                fcf_dates = [r["date"] for r in cashflow_result]
                fcf_values = [r["free_cash_flow"] for r in cashflow_result]
                fig2 = create_bar_chart(
                    fcf_dates, fcf_values, "自由現金流趨勢",
                    "日期", "自由現金流（元）",
                )
                st.plotly_chart(fig2, use_container_width=True)
            
            # 完整現金流數據表格
            st.markdown("#### 詳細現金流數據")
            if cashflow_result:
                cf_rows = []
                for r in cashflow_result:
                    cf_rows.append({
                        "日期": r["date"],
                        "營運現金流": format_large_number(r["operating_cash_flow"]),
                        "投資現金流": format_large_number(r["investing_cash_flow"]),
                        "融資現金流": format_large_number(r["financing_cash_flow"]),
                        "淨利潤": format_large_number(r["net_income"]),
                        "資本支出": format_large_number(r["capex"]),
                        "自由現金流": format_large_number(r["free_cash_flow"]),
                    })
                st.dataframe(pd.DataFrame(cf_rows), use_container_width=True, hide_index=True)
    
    # ============================================================
    # 頁籤 4：三階段財報分析
    # ============================================================
    with tab4:
        st.subheader("🎯 三階段財報分析")
        
        # 數據品質報告
        quality_report = generate_data_quality_report(financial_data, income_df, balance_df, cashflow_df)
        with st.expander(f"📋 數據品質報告（{quality_report['quality_level']}）", expanded=False):
            col_q1, col_q2 = st.columns(2)
            with col_q1:
                st.write(f"**數據品質等級**：{quality_report['quality_level']}")
                st.write(f"**財務數據年份數**：{quality_report['years_count']} 年")
                if quality_report["missing_fields"]:
                    st.write("**缺失欄位**：")
                    for f in quality_report["missing_fields"]:
                        st.write(f"  - {f}")
                else:
                    st.write("**缺失欄位**：無（資料完整）")
            with col_q2:
                st.write("**計算欄位說明**：")
                for f in quality_report["computed_fields"]:
                    st.write(f"  - {f}")
                if quality_report["limitations"]:
                    st.write("**分析限制**：")
                    for l in quality_report["limitations"]:
                        st.warning(l)
        
        st.markdown("---")
        
        # ---- 階段一：Piotroski F-Score ----
        st.markdown("### 📊 階段一：Piotroski F-Score")
        
        if fscore_result is None:
            st.warning("⚠️ 財務數據不足 2 年，無法計算 F-Score。")
        else:
            col_s1, col_s2 = st.columns([1, 2])
            with col_s1:
                total = fscore_result["total_score"]
                if total >= 7:
                    rating = "強烈看好 🌟"
                    rating_color = "green"
                elif total >= 4:
                    rating = "中性 ⚖️"
                    rating_color = "orange"
                else:
                    rating = "謹慎看待 ⚠️"
                    rating_color = "red"
                
                st.metric(label="F-Score 總分", value=f"{total} / 9", delta=f"{rating}")
            
            with col_s2:
                passed = total
                failed = 9 - total
                fig_pie = create_pie_chart(
                    ["通過", "未通過"],
                    [passed, failed],
                    "F-Score 通過率",
                    [CHART_COLORS["dark_green"], CHART_COLORS["dark_red"]],
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            display_fscore_tables(fscore_result)
        
        st.markdown("---")
        
        # ---- 階段二：杜邦分析 ----
        st.markdown("### 🔬 階段二：杜邦分析")
        
        if not dupont_result:
            st.warning("⚠️ 無法進行杜邦分析，請確認財務數據是否完整。")
        else:
            # 最新 ROE
            latest_roe = dupont_result[0]["roe_dupont"]
            st.metric(label="最新年度 ROE", value=f"{latest_roe*100:.2f}%")
            
            # 年度杜邦分析表格
            st.markdown("#### 年度杜邦分析表格")
            dupont_rows = []
            for entry in dupont_result:
                dupont_rows.append({
                    "日期": entry["date"],
                    "淨利率": f"{entry['net_margin']*100:.2f}%",
                    "資產周轉率": f"{entry['asset_turnover']:.4f}",
                    "權益乘數": f"{entry['equity_multiplier']:.2f}",
                    "計算ROE": f"{entry['roe_dupont']*100:.2f}%",
                    "直接ROE": f"{entry['roe_direct']*100:.2f}%",
                })
            st.dataframe(pd.DataFrame(dupont_rows), use_container_width=True, hide_index=True)
            
            # 趨勢圖
            dupont_dates = [e["date"] for e in dupont_result]
            fig_dup = create_line_chart(
                dupont_dates,
                [
                    ("淨利率", [e["net_margin"]*100 for e in dupont_result], CHART_COLORS["dark_green"]),
                    ("ROE", [e["roe_dupont"]*100 for e in dupont_result], CHART_COLORS["steel_blue"]),
                ],
                "ROE 與淨利率趨勢（%）",
                "百分比（%）",
            )
            st.plotly_chart(fig_dup, use_container_width=True)
            
            # 趨勢變化表格（有變化值的年度）
            trend_rows = [r for r in dupont_result if r["net_margin_change"] is not None]
            if trend_rows:
                st.markdown("#### 趨勢變化分析表格")
                trend_display = []
                for entry in trend_rows:
                    trend_display.append({
                        "日期": entry["date"],
                        "淨利率變化": f"{entry['net_margin_change']*100:+.2f}%",
                        "資產周轉率變化": f"{entry['asset_turnover_change']:+.4f}",
                        "權益乘數變化": f"{entry['equity_multiplier_change']:+.2f}",
                        "ROE 變化": f"{entry['roe_change']*100:+.2f}%",
                    })
                st.dataframe(pd.DataFrame(trend_display), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # ---- 階段三：現金流分析 ----
        st.markdown("### 💧 階段三：現金流分析")
        
        if not cashflow_result:
            st.warning("⚠️ 無法進行現金流分析，請確認財務數據是否完整。")
        else:
            latest_cf = cashflow_result[0]
            
            # 品質指標
            st.metric(
                label=f"現金流品質評估：{latest_cf['quality_rating']}",
                value=f"{latest_cf['ocf_quality_ratio']:.2f}",
                help="OCF品質比率 = 營運現金流 / 淨利潤"
            )
            
            # 現金流關鍵指標表格
            st.markdown("#### 現金流關鍵指標")
            kpi_df = pd.DataFrame([{
                "指標": "營運現金流品質比率",
                "數值": f"{latest_cf['ocf_quality_ratio']:.2f}",
                "評估": latest_cf['quality_rating'],
            }, {
                "指標": "自由現金流（最新年度）",
                "數值": format_large_number(latest_cf['free_cash_flow']),
                "評估": "正值為佳" if latest_cf['free_cash_flow'] > 0 else "需關注",
            }])
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)
            
            # 現金流結構分析
            st.markdown("#### 現金流結構分析（最新年度）")
            structure_df = pd.DataFrame([{
                "類型": "營運現金流",
                "金額": format_large_number(latest_cf['operating_cash_flow']),
            }, {
                "類型": "投資現金流",
                "金額": format_large_number(latest_cf['investing_cash_flow']),
            }, {
                "類型": "融資現金流",
                "金額": format_large_number(latest_cf['financing_cash_flow']),
            }])
            st.dataframe(structure_df, use_container_width=True, hide_index=True)
            
            # 詳細現金流數據
            st.markdown("#### 詳細現金流數據（多年度）")
            detail_rows = []
            for r in cashflow_result:
                detail_rows.append({
                    "日期": r["date"],
                    "營運現金流": format_large_number(r["operating_cash_flow"]),
                    "投資現金流": format_large_number(r["investing_cash_flow"]),
                    "融資現金流": format_large_number(r["financing_cash_flow"]),
                    "淨利潤": format_large_number(r["net_income"]),
                    "資本支出": format_large_number(r["capex"]),
                    "現金流總計": format_large_number(r["operating_cash_flow"] + r["investing_cash_flow"] + r["financing_cash_flow"]),
                })
            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
    
    # ============================================================
    # 頁籤 5：AI 分析
    # ============================================================
    with tab5:
        st.subheader("🤖 AI 深度財務分析")
        
        if not openai_key:
            st.warning("⚠️ 請在左側側邊欄填入 OpenAI API 金鑰以使用 AI 分析功能。")
        else:
            if st.button("🚀 開始 AI 分析", type="primary"):
                with st.spinner("🤖 正在使用 AI 進行三階段財務分析，請稍候（約需 30-60 秒）..."):
                    st.info("📊 正在使用 AI 進行三階段財務分析，包含 F-Score 解讀、杜邦趨勢洞察、現金流深度分析...")
                    
                    # 準備 AI 分析所需的整合數據
                    ai_data = prepare_ai_analysis_data(
                        financial_data, fscore_result, dupont_result, cashflow_result,
                        stock_id, company_info, market_cap, latest_price
                    )
                    
                    # 呼叫 OpenAI 新版 API（client.chat.completions.create）
                    ai_result = run_ai_analysis(openai_key, ai_data)
                
                if ai_result.startswith("⚠️"):
                    st.error(ai_result)
                else:
                    st.success("✅ AI 分析完成！")
                    st.markdown(ai_result)
            else:
                st.info("""
**AI 分析功能說明**

點擊「開始 AI 分析」按鈕後，系統將使用 OpenAI o4-mini 模型進行：
- 🎯 三階段評分總結（F-Score、杜邦、現金流）
- 📊 Piotroski F-Score 指標解讀
- 🔬 杜邦分析趨勢洞察
- 💧 現金流結構深度分析
- 🏛️ 台股市場特性與投資環境分析
- ⚠️ 風險因素與後續追蹤重點

分析約需 30-60 秒，請耐心等待。
                """)


if __name__ == "__main__":
    main()

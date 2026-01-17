import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
import os

# 安全导入 scipy
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==========================================
# 0. 配置与常量
# ==========================================
CONFIG_FILE = 'strategy_config_v6.json'

DEFAULT_CORE_CODES = ["518880", "513100", "588000", "512890"]
PRESET_ETFS = {
    "518880": "黄金ETF (避险锚)",
    "513100": "纳指100 (美股科技)",
    "588000": "科创50 (A股进攻)",
    "512890": "红利低波 (A股防守)",
    "511090": "30年国债 (债牛对冲)",
    "513520": "日经ETF (日本市场)",
    "510300": "沪深300 (核心资产)",
    "159915": "创业板指 (成长旧王)"
}

PRESET_CONCEPTS = [
    "机器人概念", "商业航天概念", "脑机接口", "低空经济", 
    "算力概念", "CPO概念", "人工智能", "半导体", 
    "量子科技", "6G概念", "固态电池", "数据要素",
    "车路云", "人形机器人", "信创", "创新药",
    "核污染防治", "超导概念", "冷液服务器"
]
DEFAULT_SATELLITE_CONCEPTS = ["机器人概念", "商业航天概念", "脑机接口", "低空经济", "算力概念"]

DEFAULT_PARAMS = {
    'invest_ratio': 0.8,
    'core_codes': DEFAULT_CORE_CODES,
    'core_lookback': 25, 'core_smooth': 3, 'core_top_n': 1, 'core_allow_cash': True, 'core_score_mode': '纯收益 (Return)',
    'sat_concepts': DEFAULT_SATELLITE_CONCEPTS,
    'sat_lookback': 5, 'sat_smooth': 1, 'sat_top_n': 2, 'sat_allow_cash': False, 'sat_score_mode': '量价爆发 (PV Breakout)' # 卫星默认激进模式
}

TRANSACTION_COST = 0.0001 

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                config = DEFAULT_PARAMS.copy()
                config.update(saved)
                return config
        except: return DEFAULT_PARAMS.copy()
    return DEFAULT_PARAMS.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except: pass

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(page_title="AlphaTarget v6 | 核心卫星双驱策略", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #f8f9fa; font-family: 'Roboto', sans-serif;}
    .metric-card {background-color: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
    .metric-label {color: #666; font-size: 0.85rem; text-transform: uppercase;}
    .metric-value {color: #333; font-size: 1.5rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 健壮数据层 (支持成交量 Volume)
# ==========================================
@st.cache_data(ttl=3600*12) 
def get_etf_list():
    try: return ak.fund_etf_spot_em()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_etf_data(codes, end_date_str):
    start_str = '20150101' 
    price_dict = {}
    vol_dict = {} # 新增
    name_map = {}
    etf_list = get_etf_list()
    
    for code in codes:
        name = code
        if code in PRESET_ETFS: name = PRESET_ETFS[code].split(" ")[0]
        elif not etf_list.empty:
            m = etf_list[etf_list['代码'] == code]
            if not m.empty: name = m.iloc[0]['名称']
        name_map[code] = name
        
        try:
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df.index = pd.to_datetime(df['日期'])
                price_dict[name] = df['收盘'].astype(float)
                # ETF数据通常也有成交量，尝试获取
                if '成交量' in df.columns:
                    vol_dict[name] = df['成交量'].astype(float)
                else:
                    vol_dict[name] = pd.Series(1, index=df.index) # 填充1防止报错
        except: continue

    if not price_dict: return None, None, None
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    data.dropna(how='all', inplace=True)
    
    # Volume对齐
    if vol_dict:
        vol_data = pd.concat(vol_dict, axis=1).sort_index().ffill()
        vol_data = vol_data.reindex(data.index).fillna(0)
    else:
        vol_data = pd.DataFrame(1, index=data.index, columns=data.columns)

    return data, vol_data, name_map

@st.cache_data(ttl=3600*4)
def download_concept_data(concepts, end_date_str):
    start_str = '20150101'
    price_dict = {}
    vol_dict = {} # 新增
    name_map = {}
    
    progress_bar = st.progress(0, text="启动卫星雷达，扫描行业量价数据...")
    total = len(concepts)
    
    for i, concept_name in enumerate(concepts):
        try:
            df = ak.stock_board_concept_hist_em(symbol=concept_name, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df.index = pd.to_datetime(df['日期'])
                price_dict[concept_name] = df['收盘'].astype(float)
                vol_dict[concept_name] = df['成交量'].astype(float) # 获取成交量
                name_map[concept_name] = concept_name
        except Exception:
            pass
        finally:
            progress_bar.progress((i + 1) / total)
            
    progress_bar.empty()

    if not price_dict: return None, None, None
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    
    # Volume对齐
    if vol_dict:
        vol_data = pd.concat(vol_dict, axis=1).sort_index().ffill()
        vol_data = vol_data.reindex(data.index).fillna(0)
    else:
        vol_data = pd.DataFrame(1, index=data.index, columns=data.columns)

    cols_to_drop = [c for c in data.columns if data[c].count() < 20]
    if cols_to_drop: 
        data.drop(columns=cols_to_drop, inplace=True)
        vol_data.drop(columns=cols_to_drop, inplace=True)
        
    return data, vol_data, name_map

# ==========================================
# 3. 策略引擎 (新增量价爆发模式)
# ==========================================
def calculate_score(data, vol_data, lookback, smooth, mode):
    """
    计算得分
    """
    # 1. 基础动量
    momentum = data.pct_change(lookback)
    
    if mode == '风险调整 (Risk-Adjusted)':
        volatility = data.pct_change().rolling(lookback).std() * np.sqrt(lookback)
        score = momentum / (volatility + 0.0001)
        
    elif mode == '趋势质量 (Efficiency Ratio)':
        daily_abs_change = data.diff().abs()
        path_length = daily_abs_change.rolling(lookback).sum()
        net_change = data.diff(lookback).abs()
        er = net_change / (path_length + 0.0001)
        score = momentum * er
        
    elif mode == '量价爆发 (PV Breakout)':
        # === 游资模式核心逻辑 ===
        # 1. 价格爆发：看短周期涨幅
        # 2. 资金进场：看成交量是否放大 (当前量 / 20日均量)
        # 3. 均线生命线：价格跌破 MA20 强制出局
        
        # 量比因子
        ma_vol_20 = vol_data.rolling(20).mean()
        vol_ratio = vol_data / (ma_vol_20 + 1.0) # 加1防除零
        
        # 限制量比最大影响，防止噪音
        vol_factor = vol_ratio.clip(upper=3.0) 
        
        # 核心公式：得分 = 动量 * (0.5 + 0.5 * 量比)
        # 意义：如果有量，得分会放大；如果缩量，得分会打折
        score = momentum * (0.5 + 0.5 * vol_factor)
        
        # === 熔断机制：MA20 ===
        ma_20 = data.rolling(20).mean()
        # 创建掩码：收盘价 < MA20 的位置
        mask_below_ma = data < ma_20
        
        # 将破位的得分强制设为负无穷 (强制卖出)
        score[mask_below_ma] = -np.inf
        
    else:
        score = momentum
        
    if smooth > 1: 
        score = score.rolling(smooth).mean()
        
    return score

def run_strategy(data, vol_data, params):
    lookback = params['lookback']
    smooth = params['smooth']
    threshold = 0.005 
    top_n = params['top_n']
    mode = params['score_mode']
    allow_cash = params['allow_cash']
    
    daily_ret = data.pct_change().fillna(0)
    
    # 传入 vol_data 计算得分
    score_df = calculate_score(data, vol_data, lookback, smooth, mode)
    
    p_score = score_df.shift(1).values
    p_ret = daily_ret.values
    n_days, n_assets = daily_ret.shape
    
    strategy_ret = np.zeros(n_days)
    current_holdings = [-1] * top_n 
    trade_count = 0
    holdings_hist = []
    
    for i in range(n_days):
        row_score = p_score[i]
        clean_score = np.nan_to_num(row_score, nan=-np.inf)
        
        if np.isneginf(clean_score).all():
            holdings_hist.append([-1]*top_n)
            continue
        
        # 避险检查
        if allow_cash:
            for k in range(top_n):
                if current_holdings[k] != -1:
                    s = clean_score[current_holdings[k]]
                    # 只要得分<0 或 为-inf(破均线) 就卖出
                    if s < 0 or s == -np.inf: current_holdings[k] = -1
        
        # 候选池
        curr_set = set(current_holdings)
        candidates = []
        for idx in np.argsort(clean_score)[::-1]:
            if idx not in curr_set:
                if clean_score[idx] == -np.inf: continue 
                # 只有得分>0 (正动量) 才考虑买入卫星
                if clean_score[idx] > 0: 
                    candidates.append(idx)
        
        # 换仓
        made_swap = True
        while made_swap and candidates:
            made_swap = False
            worst_h_idx = -1
            min_score = np.inf
            worst_pos = -1
            
            for k, h_idx in enumerate(current_holdings):
                s = 0.0 if h_idx == -1 else clean_score[h_idx]
                if s < min_score:
                    min_score = s
                    worst_h_idx = h_idx
                    worst_pos = k
            
            best_c_idx = candidates[0]
            if clean_score[best_c_idx] > min_score + threshold:
                cost = TRANSACTION_COST if worst_h_idx == -1 else TRANSACTION_COST * 2
                strategy_ret[i] -= cost / top_n
                trade_count += 1
                current_holdings[worst_pos] = best_c_idx
                candidates.pop(0)
                made_swap = True
                
        # 收益
        day_ret = 0.0
        for h_idx in current_holdings:
            if h_idx != -1: day_ret += p_ret[i, h_idx]
        
        strategy_ret[i] += day_ret / top_n
        holdings_hist.append(list(current_holdings))
        
    equity_curve = (1 + strategy_ret).cumprod()
    return equity_curve, trade_count, holdings_hist, strategy_ret

def calc_metrics(equity):
    if len(equity) < 2: return {}
    total = equity[-1] - 1
    days = len(equity)
    ann_ret = (1 + total) ** (252/days) - 1
    daily_ret = pd.Series(equity).pct_change().fillna(0)
    vol = daily_ret.std() * np.sqrt(252)
    dd = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
    max_dd = dd.min()
    sharpe = (ann_ret - 0.03) / (vol + 1e-9)
    return {"CAGR": ann_ret, "MaxDD": max_dd, "Sharpe": sharpe, "Vol": vol}

def metric_html(label, value, color="#333"):
    return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color}">{value}</div></div>"""

# ==========================================
# 5. 主程序 UI
# ==========================================
def main():
    if 'params' not in st.session_state:
        st.session_state.params = load_config()
    
    with st.sidebar:
        st.title("🛰️ 核心-卫星策略台")
        st.markdown("### 1. 顶层资产配置")
        core_weight = st.slider("核心策略权重", 0.0, 1.0, st.session_state.params.get('invest_ratio', 0.8), 0.1)
        
        st.divider()
        st.markdown("### 2. 回测时间机 (Time Machine)")
        date_mode = st.radio("时间模式", ["全历史 (Max)", "自定义 (Custom)"], horizontal=True)
        start_d = datetime(2016,1,1) 
        end_d = datetime.now()
        if date_mode == "自定义 (Custom)":
            c1, c2 = st.columns(2)
            start_d = datetime.combine(c1.date_input("开始", datetime(2020,1,1)), datetime.min.time())
            end_d = datetime.combine(c2.date_input("结束", datetime.now()), datetime.min.time())
        
        st.divider()
        tab_core, tab_sat = st.tabs(["🔵 核心 (ETF)", "🔴 卫星 (概念)"])
        
        with tab_core:
            all_etfs = get_etf_list()
            pre_opts = [f"{k} | {v}" for k,v in PRESET_ETFS.items()]
            curr_core = st.session_state.params.get('core_codes', DEFAULT_CORE_CODES)
            sel_core_disp = st.multiselect("核心池", pre_opts, default=[x for x in pre_opts if x.split(" | ")[0] in curr_core])
            sel_core_codes = [x.split(" | ")[0] for x in sel_core_disp]
            c_mode = st.selectbox("核心算法", ["纯收益 (Return)", "风险调整 (Risk-Adjusted)", "趋势质量 (Efficiency Ratio)"], index=0, key='c_mode')
            c_lookback = st.slider("核心-周期", 5, 60, st.session_state.params.get('core_lookback', 25))
            c_smooth = st.slider("核心-平滑", 1, 10, st.session_state.params.get('core_smooth', 3))
            c_topn = st.slider("核心-持仓", 1, 3, st.session_state.params.get('core_top_n', 1))
            c_cash = st.checkbox("核心-避险", st.session_state.params.get('core_allow_cash', True))
            
        with tab_sat:
            curr_sat = st.session_state.params.get('sat_concepts', DEFAULT_SATELLITE_CONCEPTS)
            sel_sat_concepts = st.multiselect("卫星池", PRESET_CONCEPTS, default=curr_sat)
            
            st.info("🔥 卫星新算法：【量价爆发】。结合涨幅与成交量，且破位20日线强制止损。")
            s_mode_idx = 3 # 默认选PV Breakout
            s_modes_list = ["纯收益 (Return)", "风险调整 (Risk-Adjusted)", "趋势质量 (Efficiency Ratio)", "量价爆发 (PV Breakout)"]
            if 'sat_score_mode' in st.session_state.params and st.session_state.params['sat_score_mode'] in s_modes_list:
                s_mode_idx = s_modes_list.index(st.session_state.params['sat_score_mode'])
            
            s_mode = st.selectbox("卫星算法", s_modes_list, index=s_mode_idx, key='s_mode')
            s_lookback = st.slider("卫星-周期 (建议3-5)", 2, 20, st.session_state.params.get('sat_lookback', 5))
            s_smooth = st.slider("卫星-平滑", 1, 5, st.session_state.params.get('sat_smooth', 1))
            s_topn = st.slider("卫星-持仓", 1, 5, st.session_state.params.get('sat_top_n', 2))
            s_cash = st.checkbox("卫星-避险", st.session_state.params.get('sat_allow_cash', False))

        st.divider()
        if st.button("🚀 运行双核回测"):
            new_conf = st.session_state.params.copy()
            new_conf.update({
                'invest_ratio': core_weight,
                'core_codes': sel_core_codes, 'core_lookback': c_lookback, 'core_smooth': c_smooth, 'core_top_n': c_topn, 'core_allow_cash': c_cash, 'core_score_mode': c_mode,
                'sat_concepts': sel_sat_concepts, 'sat_lookback': s_lookback, 'sat_smooth': s_smooth, 'sat_top_n': s_topn, 'sat_allow_cash': s_cash, 'sat_score_mode': s_mode
            })
            st.session_state.params = new_conf
            save_config(new_conf)
            st.rerun()

    # --- 主界面 ---
    st.title("AlphaTarget v6 | 核心卫星双驱策略")
    
    if not sel_core_codes or not sel_sat_concepts: st.warning("请配置资产池"); st.stop()

    t_date = datetime.now()
    if t_date.hour < 15: t_date -= timedelta(days=1)
    end_str = t_date.strftime('%Y%m%d')
    
    c1, c2 = st.columns(2)
    with c1:
        with st.spinner("同步核心数据..."):
            core_data, core_vol, core_map = download_etf_data(sel_core_codes, end_str)
    with c2:
        sat_data, sat_vol, sat_map = download_concept_data(sel_sat_concepts, end_str)
        
    if core_data is None or sat_data is None: st.error("数据获取失败"); st.stop()
        
    common_idx = core_data.index.intersection(sat_data.index)
    mask = (common_idx >= start_d) & (common_idx <= end_d)
    common_idx = common_idx[mask]
    
    if len(common_idx) < 20: st.error(f"数据不足"); st.stop()
    
    core_data, core_vol = core_data.loc[common_idx], core_vol.loc[common_idx]
    sat_data, sat_vol = sat_data.loc[common_idx], sat_vol.loc[common_idx]
    
    # 2. 回测
    p_core = {'lookback': c_lookback, 'smooth': c_smooth, 'top_n': c_topn, 'score_mode': c_mode, 'allow_cash': c_cash}
    core_eq, core_tr, core_hist, core_dret = run_strategy(core_data, core_vol, p_core)
    
    p_sat = {'lookback': s_lookback, 'smooth': s_smooth, 'top_n': s_topn, 'score_mode': s_mode, 'allow_cash': s_cash}
    sat_eq, sat_tr, sat_hist, sat_dret = run_strategy(sat_data, sat_vol, p_sat)
    
    # 3. 组合
    combo_dret = core_weight * core_dret + (1-core_weight) * sat_dret
    combo_eq = (1 + combo_dret).cumprod()
    
    # 4. 报表
    m_combo = calc_metrics(combo_eq)
    m_core = calc_metrics(core_eq)
    m_sat = calc_metrics(sat_eq)
    
    st.markdown("### 📊 组合总览 (Portfolio)")
    cols = st.columns(4)
    with cols[0]: st.markdown(metric_html("组合年化收益", f"{m_combo['CAGR']:.1%}", "#d62728"), unsafe_allow_html=True)
    with cols[1]: st.markdown(metric_html("组合最大回撤", f"{m_combo['MaxDD']:.1%}", "green"), unsafe_allow_html=True)
    with cols[2]: st.markdown(metric_html("卫星年化 (Sat)", f"{m_sat['CAGR']:.1%}", "#d62728"), unsafe_allow_html=True)
    with cols[3]: st.markdown(metric_html("卫星夏普", f"{m_sat['Sharpe']:.2f}", "#333"), unsafe_allow_html=True)
    
    st.write("")
    
    tab1, tab2, tab3 = st.tabs(["📈 净值透视", "🗂️ 持仓历史", "🔬 归因分析"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=common_idx, y=combo_eq, name="总组合", line=dict(color='#1e3c72', width=3)))
        fig.add_trace(go.Scatter(x=common_idx, y=core_eq, name=f"核心 (Core)", line=dict(color='#63b2ee', width=1)))
        fig.add_trace(go.Scatter(x=common_idx, y=sat_eq, name=f"卫星 (Sat)", line=dict(color='#d62728', width=1)))
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.markdown("**📅 历史持仓日历 (Holdings Log)**")
        def fmt_holdings(hist, map_d, cols):
            res = []
            for h_idxs in hist:
                names = []
                for i in h_idxs:
                    if i == -1: names.append("Cash")
                    else: names.append(map_d.get(cols[i], cols[i]))
                res.append(", ".join(names))
            return res
            
        df_hold = pd.DataFrame(index=common_idx)
        df_hold['🔵 核心持仓'] = fmt_holdings(core_hist, core_map, core_data.columns)
        df_hold['🔴 卫星持仓'] = fmt_holdings(sat_hist, sat_map, sat_data.columns)
        st.dataframe(df_hold.sort_index(ascending=False), use_container_width=True, height=500)
            
    with tab3:
        attr_data = {
            "策略": ["核心 (Core)", "卫星 (Satellite)"],
            "年化收益": [m_core['CAGR'], m_sat['CAGR']],
            "最大回撤": [m_core['MaxDD'], m_sat['MaxDD']],
            "波动率": [m_core['Vol'], m_sat['Vol']],
            "夏普比": [m_core['Sharpe'], m_sat['Sharpe']],
            "交易次数": [core_tr, sat_tr]
        }
        df_attr = pd.DataFrame(attr_data).set_index("策略")
        st.dataframe(df_attr.style.format({
            "年化收益": "{:.1%}", "最大回撤": "{:.1%}", "波动率": "{:.1%}", "夏普比": "{:.2f}"
        }), use_container_width=True)

if __name__ == "__main__":
    main()

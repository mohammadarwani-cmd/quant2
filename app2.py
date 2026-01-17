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
CONFIG_FILE = 'strategy_config_v5.json'

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

# 增加更多热门概念，并做好名称适配
PRESET_CONCEPTS = [
    "机器人概念", "商业航天概念", "脑机接口", "低空经济", 
    "算力概念", "CPO概念", "人工智能", "半导体", 
    "量子科技", "6G概念", "固态电池", "数据要素",
    "车路云", "人形机器人", "信创", "创新药"
]
DEFAULT_SATELLITE_CONCEPTS = ["机器人概念", "商业航天概念", "脑机接口", "低空经济", "算力概念"]

DEFAULT_PARAMS = {
    'invest_ratio': 0.8,
    'core_codes': DEFAULT_CORE_CODES,
    'core_lookback': 25, 'core_smooth': 3, 'core_top_n': 1, 'core_allow_cash': True,
    'sat_concepts': DEFAULT_SATELLITE_CONCEPTS,
    'sat_lookback': 10, 'sat_smooth': 2, 'sat_top_n': 2, 'sat_allow_cash': False,
    'score_mode': '纯收益 (Return)'
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
st.set_page_config(page_title="AlphaTarget v5 | 双核驱动量化系统", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
    .stApp {background-color: #f8f9fa; font-family: 'Roboto', sans-serif;}
    .metric-card {background-color: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}
    .metric-label {color: #666; font-size: 0.85rem; text-transform: uppercase;}
    .metric-value {color: #333; font-size: 1.5rem; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 健壮数据层 (Robust Data Layer)
# ==========================================
@st.cache_data(ttl=3600*12) 
def get_etf_list():
    try: return ak.fund_etf_spot_em()
    except: return pd.DataFrame()

@st.cache_data(ttl=3600*4)
def download_etf_data(codes, end_date_str):
    start_str = '20190101'
    price_dict = {}
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
        except: continue

    if not price_dict: return None, None
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    # 核心资产通常数据较好，直接dropna
    data.dropna(how='all', inplace=True)
    return (data, name_map) if len(data) >= 20 else (None, None)

@st.cache_data(ttl=3600*4)
def download_concept_data(concepts, end_date_str):
    """
    下载概念数据 (增强容错版)
    """
    start_str = '20190101'
    price_dict = {}
    name_map = {}
    
    progress_bar = st.progress(0, text="启动卫星雷达，扫描行业数据...")
    total = len(concepts)
    success_count = 0
    
    for i, concept_name in enumerate(concepts):
        try:
            # 尝试下载
            df = ak.stock_board_concept_hist_em(symbol=concept_name, period="daily", start_date=start_str, end_date=end_date_str, adjust="qfq")
            if not df.empty:
                df.index = pd.to_datetime(df['日期'])
                price_dict[concept_name] = df['收盘'].astype(float)
                name_map[concept_name] = concept_name
                success_count += 1
        except Exception:
            # 某些概念可能改名或下线，静默失败，不中断程序
            pass
        finally:
            progress_bar.progress((i + 1) / total)
            
    progress_bar.empty()

    if not price_dict: return None, None
    
    # 概念板块上线时间不一，不能简单 dropna(how='any')，否则会因为一个新概念把所有历史数据切掉
    # 策略：取并集，空值向后填充，仍然空的填0或处理为不交易
    data = pd.concat(price_dict, axis=1).sort_index().ffill()
    
    # 再次清洗：如果某列数据太少(<20天)，剔除该列，防止计算动量报错
    cols_to_drop = [c for c in data.columns if data[c].count() < 20]
    if cols_to_drop:
        data.drop(columns=cols_to_drop, inplace=True)
        # st.toast(f"已剔除数据过短的概念: {','.join(cols_to_drop)}", icon="⚠️")
        
    return data, name_map

# ==========================================
# 3. 策略引擎
# ==========================================
def calculate_score(data, lookback, smooth, mode):
    ret = data.pct_change(lookback)
    if mode == '风险调整 (Risk-Adjusted)':
        vol = data.pct_change().rolling(lookback).std() * np.sqrt(lookback)
        score = ret / (vol + 0.0001)
    else:
        score = ret
    if smooth > 1: score = score.rolling(smooth).mean()
    return score

def run_strategy(data, params):
    # 解包
    lookback = params['lookback']
    smooth = params['smooth']
    threshold = 0.005 
    top_n = params['top_n']
    mode = params['score_mode']
    allow_cash = params['allow_cash']
    
    daily_ret = data.pct_change().fillna(0)
    score_df = calculate_score(data, lookback, smooth, mode)
    
    p_score = score_df.shift(1).values
    p_ret = daily_ret.values
    n_days, n_assets = daily_ret.shape
    
    strategy_ret = np.zeros(n_days)
    current_holdings = [-1] * top_n 
    trade_count = 0
    holdings_hist = []
    
    for i in range(n_days):
        row_score = p_score[i]
        
        # 针对概念数据，可能某些列是NaN（未上市），不能all()判断
        # 处理：如果是NaN，给一个极小值
        clean_score = np.nan_to_num(row_score, nan=-np.inf)
        
        # 如果整行都是-inf（当天所有标的都没数据），跳过
        if np.isneginf(clean_score).all():
            holdings_hist.append([-1]*top_n)
            continue
        
        # 避险
        if allow_cash:
            for k in range(top_n):
                if current_holdings[k] != -1:
                    # 检查持有标的是否还在交易(非NaN/Inf)
                    s = clean_score[current_holdings[k]]
                    if s < 0 or s == -np.inf:
                        current_holdings[k] = -1
        
        # 候选
        curr_set = set(current_holdings)
        candidates = []
        for idx in np.argsort(clean_score)[::-1]:
            if idx not in curr_set:
                if clean_score[idx] == -np.inf: continue # 过滤无效数据
                if (not allow_cash) or (clean_score[idx] > 0):
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
        active_pos = 0
        for h_idx in current_holdings:
            if h_idx != -1: 
                day_ret += p_ret[i, h_idx]
                active_pos += 1
        
        # 资金利用率修正：如果是 Top N 模型，空仓部分不产生收益
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
        core_weight = st.slider("核心策略权重 (Core Weight)", 0.0, 1.0, st.session_state.params.get('invest_ratio', 0.8), 0.1)
        st.caption(f"🔵 核心(宽基): {core_weight:.0%} | 🔴 卫星(行业): {1-core_weight:.0%}")
        
        st.divider()
        
        tab_core, tab_sat = st.tabs(["🔵 核心 (ETF)", "🔴 卫星 (概念)"])
        
        with tab_core:
            all_etfs = get_etf_list()
            pre_opts = [f"{k} | {v}" for k,v in PRESET_ETFS.items()]
            curr_core = st.session_state.params.get('core_codes', DEFAULT_CORE_CODES)
            sel_core_disp = st.multiselect("核心池", pre_opts, default=[x for x in pre_opts if x.split(" | ")[0] in curr_core])
            sel_core_codes = [x.split(" | ")[0] for x in sel_core_disp]
            
            c_lookback = st.slider("核心-周期", 5, 60, st.session_state.params.get('core_lookback', 25))
            c_smooth = st.slider("核心-平滑", 1, 10, st.session_state.params.get('core_smooth', 3))
            c_topn = st.slider("核心-持仓", 1, 3, st.session_state.params.get('core_top_n', 1))
            c_cash = st.checkbox("核心-避险", st.session_state.params.get('core_allow_cash', True))
            
        with tab_sat:
            curr_sat = st.session_state.params.get('sat_concepts', DEFAULT_SATELLITE_CONCEPTS)
            sel_sat_concepts = st.multiselect("卫星池 (Concept)", PRESET_CONCEPTS, default=curr_sat)
            
            st.info("💡 建议：卫星策略应使用更短周期，更灵敏地捕捉热点。")
            s_lookback = st.slider("卫星-周期", 3, 30, st.session_state.params.get('sat_lookback', 10))
            s_smooth = st.slider("卫星-平滑", 1, 5, st.session_state.params.get('sat_smooth', 2))
            s_topn = st.slider("卫星-持仓", 1, 5, st.session_state.params.get('sat_top_n', 2))
            s_cash = st.checkbox("卫星-避险", st.session_state.params.get('sat_allow_cash', False))

        st.divider()
        if st.button("🚀 运行双核回测 (Run)"):
            new_conf = st.session_state.params.copy()
            new_conf.update({
                'invest_ratio': core_weight,
                'core_codes': sel_core_codes, 'core_lookback': c_lookback, 'core_smooth': c_smooth, 'core_top_n': c_topn, 'core_allow_cash': c_cash,
                'sat_concepts': sel_sat_concepts, 'sat_lookback': s_lookback, 'sat_smooth': s_smooth, 'sat_top_n': s_topn, 'sat_allow_cash': s_cash
            })
            st.session_state.params = new_conf
            save_config(new_conf)
            st.rerun()

    # --- 主界面 ---
    st.title("AlphaTarget v5 | 核心卫星双驱策略")
    
    if not sel_core_codes or not sel_sat_concepts:
        st.warning("请配置完整的资产池。"); st.stop()

    # 1. 下载
    t_date = datetime.now()
    if t_date.hour < 15: t_date -= timedelta(days=1)
    end_str = t_date.strftime('%Y%m%d')
    
    c1, c2 = st.columns(2)
    with c1:
        with st.spinner("同步核心数据..."):
            core_data, core_map = download_etf_data(sel_core_codes, end_str)
    with c2:
        # 卫星数据下载较慢，Spinner文案区分
        sat_data, sat_map = download_concept_data(sel_sat_concepts, end_str)
        
    if core_data is None or sat_data is None:
        st.error("数据获取失败，请检查网络或减少概念数量。"); st.stop()
        
    # 对齐
    common_idx = core_data.index.intersection(sat_data.index)
    if len(common_idx) < 50: st.error("数据重叠区间过短"); st.stop()
    core_data = core_data.loc[common_idx]
    sat_data = sat_data.loc[common_idx]
    
    # 2. 回测
    p_core = {'lookback': c_lookback, 'smooth': c_smooth, 'top_n': c_topn, 'score_mode': '纯收益 (Return)', 'allow_cash': c_cash}
    core_eq, core_tr, core_hist, core_dret = run_strategy(core_data, p_core)
    
    p_sat = {'lookback': s_lookback, 'smooth': s_smooth, 'top_n': s_topn, 'score_mode': '纯收益 (Return)', 'allow_cash': s_cash}
    sat_eq, sat_tr, sat_hist, sat_dret = run_strategy(sat_data, p_sat)
    
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
    with cols[2]: st.markdown(metric_html("组合夏普比率", f"{m_combo['Sharpe']:.2f}", "#333"), unsafe_allow_html=True)
    with cols[3]: st.markdown(metric_html("波动率 (Vol)", f"{m_combo['Vol']:.1%}", "#333"), unsafe_allow_html=True)
    
    st.write("")
    
    # 详细对比图
    tab1, tab2, tab3 = st.tabs(["📈 净值与相关性", "📝 实时信号", "🔬 归因分析"])
    
    with tab1:
        # 净值图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=common_idx, y=combo_eq, name="总组合 (Combined)", line=dict(color='#1e3c72', width=3)))
        fig.add_trace(go.Scatter(x=common_idx, y=core_eq, name=f"核心 (Core, {core_weight:.0%})", line=dict(color='#63b2ee', width=1)))
        fig.add_trace(go.Scatter(x=common_idx, y=sat_eq, name=f"卫星 (Sat, {1-core_weight:.0%})", line=dict(color='#d62728', width=1)))
        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0,r=0,t=0,b=0), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        # 滚动相关性图 (投行级分析)
        st.markdown("**🔗 核心-卫星 滚动相关性 (60日窗口)**")
        st.caption("观察：当相关性(Correlation) < 0 时，说明卫星资产有效地对冲了核心资产的风险。")
        s_corr = pd.Series(core_dret).rolling(60).corr(pd.Series(sat_dret)).dropna()
        fig_corr = px.area(x=common_idx[-len(s_corr):], y=s_corr, labels={'x':'Date', 'y':'Correlation'})
        fig_corr.update_traces(line_color='#666', fillcolor='rgba(100,100,100,0.2)')
        fig_corr.update_yaxes(range=[-1, 1])
        fig_corr.add_hline(y=0, line_dash="dash", line_color="red")
        fig_corr.update_layout(height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with tab2:
        # 信号解析
        def get_names(hist_list, map_dict, cols):
            idxs = hist_list[-1]
            names = []
            for idx in idxs:
                if idx == -1: names.append("Cash")
                else: names.append(map_dict.get(cols[idx], cols[idx]))
            return names
            
        c_hold = get_names(core_hist, core_map, core_data.columns)
        s_hold = get_names(sat_hist, sat_map, sat_data.columns)
        
        c1, c2 = st.columns(2)
        with c1:
            st.info(f"🔵 核心持仓 (Top {c_topn})")
            for n in c_hold: st.write(f"• **{n}**")
        with c2:
            st.error(f"🔴 卫星持仓 (Top {s_topn})")
            for n in s_hold: st.write(f"• **{n}**")
            
    with tab3:
        # 贡献度表格
        attr_data = {
            "策略": ["核心 (Core)", "卫星 (Satellite)"],
            "年化收益": [m_core['CAGR'], m_sat['CAGR']],
            "最大回撤": [m_core['MaxDD'], m_sat['MaxDD']],
            "波动率": [m_core['Vol'], m_sat['Vol']],
            "夏普比": [m_core['Sharpe'], m_sat['Sharpe']],
            "交易次数": [core_tr, sat_tr]
        }
        df_attr = pd.DataFrame(attr_data).set_index("策略")
        st.markdown("#### 风险收益归因 (Attribution)")
        st.dataframe(df_attr.style.format({
            "年化收益": "{:.1%}", "最大回撤": "{:.1%}", "波动率": "{:.1%}", "夏普比": "{:.2f}"
        }), use_container_width=True)

if __name__ == "__main__":
    main()

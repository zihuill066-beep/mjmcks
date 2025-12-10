import sqlite3
from pathlib import Path
import io
import zipfile
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
import logging
from openai import OpenAI
import hashlib
import secrets
import string
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import base64

# 密码相关工具函数
def generate_salt(length=16):
    """生成随机盐值"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def hash_password(password: str, salt: str = None) -> tuple:
    """哈希密码，返回(哈希值, 盐值)"""
    if salt is None:
        salt = generate_salt()

    # 使用PBKDF2算法
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # 迭代次数
    ).hex()

    return password_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """验证密码"""
    new_hash, _ = hash_password(password, salt)
    return new_hash == stored_hash

# ========== AI 配置 ==========
API_KEY = "sk-zOXHCvNjmUjPCGCmD33e25D714194773A893D2166a86D755"
API_BASE = "https://maas-api.cn-huabei-1.xf-yun.com/v1"
MODEL_ID = "xopdeepseekocr"

# 初始化客户端（全局，在侧边栏配置时重新初始化）

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

def init_ai_client(api_key: str = None, api_base: str = None, model_id: str = None):
    """初始化AI客户端"""
    global client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        return client is not None
    except Exception as e:
        st.error(f"AI客户端初始化失败: {str(e)}")
        return False


# ========== 通用 AI 调用封装 ==========
def ask_ai(messages, json_type=False, model_id=MODEL_ID):
    """
    通用 AI 查询接口
    messages: str 或 list
    json_type: 是否要求返回 JSON（默认关闭，因为情绪分析更适合自然语言）
    """
    global client
    if client is None:
        return "AI功能未初始化，请在侧边栏配置API Key"

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    extra_body = {}
    if json_type:
        extra_body = {
            "response_format": {"type": "json_object"},
            "search_disable": True
        }

    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=messages,
            extra_body=extra_body
        )
        content = resp.choices[0].message.content
        return json.loads(content) if json_type else content
    except Exception as e:
        return f"AI调用失败: {str(e)}"


# ========== 情绪 AI 解读专用函数 ==========
def ai_explain_mood(df):
    """
    输入：你的情绪 DataFrame
    输出：情绪趋势 + 关键因素 + 管理建议（面向用户、自然）
    """
    if df.empty or len(df) < 2:
        return "需要至少2条记录才能进行情绪分析。"

    # 简单统计
    avg_mood = df["mood_score"].mean()
    worst = df.loc[df["mood_score"].idxmin()]
    best = df.loc[df["mood_score"].idxmax()]
    last = df.iloc[-1]

    # 构建摘要信息
    summary = f"""
## 情绪数据统计
最近情绪平均分：{avg_mood:.2f}/10
记录总数：{len(df)}条
时间范围：{df['record_date'].min().strftime('%Y-%m-%d')} 至 {df['record_date'].max().strftime('%Y-%m-%d')}

## 关键记录点
最近一次记录：{last['mood_score']}分
- 活动：{last.get('activities', '无')}
- 备注：{last.get('notes', '无')[:50]}...

情绪最低点：{worst['mood_score']}分
- 日期：{worst['record_date'].strftime('%Y-%m-%d')}
- 活动：{worst.get('activities', '无')}
- 备注：{worst.get('notes', '无')[:50]}...

情绪最高点：{best['mood_score']}分  
- 日期：{best['record_date'].strftime('%Y-%m-%d')}
- 活动：{best.get('activities', '无')}
- 备注：{best.get('notes', '无')[:50]}...
"""

    prompt = f"""
你是一名专业的心理情绪教练，请用**温柔、现实、面向行动**的方式，分析用户近一段时间的情绪数据。

以下是用户的情绪记录摘要：
{summary}

请生成一份温暖、实用的情绪分析报告，包含以下部分：

## 1. 情绪趋势总结
用普通人能理解的语言描述整体情绪变化趋势

## 2. 可能的影响因素
基于活动记录、备注内容等，指出可能的情绪诱因

## 3. 个性化建议（3~5条）
给出具体、可执行的建议，比如：
- 如果发现某些活动带来积极情绪，建议增加这些活动
- 如果发现压力较大，提供简单的减压方法
- 如果情绪波动较大，建议稳定情绪的小技巧

## 4. 温馨提醒
用温暖的话语鼓励用户，肯定TA记录情绪的努力

**注意事项：**
- 语气温柔、避免专业术语
- 避免负面评价，用建设性语言
- 不要输出代码或技术性内容
- 针对数据特点提供具体建议
"""

    return ask_ai(prompt, json_type=False)


def transcribe_audio_file(audio_bytes):
    """将音频字节转成文字"""
    try:
        import tempfile
        import wave
        import struct

        # 将字节数据保存到临时WAV文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # 使用 speech_recognition
        recognizer = sr.Recognizer()

        with sr.AudioFile(tmp_path) as source:
            # 调整环境噪音
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

            try:
                # 尝试使用Google语音识别
                text = recognizer.recognize_google(audio, language='zh-CN')
                return text
            except sr.UnknownValueError:
                return "无法识别语音内容"
            except sr.RequestError as e:
                return f"语音识别服务出错: {e}"
            except Exception as e:
                return f"识别失败: {str(e)}"

    except ImportError as e:
        return f"语音识别依赖缺失: {e}"
    except Exception as e:
        return f"音频处理失败: {e}"
    finally:
        # 清理临时文件
        if 'tmp_path' in locals():
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def wechat_style_recorder():
    """微信式长按录音组件（修复版）"""

    # 添加CSS样式
    st.markdown("""
    <style>
    .record-instruction {
        text-align: center;
        color: #666;
        margin-top: 10px;
        font-size: 14px;
    }
    .recording-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### 🎤 语音输入")

    # 初始化session state
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""

    # 使用streamlit-mic-recorder
    audio_data = mic_recorder(
        start_prompt="🎤 长按开始录音",
        stop_prompt="⏹️ 松开结束录音",
        key="wechat_recorder",
        format="wav",
        just_once=False,
        use_container_width=True,
        callback=None
    )

    # 录音提示
    st.markdown('<p class="record-instruction">💡 提示：长按按钮说话，松开结束</p>', unsafe_allow_html=True)

    # 处理录音结果
    if audio_data and 'bytes' in audio_data and audio_data['bytes']:
        # 保存音频数据到session state
        st.session_state.audio_data = audio_data['bytes']

        # 显示录音
        st.audio(st.session_state.audio_data, format="audio/wav")

        # 显示录音状态
        st.markdown(
            '<div class="recording-status" style="background-color: #e6f3ff;">✅ 录音完成，请点击"转成文字"按钮</div>',
            unsafe_allow_html=True)

    # 如果有录音数据，显示转换按钮
    if st.session_state.audio_data:
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔤 转成文字", key="transcribe_audio", type="primary"):
                with st.spinner("正在识别语音..."):
                    text = transcribe_audio_file(st.session_state.audio_data)

                    if text and "失败" not in text and "无法识别" not in text:
                        st.session_state.transcribed_text = text
                        st.success("✅ 识别成功！")
                    else:
                        st.error(f"识别失败: {text}")
                        st.session_state.transcribed_text = ""

        with col2:
            if st.button("🗑️ 清除录音", key="clear_audio"):
                st.session_state.audio_data = None
                st.session_state.transcribed_text = ""
                st.rerun()

    # 显示识别结果
    if st.session_state.transcribed_text:
        st.text_area("识别结果",
                     st.session_state.transcribed_text,
                     height=100,
                     key="transcribed_text_area")

        # 确认使用按钮
        if st.button("✅ 使用此文字", key="use_transcribed_text"):
            # 将文字传递到记录表单
            st.session_state.voice_result = st.session_state.transcribed_text
            st.success("文字已准备就绪！")
            st.rerun()

    return st.session_state.get('voice_result', None)

def ai_generate_weekly_report(df):
    """生成周度情绪报告"""
    if df.empty or len(df) < 3:
        return "需要更多记录才能生成周报（建议至少3条）。"

    # 获取最近7天数据
    recent_df = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]
    if len(recent_df) < 2:
        return "本周记录较少，建议多记录几天。"

    # 构建周报数据
    avg_mood = recent_df["mood_score"].mean()
    mood_std = recent_df["mood_score"].std()

    # 分析活动影响
    activity_summary = ""
    if "activities" in recent_df.columns:
        activity_data = []
        for _, row in recent_df.iterrows():
            if pd.notna(row["activities"]) and row["activities"]:
                activities = [a.strip() for a in str(row["activities"]).split(",")]
                for activity in activities:
                    if activity:
                        activity_data.append({"activity": activity, "mood": row["mood_score"]})

        if activity_data:
            activity_df = pd.DataFrame(activity_data)
            activity_stats = activity_df.groupby("activity")["mood"].agg(["mean", "count"]).round(2)
            top_activities = activity_stats.sort_values("mean", ascending=False).head(3)

            activity_summary = "\n## 活动影响分析\n"
            for activity, row in top_activities.iterrows():
                activity_summary += f"- {activity}: 平均情绪 {row['mean']:.1f}分（出现{row['count']}次）\n"

    prompt = f"""
你是一名贴心的情绪管理助手，请为用户生成一份温暖、鼓励的周度情绪报告。

## 本周情绪概览
- 记录天数：{len(recent_df)}天
- 平均情绪：{avg_mood:.1f}/10
- 情绪稳定性：{'较稳定' if mood_std < 2 else '波动较大'}
- 时间范围：{recent_df['record_date'].min().strftime('%m/%d')} - {recent_df['record_date'].max().strftime('%m/%d')}

{activity_summary if activity_summary else ''}

## 请生成包含以下内容的周报：
1. **本周情绪总结**：用温暖的语言描述本周情绪特点
2. **进步与亮点**：肯定用户的积极变化和努力
3. **发现与洞察**：基于数据指出有意义的现象
4. **下周小目标**：2-3个简单可行的建议
5. **温馨鼓励**：用支持性的话语结束报告

**风格要求：**
- 语气亲切、鼓励、实用
- 避免说教，用建议而非命令
- 结合具体数据提供个性化反馈
- 保持积极向上的基调
"""

    return ask_ai(prompt, json_type=False)


# ========== 新增：查询函数 ==========
def query_records(
        conn,
        user_id: int = None,
        username: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        min_score: int = None,
        max_score: int = None,
        keyword: str = None
) -> pd.DataFrame:
    """
    查询情绪记录
    """
    # 构建基础查询
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    WHERE 1=1
    """

    params = []

    # 按用户ID筛选
    if user_id is not None:
        sql += " AND mr.user_id = ?"
        params.append(user_id)

    # 按用户名筛选
    if username is not None and username != "所有用户":
        sql += " AND u.username = ?"
        params.append(username)

    # 按日期筛选
    if start_date:
        sql += " AND DATE(mr.record_date) >= ?"
        params.append(start_date.strftime('%Y-%m-%d'))

    if end_date:
        sql += " AND DATE(mr.record_date) <= ?"
        params.append(end_date.strftime('%Y-%m-%d'))

    # 按分数筛选
    if min_score is not None:
        sql += " AND mr.mood_score >= ?"
        params.append(min_score)

    if max_score is not None:
        sql += " AND mr.mood_score <= ?"
        params.append(max_score)

    sql += " ORDER BY mr.record_date DESC"

    # 执行查询
    df = pd.read_sql(sql, conn, params=params)

    # 关键词搜索
    if keyword and not df.empty:
        keyword = keyword.lower()
        mask = (
                df["notes"].str.lower().str.contains(keyword, na=False) |
                df["activities"].str.lower().str.contains(keyword, na=False) |
                df["tags"].str.lower().str.contains(keyword, na=False)
        )
        df = df[mask]

    return df


# ========== 在这里插入批量查询功能（功能3）==========
def batch_query_records(
        conn,
        user_ids: List[int],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        return_type: str = "dataframe"  # 或 "dict", "json"
) -> Union[Dict[int, pd.DataFrame], str]:
    """
    批量查询多个用户的记录
    """
    results = {}
    for user_id in user_ids:
        # 先获取用户名
        username_result = conn.execute(
            "SELECT username FROM users WHERE user_id = ?",
            (user_id,)
        ).fetchone()

        if username_result:
            username = username_result[0]
            user_df = query_records(
                conn,
                user_id=user_id,
                username=username,
                start_date=start_date,
                end_date=end_date
            )
            results[user_id] = user_df

    if return_type == "json":
        json_result = {}
        for user_id, df in results.items():
            if not df.empty:
                json_result[str(user_id)] = df.to_dict(orient='records')
        return json.dumps(json_result, ensure_ascii=False, indent=2)
    elif return_type == "dict":
        dict_result = {}
        for user_id, df in results.items():
            if not df.empty:
                dict_result[user_id] = df.to_dict(orient='records')
        return dict_result
    else:
        return results


def get_user_record(conn, user_id: int, record_id: int) -> Optional[Dict]:
    """获取特定用户的某条记录"""
    sql = """
    SELECT mr.*
    FROM mood_records mr
    WHERE mr.user_id = ? AND mr.id = ?
    """

    df = pd.read_sql(sql, conn, params=(user_id, record_id))

    if not df.empty:
        return df.iloc[0].to_dict()
    return None


# ========== 新增：用户数据隔离函数 ==========
def load_user_data(conn, user_id: int) -> pd.DataFrame:
    """加载特定用户的数据"""
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    WHERE mr.user_id = ?
    ORDER BY mr.record_date DESC
    """

    try:
        df = pd.read_sql(sql, conn, params=(user_id,))

        # 确保日期类型正确
        if not df.empty and 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'])
        if not df.empty and 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        return df
    except Exception as e:
        st.error(f"加载用户数据失败: {e}")
        return pd.DataFrame()


# ========== 在这里插入哈希加密功能（功能4）==========
import hashlib


def calculate_data_signature(df: pd.DataFrame) -> str:
    """
    计算数据签名，用于验证数据完整性
    """
    if df.empty:
        return ""

    # 将DataFrame转换为字符串并计算哈希
    data_string = df.to_csv(index=False)
    signature = hashlib.sha256(data_string.encode()).hexdigest()

    return signature


def encrypt_sensitive_field(text: str, secret_key: str = "") -> str:
    """
    加密敏感字段（简化版，实际应用应使用更安全的加密）
    """
    if not text or not secret_key:
        return text

    # 使用HMAC进行消息认证
    h = hashlib.sha256()
    h.update(f"{text}{secret_key}".encode())
    return h.hexdigest()[:20]  # 返回部分哈希作为加密值


def verify_data_integrity(original_hash: str, current_df: pd.DataFrame) -> bool:
    """
    验证数据完整性
    """
    current_hash = calculate_data_signature(current_df)
    return original_hash == current_hash


def create_backup_with_verification(conn, backup_name: str) -> Dict:
    """
    创建带完整性验证的备份
    """
    from pathlib import Path

    # 确保备份目录存在
    BACKUP_DIR = Path("./backups")
    BACKUP_DIR.mkdir(exist_ok=True)

    backup_info = {
        'name': backup_name,
        'timestamp': datetime.now().isoformat(),
        'data_hash': '',
        'verification_passed': False,
        'backup_path': ''
    }

    try:
        # 获取所有数据并计算哈希
        all_data = pd.read_sql("SELECT * FROM mood_records", conn)
        backup_info['data_hash'] = calculate_data_signature(all_data)
        backup_info['record_count'] = len(all_data)

        # 保存备份文件
        backup_path = BACKUP_DIR / f"{backup_name}.db"
        backup_info['backup_path'] = str(backup_path)

        # 使用SQLite的备份功能
        with sqlite3.connect(backup_path) as backup_conn:
            conn.backup(backup_conn)

        # 验证备份
        with sqlite3.connect(backup_path) as backup_conn:
            backup_data = pd.read_sql("SELECT * FROM mood_records", backup_conn)
            backup_info['verification_passed'] = verify_data_integrity(
                backup_info['data_hash'],
                backup_data
            )

        # 记录备份信息
        backup_log_path = BACKUP_DIR / "backup_log.json"
        backup_log = []
        if backup_log_path.exists():
            with open(backup_log_path, 'r', encoding='utf-8') as f:
                backup_log = json.load(f)

        backup_log.append(backup_info)

        with open(backup_log_path, 'w', encoding='utf-8') as f:
            json.dump(backup_log, f, ensure_ascii=False, indent=2, default=str)

        return backup_info

    except Exception as e:
        st.error(f"备份创建失败: {e}")
        backup_info['error'] = str(e)
        return backup_info


def restore_from_backup(backup_path: str, conn) -> bool:
    """
    从备份恢复数据
    """
    try:
        # 验证备份文件
        with sqlite3.connect(backup_path) as backup_conn:
            backup_data = pd.read_sql("SELECT * FROM mood_records", backup_conn)

        if backup_data.empty:
            st.warning("备份文件为空")
            return False

        # 清空当前表
        conn.execute("DELETE FROM mood_records")

        # 恢复数据
        backup_conn = sqlite3.connect(backup_path)
        backup_conn.backup(conn)
        backup_conn.close()

        # 验证恢复的数据
        restored_data = pd.read_sql("SELECT * FROM mood_records", conn)
        if len(restored_data) == len(backup_data):
            st.success(f"成功恢复 {len(restored_data)} 条记录")
            return True
        else:
            st.error("数据恢复验证失败")
            return False

    except Exception as e:
        st.error(f"恢复失败: {e}")
        return False


# ========== 新增：管理员功能 ==========
def is_admin(username: str) -> bool:
    """检查是否为管理员（这里只是示例，实际需要更安全的认证）"""
    # 这里可以改成从配置文件或数据库读取管理员列表
    admins = ["栗子惠"]
    return username in admins


def get_all_users_data(conn) -> Dict[str, pd.DataFrame]:
    """获取所有用户的数据（仅管理员可用）"""
    sql = """
    SELECT 
        mr.*,
        u.username
    FROM mood_records mr
    JOIN users u ON mr.user_id = u.user_id
    ORDER BY u.username, mr.record_date DESC
    """

    try:
        df = pd.read_sql(sql, conn)

        if df.empty:
            return {}

        # 确保日期类型正确
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'])
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])

        # 按用户分组
        users_data = {}
        for user in df["username"].unique():
            user_df = df[df["username"] == user].copy()
            users_data[user] = user_df

        return users_data
    except Exception as e:
        st.error(f"获取所有用户数据失败: {e}")
        return {}


def get_user_stats(conn):
    """获取用户统计信息"""
    sql = """
    SELECT 
        u.username,
        COUNT(mr.id) as record_count,
        AVG(mr.mood_score) as avg_mood,
        MIN(mr.record_date) as first_record,
        MAX(mr.record_date) as last_record
    FROM users u
    LEFT JOIN mood_records mr ON u.user_id = mr.user_id
    GROUP BY u.username
    ORDER BY record_count DESC
    """

    return pd.read_sql(sql, conn)


# ========== 情绪分析工具 ==========
def analyze_mood_patterns(df: pd.DataFrame) -> Dict:
    """分析情绪模式"""
    if df.empty or 'mood_score' not in df.columns:
        return {}

    analysis = {
        "overall_score": round(df["mood_score"].mean(), 2) if not df.empty else 0,
        "trend": "上升" if len(df) > 1 and df["mood_score"].iloc[-1] > df["mood_score"].iloc[0] else "下降",
        "best_day": df.loc[df["mood_score"].idxmax(), "record_date"].strftime("%Y-%m-%d") if len(df) > 0 else None,
        "worst_day": df.loc[df["mood_score"].idxmin(), "record_date"].strftime("%Y-%m-%d") if len(df) > 0 else None,
        "consistency": round(df["mood_score"].std(), 2) if len(df) > 1 else 0
    }

    # 按星期分析
    if 'record_date' in df.columns and not df.empty:
        df["weekday"] = df["record_date"].dt.day_name()
        weekday_avg = df.groupby("weekday")["mood_score"].mean()
        if not weekday_avg.empty:
            analysis["best_weekday"] = weekday_avg.idxmax()

    return analysis


def detect_mood_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """检测情绪异常点"""
    if len(df) < 3 or 'mood_score' not in df.columns:
        return pd.DataFrame()

    scores = df["mood_score"].values
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    if std_score == 0:
        return pd.DataFrame()

    z_scores = np.abs((scores - mean_score) / std_score)
    anomalies = df[z_scores > threshold].copy()

    if not anomalies.empty:
        anomalies["z_score"] = z_scores[z_scores > threshold]
        anomalies["deviation"] = anomalies["mood_score"] - mean_score

    return anomalies


# 设置页面
st.set_page_config(
    page_title="MoodPattern — 情绪管理助手",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== 在这里插入路径管理功能（功能2）==========
# 路径配置
BASE_DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = BASE_DIR / "backups"
EXPORT_DIR = BASE_DIR / "exports"
LOG_DIR = BASE_DIR / "logs"

# 创建必要的目录
for directory in [DATA_DIR, BACKUP_DIR, EXPORT_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# 多数据库支持
def get_available_databases() -> List[Path]:
    """获取所有可用的数据库文件"""
    return list(DATA_DIR.glob("*.db"))


def create_new_database(db_name: str) -> Path:
    """创建新的数据库文件"""
    db_path = DATA_DIR / f"{db_name}.db"
    if not db_path.exists():
        conn = sqlite3.connect(db_path)
        # 初始化数据库结构
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("""
           CREATE TABLE IF NOT EXISTS users(
               user_id INTEGER PRIMARY KEY AUTOINCREMENT,
               username TEXT UNIQUE NOT NULL,
               password_hash TEXT NOT NULL,
               salt TEXT NOT NULL,  # ← 添加这一行
               email TEXT,
               is_admin INTEGER DEFAULT 0,
               created_at TEXT
           );
           """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS mood_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mood_score INTEGER CHECK (mood_score BETWEEN 1 AND 10),
            mood_label TEXT,
            activities TEXT,
            notes TEXT,
            sleep_hours REAL,
            stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
            tags TEXT,
            weather TEXT,
            record_date TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON mood_records(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_record_date ON mood_records(record_date);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mood_score ON mood_records(mood_score);")
        conn.commit()
        conn.close()
    return db_path


def manage_database_files():
    """管理数据库文件"""
    dbs = get_available_databases()
    if dbs:
        st.write("可用数据库文件:")
        for db in dbs:
            size = db.stat().st_size
            st.write(f"- {db.name} ({size:,} bytes)")
    else:
        st.info("暂无数据库文件")


# 设置默认数据库路径
DEFAULT_DB_PATH = DATA_DIR / "mood_system.db"

def init_database(db_path: Path = DEFAULT_DB_PATH):
    """初始化数据库 - 检查并创建必要表结构"""
    # 连接数据库
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON")

    # 创建表之前先检查表是否存在
    cursor = conn.cursor()

    # 检查用户表是否存在
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='users'
    """)
    users_table_exists = cursor.fetchone() is not None

    # 检查情绪记录表是否存在
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='mood_records'
    """)
    mood_records_table_exists = cursor.fetchone() is not None

    # 如果表不存在，才创建
    if not users_table_exists:
        conn.execute("""
        CREATE TABLE users(
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            email TEXT,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT
        );
        """)
        print("✅ 用户表已创建")

    if not mood_records_table_exists:
        conn.execute("""
        CREATE TABLE mood_records(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            mood_score INTEGER CHECK (mood_score BETWEEN 1 AND 10),
            mood_label TEXT,
            activities TEXT,
            notes TEXT,
            sleep_hours REAL,
            stress_level INTEGER CHECK (stress_level BETWEEN 1 AND 10),
            tags TEXT,
            weather TEXT,
            record_date TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """)

        # 创建索引
        conn.execute("CREATE INDEX idx_user_id ON mood_records(user_id);")
        conn.execute("CREATE INDEX idx_record_date ON mood_records(record_date);")
        conn.execute("CREATE INDEX idx_mood_score ON mood_records(mood_score);")
        print("✅ 情绪记录表已创建")

    # 检查是否缺少某些列（表存在但结构可能不完整）
    if users_table_exists:
        # 检查是否缺少 salt 列
        cursor.execute("PRAGMA table_info(users)")
        columns = [row[1] for row in cursor.fetchall()]

        if 'salt' not in columns:
            # 添加缺少的列
            conn.execute("ALTER TABLE users ADD COLUMN salt TEXT")
            print("✅ 为用户表添加 salt 列")

    # 创建备份日志表（如果不存在）
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='backup_logs'
    """)
    if cursor.fetchone() is None:
        conn.execute("""
        CREATE TABLE backup_logs(
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            backup_name TEXT,
            backup_time TEXT,
            record_count INTEGER,
            data_hash TEXT,
            verification_status INTEGER,
            backup_path TEXT
        );
        """)
        print("✅ 备份日志表已创建")

    conn.commit()
    return conn


# 情绪标签映射 - 纯数字
MOOD_LABELS = {
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10"
}


def create_initial_admin(conn):
    """创建初始管理员账户"""
    try:
        admin_check = conn.execute(
            "SELECT user_id FROM users WHERE username = '栗子惠'"
        ).fetchone()

        if not admin_check:
            # 创建管理员账户（默认密码：admin123）
            password_hash, salt = hash_password("admin123")

            conn.execute(
                """INSERT INTO users (username, password_hash, salt, is_admin, created_at) 
                VALUES (?, ?, ?, 1, datetime('now'))""",
                ("栗子惠", password_hash, salt)
            )
            conn.commit()
            print("初始管理员账户已创建：栗子惠/admin123")
    except Exception as e:
        print(f"创建初始管理员失败: {e}")


# ========== 主应用界面 ==========
def main():
    # 初始化数据库连接
    conn = init_database()

    def check_database_integrity(conn):
        """检查数据库完整性"""
        try:
            # 检查用户表数据
            user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            print(f"✅ 数据库检查：找到 {user_count} 个用户")

            # 检查情绪记录表数据
            record_count = conn.execute("SELECT COUNT(*) FROM mood_records").fetchone()[0]
            print(f"✅ 数据库检查：找到 {record_count} 条情绪记录")

            # 执行PRAGMA integrity_check
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
            if result == "ok":
                print("✅ 数据库完整性检查通过")
            else:
                print(f"⚠️ 数据库完整性警告：{result}")

        except Exception as e:
            print(f"⚠️ 数据库检查出错：{e}")

    # 创建初始管理员账户
    create_initial_admin(conn)

    #检查数据是否完整
    check_database_integrity(conn)

    # 侧边栏
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=80)
        st.title("🧠 MoodPattern")
        st.caption("你的情绪管理伙伴")


        # 管理员专属部分 - 保持简洁
        if 'current_user' in st.session_state and is_admin(st.session_state.current_user):
            st.divider()
            st.caption("👑 管理员模式已启用")
            # 可以完全移除 expander 内容，或保留少量关键信息
        # 用户管理部分
        st.divider()
        st.subheader("👤 用户管理")

        # 如果是新用户，显示注册表单
        if 'current_user' not in st.session_state:
            tab_login, tab_register = st.tabs(["🔐 登录", "📝 注册"])

            with tab_login:
                login_username = st.text_input("用户名", key="login_username")
                login_password = st.text_input("密码", type="password", key="login_password")

                if st.button("登录", type="primary"):
                    if login_username and login_password:
                        # 验证用户
                        user_check = conn.execute(
                            "SELECT user_id, password_hash, salt FROM users WHERE username = ?",
                            (login_username,)
                        ).fetchone()

                        if user_check and verify_password(login_password, user_check[1], user_check[2]):
                            st.session_state.current_user = login_username
                            st.session_state.user_id = user_check[0]
                            st.success(f"欢迎回来，{login_username}！")
                            st.rerun()
                        else:
                            st.error("用户名或密码错误")
                    else:
                        st.warning("请输入用户名和密码")

            with tab_register:
                reg_username = st.text_input("新用户名", key="reg_username")
                reg_password = st.text_input("设置密码", type="password", key="reg_password")
                reg_confirm = st.text_input("确认密码", type="password", key="reg_confirm")
                reg_email = st.text_input("邮箱（可选）", key="reg_email")

                if st.button("注册", type="secondary"):
                    if not reg_username:
                        st.error("请输入用户名")
                    elif not reg_password:
                        st.error("请设置密码")
                    elif reg_password != reg_confirm:
                        st.error("两次输入的密码不一致")
                    elif len(reg_password) < 6:
                        st.error("密码至少6位")
                    else:
                        # 检查用户名是否已存在
                        existing_user = conn.execute(
                            "SELECT user_id FROM users WHERE username = ?",
                            (reg_username,)
                        ).fetchone()

                        if existing_user:
                            st.error("用户名已存在")
                        else:
                            # 哈希密码
                            password_hash, salt = hash_password(reg_password)

                            # 插入新用户
                            conn.execute(
                                """INSERT INTO users (username, password_hash, salt, email, created_at) 
                                VALUES (?, ?, ?, ?, datetime('now'))""",
                                (reg_username, password_hash, salt, reg_email)
                            )
                            conn.commit()

                            # 获取新用户ID
                            new_user_id = conn.execute(
                                "SELECT user_id FROM users WHERE username = ?",
                                (reg_username,)
                            ).fetchone()[0]

                            st.session_state.current_user = reg_username
                            st.session_state.user_id = new_user_id
                            st.success(f"注册成功！欢迎使用MoodPattern")
                            st.rerun()

        # 显示当前用户
        if 'current_user' in st.session_state:
            st.divider()
            st.subheader("当前用户")
            st.success(f"👤 {st.session_state.current_user}")

            # 添加退出登录按钮
            if st.button("退出登录"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

            # 检查是否为管理员
            admin_mode = False
            if is_admin(st.session_state.current_user):
                st.success("👑 管理员模式已识别")
                admin_view = st.checkbox("管理员视图（查看所有用户）", value=False)
                if admin_view:
                    admin_mode = True
                    st.warning("⚠️ 正在查看所有用户数据")
                    st.session_state.admin_mode = True
                else:
                    if 'admin_mode' in st.session_state:
                        del st.session_state.admin_mode
            else:
                if 'admin_mode' in st.session_state:
                    del st.session_state.admin_mode

        # 分析设置
        st.divider()
        st.subheader("📊 分析设置")
        anomaly_threshold = st.slider(
            "异常检测灵敏度",
            1.0, 3.0, 2.0, 0.1,
            help="Z-score阈值，值越小越敏感"
        )

        # 数据统计
        st.divider()
        st.subheader("📦 数据统计")

        # 根据模式加载数据
        if 'current_user' in st.session_state:
            if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                # 管理员模式下显示所有用户数据
                all_users_data = get_all_users_data(conn)
                if all_users_data:
                    selected_user = st.selectbox(
                        "选择查看用户",
                        options=["所有用户"] + list(all_users_data.keys()),
                        index=0
                    )

                    if selected_user == "所有用户":
                        # 合并所有用户数据
                        all_dfs = []
                        for user, user_df in all_users_data.items():
                            all_dfs.append(user_df)
                        if all_dfs:
                            df = pd.concat(all_dfs, ignore_index=True)
                        else:
                            df = pd.DataFrame()
                    else:
                        df = all_users_data[selected_user]
                else:
                    df = pd.DataFrame()
                    st.info("暂无用户数据")
            else:
                # 普通用户模式下只加载自己的数据
                user_id = st.session_state.user_id
                df = load_user_data(conn, user_id)

            record_count = len(df) if not df.empty else 0
            st.metric("总记录数", record_count)

            if not df.empty and not ('admin_mode' in st.session_state and st.session_state.admin_mode):
                avg_mood = df["mood_score"].mean()
                st.metric("平均情绪", f"{avg_mood:.1f}/10")

        if st.button("🔄 刷新数据"):
            st.rerun()

    # 主界面
    st.title("🌈 MoodPattern — 情绪管理助手")

    # 检查是否已登录
    if 'current_user' not in st.session_state:
        st.info("请在左侧输入用户名开始使用")
        conn.close()
        return

    current_user = st.session_state.current_user
    user_id = st.session_state.user_id

    # 顶部状态栏
    if not df.empty and not ('admin_mode' in st.session_state and st.session_state.admin_mode):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            latest_mood = df.iloc[-1]["mood_score"] if not df.empty else 0
            st.metric("当前情绪", f"{latest_mood}/10", MOOD_LABELS.get(latest_mood, ""))
        with col2:
            streak = 0
            if not df.empty and 'record_date' in df.columns:
                dates = sorted(df["record_date"].unique())
                for i in range(1, min(7, len(dates)) + 1):
                    if (dates[-i].date() == (datetime.now().date() - timedelta(days=i - 1))):
                        streak += 1
                    else:
                        break
            st.metric("连续记录", f"{streak}天")
        with col3:
            if not df.empty and 'record_date' in df.columns:
                avg_week = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]["mood_score"].mean()
                st.metric("本周平均", f"{avg_week:.1f}/10" if not np.isnan(avg_week) else "--")
            else:
                st.metric("本周平均", "--")
        with col4:
            if not df.empty and 'record_date' in df.columns:
                last_record = df["record_date"].max()
                days_since = (datetime.now().date() - last_record.date()).days
                if days_since >= 3:
                    st.error(f"{days_since}天未记录")
                else:
                    st.success("记录正常")
            else:
                st.info("暂无记录")
    elif 'admin_mode' in st.session_state and st.session_state.admin_mode:
        st.info("👑 管理员视图：您可以查看和搜索所有用户的数据")

    # 标签页 - 修改这里增加Tab 7和Tab 8
    tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
        "📝 记录情绪",
        "📊 情绪分析",
        "🤖 AI助手",
        "📈 趋势",
        "⚙️ 设置",
        "🔧 管理",
        "🔐 安全中心"
    ])

    # Tab 1: 记录情绪
    with tab1:
        st.subheader("记录今日情绪")

        # 初始化录音相关的session state
        if 'voice_result' not in st.session_state:
            st.session_state.voice_result = None
        if 'audio_data' not in st.session_state:
            st.session_state.audio_data = None
        if 'transcribed_text' not in st.session_state:
            st.session_state.transcribed_text = ""

        # 选择记录方式
        record_method = st.radio(
            "🎯 选择记录方式",
            ["📝 文字输入", "🎤 语音输入", "📱 两者结合"],
            horizontal=True,
            key="record_method"
        )

        with st.form("mood_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])

            with col1:
                if record_method == "📝 文字输入":
                    notes = st.text_area(
                        "📔 今日心情日记",
                        placeholder="写下今天的感受、发生的事情、想法...",
                        height=150,
                        help="详细记录有助于更好的分析和回顾",
                        key="manual_text_area"
                    )

                elif record_method == "🎤 语音输入":
                    # 您需要确保前面有这些session state初始化
                    if 'voice_result' not in st.session_state:
                        st.session_state.voice_result = None
                    if 'audio_data' not in st.session_state:
                        st.session_state.audio_data = None
                    if 'transcribed_text' not in st.session_state:
                        st.session_state.transcribed_text = ""

                    # 然后粘贴您提供的代码
                    st.markdown("### 🎤 语音输入")

                    # 显示当前语音识别结果
                    if st.session_state.voice_result:
                        st.success(f"✅ 已有识别内容：{st.session_state.voice_result[:50]}...")
                        if st.button("🗑️ 清除语音内容", key="clear_existing_voice"):
                            st.session_state.voice_result = None
                            st.session_state.audio_data = None
                            st.session_state.transcribed_text = ""
                            st.rerun()

                    # 录音组件
                    st.markdown("#### 步骤1：录音")
                    audio_data = mic_recorder(
                        start_prompt="🎤 长按开始录音",
                        stop_prompt="⏹️ 松开结束录音",
                        key="wechat_recorder",
                        format="wav",
                        just_once=False,
                        use_container_width=True
                    )

                    # 录音提示
                    st.markdown('<p style="color: #666; font-size: 14px;">💡 提示：长按按钮说话，松开结束</p>',
                                unsafe_allow_html=True)

                    # 如果检测到录音数据
                    if audio_data is not None:
                        # 保存到session state
                        st.session_state.audio_data = audio_data
                        st.success("✅ 录音完成！请点击刷新按钮")

                        # 强制刷新按钮
                        if st.button("🔄 刷新界面显示录音", key="refresh_audio"):
                            st.rerun()

                    # 如果有录音数据，显示处理选项
                    if 'audio_data' in st.session_state and st.session_state.audio_data is not None:
                        st.markdown("#### 步骤2：处理录音")

                        # 显示音频
                        audio_bytes = st.session_state.audio_data.get('bytes', b'')
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/wav")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                if st.button("🔤 转成文字", key="transcribe_audio", type="primary"):
                                    with st.spinner("正在识别语音..."):
                                        text = transcribe_audio_file(audio_bytes)

                                        if text and "失败" not in text and "无法识别" not in text:
                                            st.session_state.transcribed_text = text
                                            st.session_state.voice_result = text
                                            st.success("✅ 识别成功！")
                                            # 自动刷新显示结果
                                            st.rerun()
                                        else:
                                            st.error(f"识别失败: {text}")
                                            st.session_state.transcribed_text = ""

                            with col2:
                                if st.button("🗑️ 清除录音", key="clear_audio"):
                                    st.session_state.audio_data = None
                                    st.rerun()

                            with col3:
                                if st.button("🎤 重新录音", key="re_record"):
                                    st.session_state.audio_data = None
                                    st.rerun()

                    # 文本输入框
                    if st.session_state.voice_result:
                        st.markdown("#### 步骤3：编辑结果")
                        notes = st.text_area(
                            "📝 编辑识别结果",
                            value=st.session_state.voice_result,
                            height=150,
                            key="voice_text_area_edit"
                        )
                    else:
                        # 如果还在等待，显示提示
                        if 'audio_data' not in st.session_state or st.session_state.audio_data is None:
                            st.info("👆 请先录音，然后点击刷新按钮")
                            notes = st.text_area(
                                "等待录音...",
                                placeholder="请先录音并转文字",
                                height=150,
                                key="waiting_voice_area",
                                disabled=True
                            )
                        else:
                            notes = st.text_area(
                                "录音已就绪，请转文字",
                                placeholder="点击'转成文字'按钮",
                                height=150,
                                key="ready_voice_area",
                                disabled=True
                            )
                elif record_method == "📱 两者结合":
                    # 文字部分
                    st.markdown("### 📝 文字记录部分")
                    text_part = st.text_area(
                        "先写下你的感受...",
                        placeholder="在这里输入文字记录...",
                        height=100,
                        key="text_part_area"
                    )

                    # 语音部分状态
                    if st.session_state.voice_result:
                        st.success(f"✅ 已有语音内容：{st.session_state.voice_result[:30]}...")
                        if st.button("🗑️ 清除语音", key="clear_combined_voice"):
                            st.session_state.voice_result = None
                            st.rerun()

                    # 语音补充录音
                    st.markdown("### 🎤 语音补充")

                    # 录音组件CSS
                    st.markdown("""
                    <style>
                    .record-instruction {
                        text-align: center;
                        color: #666;
                        margin-top: 10px;
                        font-size: 14px;
                    }
                    .recording-status {
                        padding: 10px;
                        border-radius: 5px;
                        margin: 10px 0;
                        text-align: center;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    # 使用streamlit-mic-recorder
                    audio_data_combined = mic_recorder(
                        start_prompt="🎤 长按录音补充",
                        stop_prompt="⏹️ 松开结束",
                        key="wechat_recorder_combined",
                        format="wav",
                        just_once=False,
                        use_container_width=True
                    )

                    # 录音提示
                    st.markdown('<p class="record-instruction">💡 长按录音补充，松开结束</p>', unsafe_allow_html=True)

                    # 处理录音结果
                    if audio_data_combined and 'bytes' in audio_data_combined and audio_data_combined['bytes']:
                        # 显示录音
                        st.audio(audio_data_combined['bytes'], format="audio/wav")

                        # 转换按钮
                        if st.button("🔤 转换语音补充", key="transcribe_combined"):
                            with st.spinner("正在识别语音补充..."):
                                text = transcribe_audio_file(audio_data_combined['bytes'])

                                if text and "失败" not in text and "无法识别" not in text:
                                    st.session_state.voice_result = text
                                    st.success("✅ 语音补充识别成功！")
                                    st.rerun()
                                else:
                                    st.error(f"识别失败: {text}")

                    # 合并结果
                    combined_text = text_part
                    if st.session_state.voice_result:
                        combined_text += f"\n\n【语音补充】\n{st.session_state.voice_result}"

                    notes = st.text_area(
                        "📋 合并后的内容",
                        value=combined_text,
                        height=150,
                        key="combined_text_area"
                    )

                with col2:
                    # 情绪评分滑块 - 修复版
                    mood_score = st.slider(
                        "情绪分数",
                        1, 10, 5,
                        help="1=非常低落, 10=非常开心",
                        key="mood_score_main"  # 确保key唯一
                    )


                    # 直接硬编码映射，避免任何可能的变量覆盖
                    MOOD_LABELS_FIXED = {
                        1: "非常低落",
                        2: "低落",
                        3: "有点低落",
                        4: "轻微低落",
                        5: "平静",
                        6: "轻微愉悦",
                        7: "愉悦",
                        8: "开心",
                        9: "非常开心",
                        10: "兴奋"
                    }

                    mood_label = MOOD_LABELS_FIXED.get(mood_score, "未知")
                    st.markdown(f"### {mood_label}")

                    st.divider()

                # 睡眠时长
                sleep_hours = st.slider(
                    "😴 睡眠时长(小时)",
                    0.0, 12.0, 7.0, 0.5,
                    help="昨晚睡了多久？"
                )

                # 睡眠质量提示
                if sleep_hours < 6:
                    st.warning("😴 睡眠不足，注意休息")
                elif sleep_hours > 9:
                    st.info("😴 睡得不错")

                st.divider()

                # 压力水平
                stress_level = st.slider(
                    "💼 压力水平",
                    1, 10, 5,
                    help="1=无压力, 10=压力极大"
                )

                # 压力提示
                if stress_level >= 8:
                    st.warning("😰 压力较大，记得放松")
                elif stress_level <= 3:
                    st.success("😌 状态很放松")

            # 其他表单元素
            st.divider()

            col_extra1, col_extra2 = st.columns(2)

            with col_extra1:
                # 天气选择
                weather = st.selectbox(
                    "☁️ 天气",
                    ["", "☀️ 晴天", "⛅ 多云", "🌧️ 雨天", "❄️ 雪天", "💨 大风", "🌫️ 雾天", "其他"],
                    help="今天的天气如何？"
                )

                # 活动选择
                activities = st.multiselect(
                    "🏃 今日活动",
                    ["工作", "学习", "运动", "社交", "娱乐", "休息", "家务", "购物", "旅行", "其他"],
                    help="选择今天的活动"
                )

            with col_extra2:
                # 标签选择
                tags = st.multiselect(
                    "🏷️ 标签",
                    ["重要事件", "突破", "挑战", "放松", "思考", "成就", "感恩", "困惑", "成长", "启发"],
                    help="给今天贴个标签吧"
                )

                # 自定义标签
                custom_tag = st.text_input(
                    "自定义标签",
                    placeholder="输入自定义标签...",
                    help="输入你的独特标签"
                )

                if custom_tag and custom_tag.strip():
                    if tags is None:
                        tags = []
                    tags.append(custom_tag.strip())

            st.divider()

            # 预览区域（可选）
            with st.expander("📋 记录预览", expanded=False):
                if notes and notes.strip():
                    st.write("**心情日记：**")
                    st.write(notes[:200] + ("..." if len(notes) > 200 else ""))

                st.write(f"**情绪分数：** {mood_score} ({mood_label})")
                st.write(f"**睡眠时长：** {sleep_hours}小时")
                st.write(f"**压力水平：** {stress_level}/10")

                if weather:
                    st.write(f"**天气：** {weather}")

                if activities:
                    st.write(f"**活动：** {', '.join(activities)}")

                if tags:
                    st.write(f"**标签：** {', '.join(tags)}")

            # ========== 表单提交按钮 ==========
            submitted = st.form_submit_button("💾 保存记录", type="primary", use_container_width=True)

            if submitted:
                # 获取情绪标签
                mood_label = MOOD_LABELS.get(mood_score, "")

                # 处理笔记内容
                final_notes = notes.strip() if notes else ""

                # 如果是语音输入模式，优先使用session state中的结果
                if record_method == "🎤 语音输入" and st.session_state.voice_result:
                    final_notes = st.session_state.voice_result
                elif record_method == "📱 两者结合":
                    text_part_value = text_part.strip() if text_part else ""
                    if st.session_state.voice_result:
                        final_notes = f"{text_part_value}\n\n【语音补充】\n{st.session_state.voice_result}"
                    else:
                        final_notes = text_part_value

                # 验证是否有内容
                if not final_notes or final_notes.strip() == "":
                    st.warning("⚠️ 请填写心情日记内容")
                    st.stop()

                # 处理标签
                all_tags = []
                if tags:
                    all_tags.extend(tags)

                # 插入记录
                sql = """
                    INSERT INTO mood_records(
                        user_id, mood_score, mood_label, activities, notes, 
                        sleep_hours, stress_level, tags, weather, record_date, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, date('now'), datetime('now'))
                    """

                params = (
                    user_id,
                    mood_score,
                    mood_label,
                    ", ".join(activities) if activities else "",
                    final_notes,
                    sleep_hours,
                    stress_level,
                    ", ".join(all_tags) if all_tags else "",
                    weather if weather else ""
                )

                try:
                    conn.execute(sql, params)
                    conn.commit()

                    st.success("🎉 记录已保存！")
                    st.balloons()

                    # 根据情绪分数显示不同的反馈
                    if mood_score >= 8:
                        st.info("✨ 继续保持好心情！今天的你很棒！")
                    elif mood_score <= 4:
                        st.info("💙 感谢你记录下这些感受。无论情绪如何，都是真实的你。")
                    else:
                        st.info("📝 记录完成！回头看看这些记录，会发现自己的成长。")

                    # 清空session state
                    keys_to_clear = ['voice_result', 'audio_data', 'transcribed_text']
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]

                except Exception as e:
                    st.error(f"❌ 保存失败: {str(e)}")
                    st.info("请检查数据格式或联系管理员")

        # 表单外的额外功能
        st.divider()

        # 快速记录选项（小功能）
        with st.expander("⚡ 快速记录（跳过详细表单）", expanded=False):
            quick_mood = st.slider("快速情绪评分", 1, 10, 5, key="quick_mood")
            quick_notes = st.text_area("快速备注", placeholder="简单记录...", height=60, key="quick_notes")

            if st.button("快速保存", key="quick_save"):
                if quick_notes and quick_notes.strip():
                    sql = """
                        INSERT INTO mood_records(
                            user_id, mood_score, mood_label, notes, record_date, created_at
                        ) VALUES (?, ?, ?, ?, date('now'), datetime('now'))
                        """
                    conn.execute(sql, (
                        user_id,
                        quick_mood,
                        MOOD_LABELS.get(quick_mood, ""),
                        quick_notes
                    ))
                    conn.commit()
                    st.success("✅ 快速记录已保存！")
                    st.rerun()
                else:
                    st.warning("请填写备注内容")


    # Tab 2: 情绪分析
    with tab2:
        st.subheader("情绪分析报告")

        if df.empty:
            st.info("📝 还没有记录，先去记录一下吧！")
        else:
            # 选择分析范围
            period = st.radio(
                "分析时段",
                ["最近7天", "最近30天", "全部记录"],
                horizontal=True
            )

            if period == "最近7天":
                analysis_df = df[df["record_date"] >= (datetime.now() - timedelta(days=7))]
            elif period == "最近30天":
                analysis_df = df[df["record_date"] >= (datetime.now() - timedelta(days=30))]
            else:
                analysis_df = df

            if analysis_df.empty:
                st.warning("该时段暂无记录")
            else:
                # 基本统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("平均情绪", f"{analysis_df['mood_score'].mean():.1f}")
                with col2:
                    st.metric("最高情绪", f"{analysis_df['mood_score'].max():.0f}")
                with col3:
                    st.metric("最低情绪", f"{analysis_df['mood_score'].min():.0f}")

                # 情绪趋势图
                st.subheader("📈 情绪趋势")
                fig, ax = plt.subplots(figsize=(10, 4))
                analysis_df_sorted = analysis_df.sort_values("record_date")
                ax.plot(analysis_df_sorted["record_date"], analysis_df_sorted["mood_score"],
                        marker='o', linewidth=2, markersize=6)
                ax.axhline(y=analysis_df_sorted["mood_score"].mean(), color='r',
                           linestyle='--', alpha=0.5, label=f"平均线 ({analysis_df_sorted['mood_score'].mean():.1f})")
                ax.set_xlabel("Date")
                ax.set_ylabel("Mood Score")
                ax.set_ylim(0, 10.5)
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

                # 情绪分布
                st.subheader("📊 情绪分布")
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                bins = np.arange(1, 12) - 0.5
                ax2.hist(analysis_df_sorted["mood_score"], bins=bins,
                         edgecolor='black', alpha=0.7)
                ax2.set_xlabel("Mood Score")
                ax2.set_ylabel("Frequency")
                ax2.set_xticks(range(1, 11))
                st.pyplot(fig2)

                # 异常检测
                st.subheader("🔍 情绪异常检测")
                anomalies = detect_mood_anomalies(analysis_df_sorted, anomaly_threshold)
                if not anomalies.empty:
                    st.warning(f"检测到 {len(anomalies)} 个异常情绪点")
                    st.dataframe(
                        anomalies[["record_date", "mood_score", "notes", "z_score"]].sort_values("z_score",
                                                                                                 ascending=False),
                        use_container_width=True
                    )
                else:
                    st.success("情绪波动正常")

                # 活动关联分析
                if "activities" in analysis_df_sorted.columns:
                    st.subheader("🏃 活动与情绪关联")
                    activity_data = []
                    for _, row in analysis_df_sorted.iterrows():
                        if pd.notna(row["activities"]) and row["activities"].strip():
                            activities = [a.strip() for a in str(row["activities"]).split(",")]
                            for activity in activities:
                                if activity:
                                    activity_data.append({"activity": activity, "mood": row["mood_score"]})

                    if activity_data:
                        activity_df = pd.DataFrame(activity_data)
                        activity_stats = activity_df.groupby("activity")["mood"].agg(["mean", "count"]).round(2)
                        activity_stats = activity_stats[activity_stats["count"] >= 2]  # 至少出现2次

                        if not activity_stats.empty:
                            st.dataframe(
                                activity_stats.sort_values("mean", ascending=False),
                                use_container_width=True
                            )

                # 记录查询功能
                st.subheader("🔍 记录查询")

                with st.expander("高级查询", expanded=False):
                    col1, col2 = st.columns(2)

                    with col1:
                        # 如果是管理员，可以选择用户
                        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                            query_user = st.selectbox(
                                "查询用户",
                                options=["所有用户"] + list(df["username"].unique()) if 'username' in df.columns else [
                                    "所有用户"],
                                index=0
                            )
                        else:
                            query_user = st.text_input("查询用户", value=current_user)

                        query_start = st.date_input("开始日期",
                                                    value=datetime.now().date() - timedelta(days=30))
                        query_end = st.date_input("结束日期",
                                                  value=datetime.now().date())

                    with col2:
                        query_min = st.slider("最低分数", 1, 10, 1)
                        query_max = st.slider("最高分数", 1, 10, 10)
                        keyword = st.text_input("关键词搜索",
                                                placeholder="在备注/活动/标签中搜索")

                    if st.button("执行查询", type="secondary"):
                        # 处理用户查询条件
                        user_filter_id = None
                        user_filter_name = None

                        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
                            if query_user and query_user != "所有用户":
                                user_filter_name = query_user
                        else:
                            user_filter_id = user_id

                        query_result = query_records(
                            conn,
                            user_id=user_filter_id,
                            username=user_filter_name,
                            start_date=datetime.combine(query_start, datetime.min.time()),
                            end_date=datetime.combine(query_end, datetime.max.time()),
                            min_score=query_min,
                            max_score=query_max,
                            keyword=keyword
                        )

                        if query_result.empty:
                            st.info("未找到匹配的记录")
                        else:
                            st.success(f"找到 {len(query_result)} 条记录")

                            # 显示结果
                            display_cols = ["record_date", "username", "mood_score", "mood_label",
                                            "activities", "notes", "tags"]
                            # 确保列存在
                            available_cols = [col for col in display_cols if col in query_result.columns]
                            display_df = query_result[available_cols].copy()

                            # 格式化日期
                            if "record_date" in display_df.columns:
                                display_df["record_date"] = pd.to_datetime(display_df["record_date"]).dt.strftime(
                                    "%Y-%m-%d %H:%M")

                            # 分页显示
                            page_size = 10
                            total_pages = max(1, len(display_df) // page_size + (
                                1 if len(display_df) % page_size > 0 else 0))
                            page = st.number_input("页码", min_value=1, max_value=total_pages, value=1)

                            start_idx = (page - 1) * page_size
                            end_idx = min(start_idx + page_size, len(display_df))

                            st.dataframe(
                                display_df.iloc[start_idx:end_idx],
                                use_container_width=True,
                                hide_index=True
                            )

                            # 导出选项
                            csv_data = query_result.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 导出查询结果",
                                csv_data,
                                file_name=f"查询结果_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

    # Tab 3: AI助手
    with tab3:
        st.subheader("🤖 AI情绪助手")

        # 检查AI服务状态
        if client is None:
            st.info("💡 请在侧边栏配置并连接AI服务")
        else:
            st.success("✅ AI助手已就绪")

            # AI功能选择
            ai_function = st.radio(
                "选择AI功能",
                ["情绪分析报告", "周度情绪总结", "个性化对话"],
                horizontal=True
            )

            if df.empty:
                st.warning("暂无记录可供分析")
            else:
                if ai_function == "情绪分析报告":
                    st.markdown("#### 📋 情绪综合分析")
                    st.caption("基于你的所有记录，AI将提供全面的情绪分析和建议")

                    if st.button("生成情绪分析报告", type="primary"):
                        with st.spinner("AI正在分析你的情绪数据..."):
                            result = ai_explain_mood(df)

                        st.markdown("---")
                        st.markdown("### 🧠 AI情绪分析报告")
                        st.markdown(result)

                        # 下载选项
                        st.download_button(
                            "📥 下载报告",
                            result,
                            file_name=f"情绪分析报告_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )

                elif ai_function == "周度情绪总结":
                    st.markdown("#### 📅 本周情绪总结")
                    st.caption("分析最近7天的情绪变化和模式")

                    if st.button("生成周度总结", type="primary"):
                        with st.spinner("AI正在生成周报..."):
                            result = ai_generate_weekly_report(df)

                        st.markdown("---")
                        st.markdown("### 📊 本周情绪周报")
                        st.markdown(result)

                elif ai_function == "个性化对话":
                    st.markdown("#### 💬 与AI情绪教练对话")
                    st.caption("可以询问任何与情绪、压力、心理健康相关的问题")

                    user_question = st.text_area(
                        "你想聊什么？",
                        placeholder="例如：\n• 最近压力很大怎么办？\n• 如何保持积极心态？\n• 情绪低落时可以做些什么？",
                        height=100
                    )

                    if st.button("发送问题", type="primary") and user_question:
                        with st.spinner("AI正在思考..."):
                            # 构建更专业的系统提示
                            system_prompt = """你是一名专业的心理情绪教练，拥有丰富的情绪管理和心理健康知识。
    你的回答应该：
    1. 温暖、支持、非评判性
    2. 提供具体、可操作的建议
    3. 基于科学心理学原理
    4. 用普通人能理解的语言
    5. 鼓励积极改变和成长"""

                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_question}
                            ]

                            response = ask_ai(messages, json_type=False)

                        st.markdown("---")
                        st.markdown("### 🤖 AI回复")
                        st.markdown(response)

    # Tab 4: 趋势
    with tab4:
        st.subheader("长期趋势分析")

        if len(df) < 7:
            st.info("需要更多记录来显示趋势分析")
        else:
            # 周趋势
            df["week"] = df["record_date"].dt.isocalendar().week
            weekly_avg = df.groupby("week")["mood_score"].mean()

            # 月趋势
            df["month"] = df["record_date"].dt.to_period("M").astype(str)
            monthly_avg = df.groupby("month")["mood_score"].mean()

            col1, col2 = st.columns(2)
            with col1:
                st.line_chart(weekly_avg)
                st.caption("周平均情绪趋势")
            with col2:
                st.line_chart(monthly_avg)
                st.caption("月平均情绪趋势")

            # 相关性分析
            if 'sleep_hours' in df.columns and 'stress_level' in df.columns:
                st.subheader("🔗 因素关联分析")
                numeric_cols = ["mood_score", "sleep_hours", "stress_level"]
                numeric_df = df[numeric_cols].dropna()

                if not numeric_df.empty:
                    corr_df = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
                    ax.set_xticks(range(len(corr_df.columns)))
                    ax.set_yticks(range(len(corr_df.columns)))
                    ax.set_xticklabels(corr_df.columns, rotation=45)
                    ax.set_yticklabels(corr_df.columns)

                    # 添加数值标签
                    for i in range(len(corr_df.columns)):
                        for j in range(len(corr_df.columns)):
                            text = ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}",
                                           ha="center", va="center", color="black")

                    plt.colorbar(im)
                    st.pyplot(fig)
                    st.caption("情绪与其他因素的相关性（颜色越暖正相关越强）")

    # Tab 5: 设置
    with tab5:
        st.subheader("设置与导出")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 数据导出")

            # CSV导出
            if not df.empty:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "导出CSV",
                    csv_data,
                    file_name=f"mood_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

            # 完整报告导出
            if st.button("生成完整报告", type="primary"):
                with st.spinner("生成报告中..."):
                    # 创建ZIP文件
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # 添加CSV
                        zip_file.writestr("mood_records.csv", df.to_csv(index=False))

                        # 添加总结文本
                        summary = f"""MoodPattern 数据报告
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    记录总数: {len(df)}
    """
                        if not df.empty and 'record_date' in df.columns:
                            summary += f"""时间范围: {df['record_date'].min().strftime('%Y-%m-%d')} 至 {df['record_date'].max().strftime('%Y-%m-%d')}
    平均情绪: {df['mood_score'].mean():.2f}/10
    情绪波动: {df['mood_score'].std():.2f}
    """
                        else:
                            summary += "暂无详细数据统计"

                        zip_file.writestr("summary.txt", summary)

                        # 添加图表
                        if not df.empty and 'record_date' in df.columns:
                            # 趋势图
                            fig, ax = plt.subplots(figsize=(10, 5))
                            df_sorted = df.sort_values("record_date")
                            ax.plot(df_sorted["record_date"], df_sorted["mood_score"], marker='o')
                            ax.set_title("情绪趋势图")
                            ax.set_xlabel("日期")
                            ax.set_ylabel("情绪分数")
                            ax.grid(True, alpha=0.3)

                            img_buffer = io.BytesIO()
                            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                            zip_file.writestr("trend_chart.png", img_buffer.getvalue())
                            plt.close(fig)

                    zip_buffer.seek(0)

                    st.download_button(
                        "📥 下载报告ZIP",
                        zip_buffer.getvalue(),
                        file_name=f"mood_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

        with col2:
            st.subheader("⚠️ 数据管理")

            # 数据清理（只删除当前用户数据）
            if st.button("清除我的记录", type="secondary"):
                if st.checkbox("确认永久删除所有记录"):
                    conn.execute("DELETE FROM mood_records WHERE user_id = ?", (user_id,))
                    conn.commit()
                    st.success("所有记录已清除")
                    st.rerun()

            st.divider()

            # 关于
            st.subheader("ℹ️ 关于")
            st.markdown("""
                **MoodPattern — 情绪管理助手**

                一个专注情绪管理与心理健康的工具。

                功能特点：
                - 📝 情绪日记记录
                - 📊 数据可视化分析
                - 🤖 AI智能建议（基于讯飞星辰API）
                - 🔍 模式识别
                - 💾 数据库存储
                - 📤 数据导出

                **AI功能说明：**
                使用讯飞星辰API提供智能情绪分析和建议。
                请在侧边栏配置API信息后使用。
                """)

    # Tab 6: 管理
    with tab6:
        st.subheader("🔧 系统管理")

        # 管理员功能
        if 'admin_mode' in st.session_state and st.session_state.admin_mode:
            st.info("👑 管理员管理面板")

            # 用户统计
            st.subheader("📊 用户统计")
            if st.button("显示用户统计"):
                stats_df = get_user_stats(conn)
                if not stats_df.empty:
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("暂无用户数据")

            # 批量操作功能
            st.subheader("🔄 批量操作")

            # 获取所有用户ID
            all_users = pd.read_sql("SELECT user_id, username FROM users", conn)

            if not all_users.empty:
                # 批量查询
                st.write("批量查询用户数据:")
                selected_user_ids = st.multiselect(
                    "选择用户",
                    options=[f"{row['user_id']} - {row['username']}" for _, row in all_users.iterrows()]
                )

                if selected_user_ids and st.button("执行批量查询"):
                    user_ids = [int(uid.split(" - ")[0]) for uid in selected_user_ids]

                    start_date = st.date_input("开始日期",
                                               value=datetime.now().date() - timedelta(days=30),
                                               key="batch_start")
                    end_date = st.date_input("结束日期",
                                             value=datetime.now().date(),
                                             key="batch_end")

                    results = batch_query_records(
                        conn,
                        user_ids=user_ids,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.max.time()),
                        return_type="dataframe"
                    )

                    if results:
                        total_records = sum(len(df) for df in results.values() if not df.empty)
                        st.success(f"批量查询完成，共获取 {total_records} 条记录")

                        # 显示每个用户的结果摘要
                        for user_id, user_df in results.items():
                            if not user_df.empty:
                                username = all_users[all_users['user_id'] == user_id]['username'].iloc[0]
                                with st.expander(f"用户: {username} ({len(user_df)} 条记录)"):
                                    st.dataframe(user_df.head(10), use_container_width=True)

            # 记录管理
            st.subheader("📝 记录管理")
            col1, col2 = st.columns(2)

            with col1:
                # 更新记录
                update_id = st.number_input("更新记录ID", min_value=1, step=1)
                if update_id:
                    # 获取记录详情
                    record_df = pd.read_sql("""
                        SELECT mr.*, u.username 
                        FROM mood_records mr 
                        JOIN users u ON mr.user_id = u.user_id 
                        WHERE mr.id = ?
                        """, conn, params=(update_id,))

                    if not record_df.empty:
                        record = record_df.iloc[0]
                        st.write(f"当前记录：用户={record['username']}, 分数={record['mood_score']}")

                        new_mood = st.slider("新情绪值", 1, 10, record['mood_score'], key="update_mood")
                        new_notes = st.text_input("新备注", value=record['notes'] if record['notes'] else "",
                                                  key="update_notes")

                        if st.button("更新记录"):
                            conn.execute("""
                                UPDATE mood_records 
                                SET mood_score = ?, notes = ?, created_at = datetime('now')
                                WHERE id = ?
                                """, (new_mood, new_notes, update_id))
                            conn.commit()
                            st.success("记录更新成功！")
                            st.rerun()

            with col2:
                # 删除记录
                delete_id = st.number_input("删除记录ID", min_value=1, step=1, key="admin_delete")
                if st.button("删除记录"):
                    conn.execute("DELETE FROM mood_records WHERE id = ?", (delete_id,))
                    conn.commit()
                    st.success("记录删除成功！")
                    st.rerun()

            # 数据库维护
            st.subheader("🛠️ 数据库维护")
            if st.button("优化数据库"):
                conn.execute("VACUUM")
                conn.commit()
                st.success("数据库优化完成")

            if st.button("导出完整数据库"):
                # 导出整个数据库
                db_data = conn.cursor().execute("SELECT * FROM mood_records").fetchall()
                df_all = pd.DataFrame(db_data, columns=[desc[0] for desc in conn.cursor().description])
                csv_data = df_all.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "📥 导出完整数据库",
                    csv_data,
                    file_name=f"mood_database_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("🔒 仅管理员可访问此页面")


    # Tab 7: 安全中心（功能7）
    with tab7:
        st.subheader("🔐 安全中心")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🛡️ 数据保护")

            # 数据加密
            st.write("**数据加密设置**")
            encryption_key = st.text_input("加密密钥", type="password",
                                           help="用于加密敏感数据的密钥")

            if encryption_key:
                test_data = st.text_area("测试加密数据",
                                         placeholder="输入要加密的测试数据")
                if test_data and st.button("测试加密"):
                    encrypted = encrypt_sensitive_field(test_data, encryption_key)
                    st.code(f"加密结果: {encrypted}")

            # 数据完整性验证
            st.write("**数据完整性**")
            if st.button("验证所有数据完整性"):
                # 计算当前数据的哈希
                current_hash = calculate_data_signature(df)

                # 从数据库获取原始哈希（这里简化处理，实际应从备份或日志中获取）
                try:
                    backup_log_path = BACKUP_DIR / "backup_log.json"
                    if backup_log_path.exists():
                        with open(backup_log_path, 'r', encoding='utf-8') as f:
                            backup_log = json.load(f)

                        if backup_log:
                            latest_backup = backup_log[-1]
                            original_hash = latest_backup.get('data_hash', '')

                            if original_hash and verify_data_integrity(original_hash, df):
                                st.success("✅ 数据完整性验证通过")
                            else:
                                st.error("❌ 数据完整性验证失败")
                        else:
                            st.info("暂无备份记录")
                    else:
                        st.info("暂无备份文件")
                except Exception as e:
                    st.error(f"验证出错: {e}")

        with col2:
            st.subheader("💾 备份管理")

            # 创建备份
            backup_name = st.text_input("备份名称",
                                        value=f"backup_{datetime.now().strftime('%Y%m%d_%H%M')}")

            if st.button("创建备份", type="primary"):
                with st.spinner("正在创建备份..."):
                    backup_info = create_backup_with_verification(conn, backup_name)

                    if backup_info.get('verification_passed'):
                        st.success(f"✅ 备份创建成功！")
                        st.json(backup_info)
                    else:
                        st.error("备份创建失败或验证未通过")

            # 查看备份列表
            st.write("**备份列表**")
            try:
                backup_log_path = BACKUP_DIR / "backup_log.json"

                if backup_log_path.exists():
                    with open(backup_log_path, 'r', encoding='utf-8') as f:
                        backup_log = json.load(f)

                    if backup_log:
                        for i, backup in enumerate(reversed(backup_log[-5:]), 1):

                            with st.expander(f"备份 {i}: {backup.get('name', '未知')}"):
                                st.write(f"时间: {backup.get('timestamp', '未知')}")
                                st.write(f"记录数: {backup.get('record_count', 0)}")
                                st.write(f"验证状态: {'✅ 通过' if backup.get('verification_passed') else '❌ 失败'}")

                                # 放到 expander 内（每个备份都有按钮）
                                backup_path = backup.get('backup_path', '')
                                if backup_path and st.button(f"恢复此备份", key=f"restore_{i}"):
                                    if restore_from_backup(backup_path, conn):
                                        st.success("恢复成功！请刷新页面查看最新数据")
                                        st.rerun()
                                    else:
                                        st.error("恢复失败")

                    else:
                        st.info("暂无备份记录")

                else:
                    st.info("暂无备份文件")

            except Exception as e:
                st.error(f"加载备份时出错：{e}")


    # 最后关闭数据库连接
    st.divider()
    if st.button("关闭数据库连接"):
        conn.close()
        st.success("数据库连接已关闭")


if __name__ == "__main__":
    main()
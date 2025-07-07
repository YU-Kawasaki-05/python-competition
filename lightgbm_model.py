# NFL Draft Prediction - LightGBMモデル実装

# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

# 設定
warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', None)

print("=== NFL Draft Prediction - LightGBMモデル実装開始 ===")

# 既存の特徴量エンジニアリング関数をインポート
def create_basic_features(df):
    """基本的な特徴量を作成"""
    df_new = df.copy()
    
    # 1. BMI（Body Mass Index）
    df_new['BMI'] = df_new['Weight'] / (df_new['Height'] ** 2)
    
    # 2. パワー・ウェイト・レシオ
    df_new['Power_Weight_Ratio'] = df_new['Bench_Press_Reps'] / df_new['Weight']
    
    # 3. スピード・パワー・インデックス（垂直跳び / 40ヤード走）
    df_new['Speed_Power_Index'] = df_new['Vertical_Jump'] / df_new['Sprint_40yd']
    
    # 4. アジリティ・スコア（3cone + shuttle）
    df_new['Agility_Score'] = df_new['Agility_3cone'] + df_new['Shuttle']
    
    # 5. 爆発力指標（立ち幅跳び * 垂直跳び）
    df_new['Explosiveness'] = df_new['Broad_Jump'] * df_new['Vertical_Jump']
    
    # 6. スピード×爆発力
    df_new['Speed_Explosiveness'] = df_new['Explosiveness'] / df_new['Sprint_40yd']
    
    # 7. 体格指標（身長×体重）
    df_new['Size_Index'] = df_new['Height'] * df_new['Weight']
    
    # 8. ベンチプレス効率（回数/体重×身長）
    df_new['Bench_Efficiency'] = df_new['Bench_Press_Reps'] / df_new['Size_Index']
    
    # 9. 年齢グループ（カテゴリ化）
    df_new['Age_Group'] = pd.cut(df_new['Age'], bins=[0, 21, 23, 30], 
                                labels=['Young', 'Standard', 'Veteran'], 
                                include_lowest=True)
    
    print("基本特徴量作成完了")
    return df_new

def create_position_features(df):
    """ポジション別の特徴量を作成"""
    df_new = df.copy()
    
    # ポジションタイプ別の正規化特徴量
    numerical_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 
                     'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    for col in numerical_cols:
        if col in df_new.columns:
            # ポジションタイプ別の平均値で正規化
            position_means = df_new.groupby('Position_Type')[col].transform('mean')
            df_new[f'{col}_vs_Position_Mean'] = df_new[col] / position_means
            
            # ポジションタイプ別の標準偏差で正規化
            position_stds = df_new.groupby('Position_Type')[col].transform('std')
            df_new[f'{col}_Position_Zscore'] = (df_new[col] - position_means) / position_stds
    
    # ポジション特化の組み合わせ特徴量
    # 1. オフェンシブライン用特徴量
    offensive_line_mask = df_new['Position_Type'] == 'offensive_lineman'
    df_new['OL_Strength_Index'] = 0
    df_new.loc[offensive_line_mask, 'OL_Strength_Index'] = (
        df_new.loc[offensive_line_mask, 'Bench_Press_Reps'] * 
        df_new.loc[offensive_line_mask, 'Weight'] / 100
    )
    
    # 2. スキルポジション用特徴量
    skill_positions = ['backs_receivers']
    skill_mask = df_new['Position_Type'].isin(skill_positions)
    df_new['Skill_Speed_Index'] = 0
    df_new.loc[skill_mask, 'Skill_Speed_Index'] = (
        df_new.loc[skill_mask, 'Vertical_Jump'] * 
        df_new.loc[skill_mask, 'Broad_Jump'] / 
        df_new.loc[skill_mask, 'Sprint_40yd']
    )
    
    # 3. ディフェンス用特徴量
    defense_positions = ['defensive_lineman', 'line_backer', 'defensive_back']
    defense_mask = df_new['Position_Type'].isin(defense_positions)
    df_new['Defense_Athleticism'] = 0
    df_new.loc[defense_mask, 'Defense_Athleticism'] = (
        (df_new.loc[defense_mask, 'Vertical_Jump'] + 
         df_new.loc[defense_mask, 'Broad_Jump']) / 
        (df_new.loc[defense_mask, 'Sprint_40yd'] + 
         df_new.loc[defense_mask, 'Agility_3cone'])
    )
    
    print("ポジション特化特徴量作成完了")
    return df_new

def create_school_features(df):
    """学校に関する特徴量を作成"""
    df_new = df.copy()
    
    # 訓練データのみを使って学校統計を計算（データリークを防ぐ）
    train_data = df_new[df_new['Drafted'].notna()].copy()
    
    # 学校別の指名率と選手数
    school_stats = train_data.groupby('School').agg({
        'Drafted': ['count', 'mean']
    })
    school_stats.columns = ['School_Player_Count', 'School_Draft_Rate']
    
    # 5人未満の学校は統計が不安定なので平均値で補完
    overall_draft_rate = train_data['Drafted'].mean()
    school_stats.loc[school_stats['School_Player_Count'] < 5, 'School_Draft_Rate'] = overall_draft_rate
    
    # 学校プレステージカテゴリ
    def categorize_school_prestige(draft_rate, player_count):
        if pd.isna(draft_rate):
            return 'Unknown'
        elif draft_rate >= 0.8:
            return 'Elite'
        elif draft_rate >= 0.6:
            return 'High'
        elif draft_rate >= 0.4:
            return 'Medium'
        else:
            return 'Low'
    
    school_stats['School_Prestige'] = school_stats.apply(
        lambda x: categorize_school_prestige(x['School_Draft_Rate'], x['School_Player_Count']), axis=1
    )
    
    # 全データにマージ
    df_new = df_new.merge(school_stats, left_on='School', right_index=True, how='left')
    
    # 新しい学校（テストデータのみに存在）は平均値で補完
    df_new['School_Draft_Rate'].fillna(overall_draft_rate, inplace=True)
    df_new['School_Player_Count'].fillna(1, inplace=True)
    df_new['School_Prestige'].fillna('Unknown', inplace=True)
    
    print("学校プレステージ特徴量作成完了")
    return df_new

def advanced_missing_value_handling(df):
    """ポジション別・パターン別の欠損値処理"""
    df_new = df.copy()
    
    # 身体能力テスト系の列
    performance_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 
                       'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    # 1. 欠損パターンフラグの作成
    for col in performance_cols:
        df_new[f'{col}_is_missing'] = df_new[col].isnull().astype(int)
    
    # 2. 欠損値の総数
    df_new['Total_Missing_Tests'] = df_new[[f'{col}_is_missing' for col in performance_cols]].sum(axis=1)
    
    # 3. ポジション別の中央値で補完
    for col in performance_cols:
        if col in df_new.columns:
            # ポジションタイプ別の中央値
            position_medians = df_new.groupby('Position_Type')[col].median()
            
            for position in position_medians.index:
                mask = (df_new['Position_Type'] == position) & (df_new[col].isnull())
                df_new.loc[mask, col] = position_medians[position]
            
            # 残った欠損値は全体の中央値で補完
            overall_median = df_new[col].median()
            df_new[col].fillna(overall_median, inplace=True)
    
    # 4. Age の補完
    if 'Age' in df_new.columns:
        # ポジション別の中央値で補完
        age_medians = df_new.groupby('Position_Type')['Age'].median()
        for position in age_medians.index:
            mask = (df_new['Position_Type'] == position) & (df_new['Age'].isnull())
            df_new.loc[mask, 'Age'] = age_medians[position]
        
        # 残った欠損値は全体の中央値で補完
        df_new['Age'].fillna(df_new['Age'].median(), inplace=True)
    
    print("改良された欠損値処理完了")
    return df_new

def encode_categorical_features(df):
    """カテゴリ変数の効果的なエンコーディング"""
    df_new = df.copy()
    
    # Label Encodingする列
    label_encode_cols = ['School', 'Player_Type', 'Position_Type', 'Position', 
                        'Age_Group', 'School_Prestige']
    
    encoders = {}
    for col in label_encode_cols:
        if col in df_new.columns:
            le = LabelEncoder()
            df_new[col] = le.fit_transform(df_new[col].astype(str))
            encoders[col] = le
    
    # 年度の処理（既に数値なのでそのまま）
    # 年度の周期性を考慮した特徴量
    if 'Year' in df_new.columns:
        df_new['Year_sin'] = np.sin(2 * np.pi * (df_new['Year'] - df_new['Year'].min()) / 
                                   (df_new['Year'].max() - df_new['Year'].min()))
        df_new['Year_cos'] = np.cos(2 * np.pi * (df_new['Year'] - df_new['Year'].min()) / 
                                   (df_new['Year'].max() - df_new['Year'].min()))
    
    print("カテゴリ変数エンコーディング完了")
    return df_new, encoders

def prepare_final_dataset(df, train_len):
    """最終的なデータセットの準備"""
    # 使用しない列を除外
    exclude_cols = ['Id', 'Drafted']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 数値型以外の列があれば確認
    non_numeric_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        print(f"警告: 数値型以外の列があります: {list(non_numeric_cols)}")
        # 強制的に数値型に変換
        for col in non_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 無限大や極端に大きな値の処理
    for col in feature_cols:
        if col in df.columns:
            # 無限大を NaN に変換
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # 残った NaN を中央値で補完
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    
    # データを分割
    train_processed = df[:train_len].copy()
    test_processed = df[train_len:].copy()
    
    # 特徴量とターゲットを分離
    X_train = train_processed[feature_cols]
    y_train = train_processed['Drafted']
    X_test = test_processed[feature_cols]
    
    print(f"特徴量数: {len(feature_cols)}")
    print(f"訓練データサイズ: {X_train.shape}")
    print(f"テストデータサイズ: {X_test.shape}")
    print(f"欠損値確認 - 訓練: {X_train.isnull().sum().sum()}, テスト: {X_test.isnull().sum().sum()}")
    
    return X_train, y_train, X_test, feature_cols

# データ準備
print("\n1. データ読み込み・特徴量エンジニアリング実行中...")
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample_submission = pd.read_csv('data/sample_submission_6_20修正.csv')

print(f"Train: {train.shape}, Test: {test.shape}")

# データのコピーを作成（元データを保持）
train_fe = train.copy()
test_fe = test.copy()

# 全データを結合（特徴量エンジニアリングのため）
all_data = pd.concat([train_fe, test_fe], ignore_index=True)

# 特徴量エンジニアリングパイプライン実行
all_data_fe = create_basic_features(all_data)
all_data_fe = create_position_features(all_data_fe)
all_data_fe = create_school_features(all_data_fe)
all_data_fe = advanced_missing_value_handling(all_data_fe)
all_data_fe, label_encoders = encode_categorical_features(all_data_fe)

# 最終データセットの準備
X_train_fe, y_train_fe, X_test_fe, feature_cols_fe = prepare_final_dataset(all_data_fe, len(train))

# LightGBMモデルの実装・評価
def evaluate_lightgbm_model(X_train, y_train, n_splits=5, random_state=42):
    """LightGBMモデルの交差検証評価"""
    
    # Stratified K-Fold設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # スコア格納用
    cv_scores = []
    feature_importance_sum = np.zeros(X_train.shape[1])
    
    print("\n2. LightGBM 5-Fold交差検証実行中...")
    
    # LightGBMパラメータ設定
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbosity': -1,
        'random_state': random_state
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        # データ分割
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # LightGBMデータセット作成
        train_dataset = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_dataset = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_dataset)
        
        # モデル訓練
        lgb_model = lgb.train(
            lgb_params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'valid'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)  # ログ出力を抑制
            ]
        )
        
        # 予測・評価
        y_pred_proba = lgb_model.predict(X_fold_val, num_iteration=lgb_model.best_iteration)
        fold_score = roc_auc_score(y_fold_val, y_pred_proba)
        cv_scores.append(fold_score)
        
        # 特徴量重要度の累積
        feature_importance_sum += lgb_model.feature_importance(importance_type='gain')
        
        print(f"Fold {fold}: ROC AUC = {fold_score:.5f} (best_iteration: {lgb_model.best_iteration})")
    
    # 結果の集計
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    feature_importance_avg = feature_importance_sum / n_splits
    
    print(f"\n=== LightGBM 最終結果 ===")
    print(f"平均ROC AUC: {mean_score:.5f} ± {std_score:.5f}")
    print(f"ベースラインとの比較: {mean_score:.5f} vs 0.80792")
    print(f"RandomForestとの比較: {mean_score:.5f} vs 0.83743")
    
    if mean_score > 0.83743:
        improvement = ((mean_score - 0.83743) / 0.83743) * 100
        print(f"🎉 RandomForest改善: +{improvement:.2f}%")
    elif mean_score > 0.80792:
        improvement = ((mean_score - 0.80792) / 0.80792) * 100
        print(f"✅ ベースライン改善: +{improvement:.2f}%")
    else:
        decline = ((0.80792 - mean_score) / 0.80792) * 100
        print(f"⚠️  ベースライン下回り: -{decline:.2f}%")
    
    return cv_scores, feature_importance_avg, lgb_params

# LightGBMモデルの評価実行
cv_scores_lgb, feature_importance_lgb, best_params = evaluate_lightgbm_model(X_train_fe, y_train_fe)

# 特徴量重要度の分析
feature_importance_df_lgb = pd.DataFrame({
    'feature': feature_cols_fe,
    'importance': feature_importance_lgb
}).sort_values('importance', ascending=False)

print(f"\n=== LightGBM 特徴量重要度TOP10 ===")
print(feature_importance_df_lgb.head(10))

# RandomForestとの特徴量重要度比較
print(f"\n=== RandomForest vs LightGBM 特徴量重要度比較 ===")
print("RandomForest TOP5:")
print("1. Age_Group (0.255767)")
print("2. Age (0.048578)")
print("3. Sprint_40yd_Position_Zscore (0.047013)")
print("4. School_Draft_Rate (0.044324)")
print("5. Sprint_40yd_vs_Position_Mean (0.039080)")

print(f"\nLightGBM TOP5:")
for i, (_, row) in enumerate(feature_importance_df_lgb.head(5).iterrows(), 1):
    print(f"{i}. {row['feature']} ({row['importance']:.0f})")

# 最終モデルの訓練とテストデータ予測
def create_lightgbm_submission(X_train, y_train, X_test, test_ids, params, random_state=42):
    """LightGBMで最終的な提出ファイルを作成"""
    
    print("\n3. LightGBM 最終モデル訓練中...")
    
    # 全訓練データでLightGBMデータセット作成
    train_dataset = lgb.Dataset(X_train, label=y_train)
    
    # モデル訓練（全データ使用）
    final_lgb_model = lgb.train(
        params,
        train_dataset,
        num_boost_round=1000,
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    # テストデータで予測
    test_predictions = final_lgb_model.predict(X_test)
    
    # 提出ファイル作成
    submission = pd.DataFrame({
        'Id': test_ids,
        'Drafted': test_predictions
    })
    
    # ファイル保存
    submission_filename = 'lightgbm_submission.csv'
    submission.to_csv(submission_filename, index=False)
    
    print(f"LightGBM提出ファイル保存完了: {submission_filename}")
    print(f"予測値の統計:")
    print(f"  平均: {test_predictions.mean():.4f}")
    print(f"  中央値: {np.median(test_predictions):.4f}")
    print(f"  最小値: {test_predictions.min():.4f}")
    print(f"  最大値: {test_predictions.max():.4f}")
    print(f"  標準偏差: {test_predictions.std():.4f}")
    
    return submission, final_lgb_model

# LightGBM最終提出ファイルの作成
test_ids = test['Id'].values
final_submission_lgb, final_lgb_model = create_lightgbm_submission(X_train_fe, y_train_fe, X_test_fe, test_ids, best_params)

print(f"\n=== LightGBM実装完了 ===")
print(f"- LightGBMのCVスコア: {np.mean(cv_scores_lgb):.5f}")
print(f"- RandomForest比較: {np.mean(cv_scores_lgb):.5f} vs 0.83743")
print(f"- ベースライン比較: {np.mean(cv_scores_lgb):.5f} vs 0.80792")
print(f"- 提出ファイル: lightgbm_submission.csv") 
"""
# ml_api/model/train_models/train_regression_match_model.py
import os
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction import DictVectorizer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sentence_transformers import SentenceTransformer, util
import optuna
import warnings

warnings.filterwarnings("ignore")


def extract_tech_features(df: pd.DataFrame, text_col: str, side: str) -> pd.DataFrame:
    techs = [
        "sap", "python", "excel", "sql", "java", "power bi", "r", "tableau",
        "machine learning", "deep learning", "cloud", "azure", "aws", "linux",
        "c#", "c++", "react", "node", "git", "jira"
    ]
    for tec in techs:
        col_name = f"{side}_tec_{tec.replace(' ', '_')}"
        df[col_name] = (
            df[text_col]
            .astype(str)
            .str.lower()
            .str.contains(tec, na=False)
            .astype(int)
        )
    return df


def gerar_match_data():
    print("🔍 Gerando match_data.csv em ml_api/data/refined/ …")
    cand = pd.read_csv("ml_api/data/refined/candidatos.csv")
    vaga = pd.read_csv("ml_api/data/refined/vagas.csv")
    with open("ml_api/data/extracted/prospects.json", "r", encoding="utf-8") as f:
        prospects = json.load(f)

    rows = []
    for vid, data in prospects.items():
        for p in data.get("prospects", []):
            code = p.get("codigo")
            match_label = 1 if "Contratado" in p.get("situacao_candidado", "") else 0
            if code is not None:
                rows.append({
                    "codigo_vaga": int(vid),
                    "codigo_candidato": int(code),
                    "match": match_label
                })
    df = pd.DataFrame(rows)
    df = df.merge(cand, left_on="codigo_candidato", right_on="id", how="left")
    df = df.merge(vaga, left_on="codigo_vaga", right_on="vaga_id", how="left", suffixes=("_cand","_vaga"))
    df = extract_tech_features(df, "texto_classificado", "cand")
    df = extract_tech_features(df, "descricao", "vaga")

    # Ajuste de nomes para consistência
    df = df.rename(columns={
        "cluster": "cluster_cand",
        "classification": "cluster_vaga",
        "nível": "nivel",
        "é_sap": "eh_sap"
    })
    df.dropna(subset=["cluster_cand","cluster_vaga","nivel","eh_sap"], inplace=True)

    # Encoding categórico
    df["cluster_cand_enc"] = LabelEncoder().fit_transform(df["cluster_cand"])
    df["cluster_vaga_enc"] = LabelEncoder().fit_transform(df["cluster_vaga"])
    df["nivel_enc"] = LabelEncoder().fit_transform(df["nivel"])
    df["eh_sap"] = df["eh_sap"].astype(int)

    print("🔎 Calculando similaridade semântica SBERT…")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    textos_cand = df["texto_classificado"].astype(str).tolist()
    textos_vaga = df["descricao"].astype(str).tolist()
    emb_cand = sbert.encode(textos_cand, batch_size=32, show_progress_bar=True)
    emb_vaga = sbert.encode(textos_vaga, batch_size=32, show_progress_bar=True)
    sims = [float(util.cos_sim(a, b)) for a, b in zip(emb_cand, emb_vaga)]
    df["sim_sbert"] = sims
    print("✅ Similaridade semântica calculada e adicionada como feature.")

    # Seleção de features
    num_feats = ["cluster_cand_enc","cluster_vaga_enc","nivel_enc","eh_sap","sim_sbert"]
    tech_feats = [c for c in df.columns if c.startswith("cand_tec_") or c.startswith("vaga_tec_")]
    features = num_feats + tech_feats

    df_final = df[["codigo_vaga","codigo_candidato","match"] + features]
    os.makedirs("ml_api/data/refined", exist_ok=True)
    df_final.to_csv("ml_api/data/refined/match_data.csv", index=False)
    print("✅ match_data.csv salvo em: ml_api/data/refined/match_data.csv")
    return df_final, features


def objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255)
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pool = Pool(X_tr, y_tr)
        model = CatBoostClassifier(**params, verbose=False)
        model.fit(pool)
        preds = model.predict(X_val)
        scores.append(f1_score(y_val, preds))
    return np.mean(scores)


def balance_epoch(X, y, pos_frac=0.8, random_state=None):
    pos = X[y == 1]
    neg = X[y == 0]
    n_pos = int(min(len(pos), len(neg) * pos_frac / (1 - pos_frac)))
    n_neg = int(n_pos * (1 - pos_frac) / pos_frac)
    pos_sample = pos.sample(n=n_pos, replace=n_pos > len(pos), random_state=random_state)
    neg_sample = neg.sample(n=n_neg, replace=n_neg > len(neg), random_state=random_state)
    X_ep = pd.concat([pos_sample, neg_sample])
    y_ep = pd.Series([1]*len(pos_sample) + [0]*len(neg_sample), index=X_ep.index)
    return X_ep.sample(frac=1, random_state=random_state), y_ep.sample(frac=1, random_state=random_state)


def treinar_otimista(df, features, n_epochs=30, pos_frac=0.8):
    X, y = df[features], df['match']
    best = {'f1': 0, 'model': None, 'thr': 0.5, 'precision': 0, 'recall': 0}
    for epoch in range(1, n_epochs+1):
        print(f"🌊 Época {epoch}/{n_epochs} — balanceando para {int(pos_frac*100)}% positivos…")
        X_bal, y_bal = balance_epoch(X, y, pos_frac, random_state=epoch)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42+epoch
        )
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_tr, y_tr), n_trials=8, show_progress_bar=False)
        params = study.best_params
        estimators = [
            ('cb', CatBoostClassifier(**params, verbose=False)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42+epoch)),
            ('lgb', LGBMClassifier(n_estimators=200, random_state=42+epoch))
        ]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=CatBoostClassifier(**params, verbose=False),
            cv=3
        )
        stack.fit(X_tr, y_tr)
        probs = stack.predict_proba(X_te)[:,1]
        prec, rec, thr = precision_recall_curve(y_te, probs)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
        ix = np.argmax(f1s)
        thr_opt = thr[ix] if ix < len(thr) else 0.5
        preds = (probs >= thr_opt).astype(int)
        f1e = f1_score(y_te, preds)
        p  = precision_score(y_te, preds)
        r  = recall_score(y_te, preds)
        print(classification_report(y_te, preds, digits=4))
        if f1e > best['f1']:
            best.update({'f1': f1e, 'model': stack, 'thr': thr_opt, 'precision': p, 'recall': r})
    print(f"💾 Melhor modelo otimista: F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, thr={best['thr']:.2f}")
    return best


def treinar_pessimista(df, features, n_epochs=20):
    X, y = df[features], df['match']
    best = {'f1': 0, 'model': None, 'thr': 0.5, 'precision': 0, 'recall': 0}
    for epoch in range(n_epochs):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=142+epoch
        )
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, X_tr, y_tr), n_trials=5, show_progress_bar=False)
        params = study.best_params
        estimators = [
            ('cb', CatBoostClassifier(**params, verbose=False)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=142+epoch)),
            ('lgb', LGBMClassifier(n_estimators=200, random_state=142+epoch))
        ]
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=CatBoostClassifier(**params, verbose=False),
            cv=3
        )
        stack.fit(X_tr, y_tr)
        probs = stack.predict_proba(X_te)[:,1]
        prec, rec, thr = precision_recall_curve(y_te, probs)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
        ix = np.argmax(f1s)
        thr_opt = thr[ix] if ix < len(thr) else 0.5
        preds = (probs >= thr_opt).astype(int)
        f1e = f1_score(y_te, preds)
        p  = precision_score(y_te, preds)
        r  = recall_score(y_te, preds)
        if f1e > best['f1']:
            best.update({'f1': f1e, 'model': stack, 'thr': thr_opt, 'precision': p, 'recall': r})
    print(f"💾 Melhor modelo pessimista: F1={best['f1']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}, thr={best['thr']:.2f}")
    return best


def pipeline_duplo():
    df, features = gerar_match_data()

    best_oti = treinar_otimista(df, features, n_epochs=30, pos_frac=0.8)
    best_pes = treinar_pessimista(df, features, n_epochs=20)

    os.makedirs("ml_api/model", exist_ok=True)
    joblib.dump(
        {'model': best_oti['model'], 'features': features, 'threshold': best_oti['thr']},
        "ml_api/model/match_otimista.pkl"
    )
    joblib.dump(
        {'model': best_pes['model'], 'features': features, 'threshold': best_pes['thr']},
        "ml_api/model/match_pessimista.pkl"
    )

    # serializa pipeline de pré‑processamento
    print("🛠️ Serializando pipeline de pré‑processamento...")
    X_dicts = df[features].to_dict(orient='records')
    feature_pipeline = DictVectorizer(sparse=False)
    feature_pipeline.fit(X_dicts)
    joblib.dump(feature_pipeline, "ml_api/model/feature_pipeline.joblib")
    print("💾 Pipeline salvo em ml_api/model/feature_pipeline.joblib")

    # grava métricas de produção
    metrics = {
        "otimista": {
            "f1": best_oti['f1'],
            "precision": best_oti['precision'],
            "recall": best_oti['recall'],
            "threshold": best_oti['thr']
        },
        "pessimista": {
            "f1": best_pes['f1'],
            "precision": best_pes['precision'],
            "recall": best_pes['recall'],
            "threshold": best_pes['thr']
        }
    }
    with open("ml_api/model/match_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("💾 Métricas salvas em ml_api/model/match_metrics.json\n")

    # ───── RESUMO FINAL PARA PRODUÇÃO ─────
    print("🎯 Resumo dos modelos em produção:")
    print(f" • Modelo OTIMISTA → F1={metrics['otimista']['f1']:.4f}, "
          f"Precision={metrics['otimista']['precision']:.4f}, "
          f"Recall={metrics['otimista']['recall']:.4f}, "
          f"Threshold={metrics['otimista']['threshold']:.2f}")
    print(f" • Modelo PESSIMISTA → F1={metrics['pessimista']['f1']:.4f}, "
          f"Precision={metrics['pessimista']['precision']:.4f}, "
          f"Recall={metrics['pessimista']['recall']:.4f}, "
          f"Threshold={metrics['pessimista']['threshold']:.2f}")
    print("✅ Treinamento concluído e modelos prontos para produção.")


if __name__ == "__main__":
    pipeline_duplo()
""" 

# train_regression_match_model_improved.py

import os
import json
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score, precision_recall_curve,
    precision_score, recall_score, make_scorer
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction import DictVectorizer
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from sentence_transformers import SentenceTransformer, util
import optuna
import warnings

warnings.filterwarnings("ignore")


def extract_tech_features(df: pd.DataFrame, text_col: str, side: str) -> pd.DataFrame:
    techs = [
        "sap", "python", "excel", "sql", "java", "power bi", "r", "tableau",
        "machine learning", "deep learning", "cloud", "azure", "aws", "linux",
        "c#", "c++", "react", "node", "git", "jira"
    ]
    for tec in techs:
        col_name = f"{side}_tec_{tec.replace(' ', '_')}"
        df[col_name] = (
            df[text_col].astype(str)
                         .str.lower()
                         .str.contains(tec, na=False)
                         .astype(int)
        )
    return df


def gerar_match_data():
    print("🔍 Gerando match_data.csv em ml_api/data/refined/ …")
    cand = pd.read_csv("ml_api/data/refined/candidatos.csv")
    vaga = pd.read_csv("ml_api/data/refined/vagas.csv")
    with open("ml_api/data/extracted/prospects.json", "r", encoding="utf-8") as f:
        prospects = json.load(f)

    rows = []
    for vid, data in prospects.items():
        for p in data.get("prospects", []):
            code = p.get("codigo")
            match_label = 1 if "Contratado" in p.get("situacao_candidado", "") else 0
            if code is not None:
                rows.append({
                    "codigo_vaga": int(vid),
                    "codigo_candidato": int(code),
                    "match": match_label
                })
    df = pd.DataFrame(rows)
    df = df.merge(cand, left_on="codigo_candidato", right_on="id", how="left")
    df = df.merge(vaga, left_on="codigo_vaga", right_on="vaga_id", how="left", suffixes=("_cand","_vaga"))
    df = extract_tech_features(df, "texto_classificado", "cand")
    df = extract_tech_features(df, "descricao", "vaga")

    df = df.rename(columns={
        "cluster": "cluster_cand",
        "classification": "cluster_vaga",
        "nível": "nivel",
        "é_sap": "eh_sap"
    })
    df.dropna(subset=["cluster_cand","cluster_vaga","nivel","eh_sap"], inplace=True)

    df["cluster_cand_enc"] = LabelEncoder().fit_transform(df["cluster_cand"])
    df["cluster_vaga_enc"] = LabelEncoder().fit_transform(df["cluster_vaga"])
    df["nivel_enc"] = LabelEncoder().fit_transform(df["nivel"])
    df["eh_sap"] = df["eh_sap"].astype(int)

    print("🔎 Calculando similaridade semântica SBERT…")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    textos_cand = df["texto_classificado"].astype(str).tolist()
    textos_vaga = df["descricao"].astype(str).tolist()
    emb_cand = sbert.encode(textos_cand, batch_size=32, show_progress_bar=True)
    emb_vaga = sbert.encode(textos_vaga, batch_size=32, show_progress_bar=True)
    sims = [float(util.cos_sim(a, b)) for a, b in zip(emb_cand, emb_vaga)]
    df["sim_sbert"] = sims

    num_feats = ["cluster_cand_enc","cluster_vaga_enc","nivel_enc","eh_sap","sim_sbert"]
    tech_feats = [c for c in df.columns if c.startswith("cand_tec_") or c.startswith("vaga_tec_")]
    features = num_feats + tech_feats

    df_final = df[["codigo_vaga","codigo_candidato","match"] + features]
    os.makedirs("ml_api/data/refined", exist_ok=True)
    df_final.to_csv("ml_api/data/refined/match_data.csv", index=False)
    print("✅ match_data.csv salvo em ml_api/data/refined/match_data.csv")
    return df_final, features


def objective(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255)
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pool = Pool(X_tr, y_tr)
        model = CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced')
        model.fit(pool)
        preds = model.predict(X_val)
        f1_scores.append(f1_score(y_val, preds))
    return np.mean(f1_scores)


def balance_epoch(X, y, pos_frac=0.8, random_state=None):
    pos = X[y == 1]
    neg = X[y == 0]
    n_pos = int(min(len(pos), len(neg) * pos_frac / (1 - pos_frac)))
    n_neg = int(n_pos * (1 - pos_frac) / pos_frac)
    pos_sample = pos.sample(n=n_pos, replace=n_pos > len(pos), random_state=random_state)
    neg_sample = neg.sample(n=n_neg, replace=n_neg > len(neg), random_state=random_state)
    X_ep = pd.concat([pos_sample, neg_sample])
    y_ep = pd.Series([1]*len(pos_sample) + [0]*len(neg_sample), index=X_ep.index)
    return X_ep.sample(frac=1, random_state=random_state), y_ep.sample(frac=1, random_state=random_state)


def treinar_otimista(df, features, n_epochs=30, pos_frac=0.8):
    X, y = df[features], df['match']
    best = {'f1':0, 'model':None, 'thr':0.5, 'prec':0, 'rec':0}
    for epoch in range(1, n_epochs+1):
        Xb, yb = balance_epoch(X, y, pos_frac, epoch)
        Xt, Xv, yt, yv = train_test_split(Xb, yb, test_size=0.2,
                                          stratify=yb, random_state=42+epoch)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective(t, Xt, yt), n_trials=8, show_progress_bar=False)
        params = study.best_params
        ests = [
            ('cb', CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight={0:3,1:1}, random_state=42+epoch)),
            ('lgb', LGBMClassifier(n_estimators=200, class_weight={0:3,1:1}, random_state=42+epoch))
        ]
        stack = StackingClassifier(estimators=ests,
                                   final_estimator=CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced'),
                                   cv=3)
        stack.fit(Xt, yt)
        probs = stack.predict_proba(Xv)[:,1]
        precs, recs, ths = precision_recall_curve(yv, probs)
        f1s = 2 * (precs*recs)/(precs+recs+1e-9)
        ix = np.argmax(f1s)
        thr = ths[ix] if ix < len(ths) else 0.5
        preds = (probs>=thr).astype(int)
        f1e = f1_score(yv, preds); p=precision_score(yv,preds); r=recall_score(yv,preds)
        if f1e>best['f1']:
            best.update({'f1':f1e,'model':stack,'thr':thr,'prec':p,'rec':r})
    print(f"💾 Otimista → F1={best['f1']:.4f}, Prec={best['prec']:.4f}, Rec={best['rec']:.4f}, thr={best['thr']:.2f}")
    return best


def objective_neg(trial, X, y):
    # maximiza F1 da classe 0
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255)
    }
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    neg_f1 = make_scorer(lambda yt, yp: f1_score(yt, yp, pos_label=0))
    scores = []
    for ti, vi in skf.split(X, y):
        Xt, Xv = X.iloc[ti], X.iloc[vi]
        yt, yv = y.iloc[ti], y.iloc[vi]
        pool = Pool(Xt, yt)
        model = CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced')
        model.fit(pool)
        preds = model.predict(Xv)
        scores.append(f1_score(yv, preds, pos_label=0))
    return np.mean(scores)


def balance_neg(X, y, neg_frac=0.8, random_state=None):
    neg = X[y==0]; pos = X[y==1]
    n_neg = int(min(len(neg), len(pos)*neg_frac/(1-neg_frac)))
    n_pos = int(n_neg*(1-neg_frac)/neg_frac)
    neg_s = neg.sample(n=n_neg, replace=n_neg>len(neg), random_state=random_state)
    pos_s = pos.sample(n=n_pos, replace=n_pos>len(pos), random_state=random_state)
    Xep = pd.concat([neg_s,pos_s])
    yep = pd.Series([0]*len(neg_s)+[1]*len(pos_s), index=Xep.index)
    return Xep.sample(frac=1, random_state=random_state), yep.sample(frac=1, random_state=random_state)


def treinar_pessimista(df, features, n_epochs=20):
    X, y = df[features], df['match']
    best = {'f1':0,'model':None,'thr':0.5,'prec':0,'rec':0}
    for epoch in range(n_epochs):
        Xb, yb = balance_neg(X, y, neg_frac=0.8, random_state=epoch)
        Xt, Xv, yt, yv = train_test_split(Xb, yb, test_size=0.2,
                                          stratify=yb, random_state=142+epoch)
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objective_neg(t, Xt, yt), n_trials=5, show_progress_bar=False)
        params = study.best_params
        ests = [
            ('cb', CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, class_weight={0:3,1:1}, random_state=142+epoch)),
            ('lgb', LGBMClassifier(n_estimators=200, class_weight={0:3,1:1}, random_state=142+epoch))
        ]
        stack = StackingClassifier(estimators=ests,
                                   final_estimator=CatBoostClassifier(**params, verbose=False, auto_class_weights='Balanced'),
                                   cv=3)
        stack.fit(Xt, yt)
        probs = stack.predict_proba(Xv)[:,1]
        precs, recs, ths = precision_recall_curve(yv, probs)
        f1s = 2*(precs*recs)/(precs+recs+1e-9)
        ix = np.argmax(f1s)
        thr = ths[ix] if ix < len(ths) else 0.5
        preds = (probs>=thr).astype(int)
        f1e = f1_score(yv,preds); p=precision_score(yv,preds); r=recall_score(yv,preds)
        if f1e>best['f1']:
            best.update({'f1':f1e,'model':stack,'thr':thr,'prec':p,'rec':r})
    print(f"💾 Pessimista → F1={best['f1']:.4f}, Prec={best['prec']:.4f}, Rec={best['rec']:.4f}, thr={best['thr']:.2f}")
    return best


def pipeline_duplo():
    df, features = gerar_match_data()
    best_o = treinar_otimista(df, features)
    best_p = treinar_pessimista(df, features)

    os.makedirs("ml_api/model", exist_ok=True)
    joblib.dump({'model':best_o['model'],'features':features,'threshold':best_o['thr']},
                "ml_api/model/match_otimista.pkl")
    joblib.dump({'model':best_p['model'],'features':features,'threshold':best_p['thr']},
                "ml_api/model/match_pessimista.pkl")

    # serializa pipeline
    Xd = df[features].to_dict(orient='records')
    pipe = DictVectorizer(sparse=False); pipe.fit(Xd)
    joblib.dump(pipe, "ml_api/model/feature_pipeline.joblib")

    metrics = {
      "otimista": {"f1":best_o['f1'],"precision":best_o['prec'],"recall":best_o['rec'],"thr":best_o['thr']},
      "pessimista":{"f1":best_p['f1'],"precision":best_p['prec'],"recall":best_p['rec'],"thr":best_p['thr']}
    }
    with open("ml_api/model/match_metrics.json","w") as f:
        json.dump(metrics,f,indent=2)

    print("\n🎯 Modelos em produção:")
    print(f" • OTIMISTA → F1={metrics['otimista']['f1']:.4f}, Precision={metrics['otimista']['precision']:.4f}, Recall={metrics['otimista']['recall']:.4f}, thr={metrics['otimista']['thr']:.2f}")
    print(f" • PESSIMISTA → F1={metrics['pessimista']['f1']:.4f}, Precision={metrics['pessimista']['precision']:.4f}, Recall={metrics['pessimista']['recall']:.4f}, thr={metrics['pessimista']['thr']:.2f}")
    print("✅ Treinamento concluído.")
    

if __name__ == "__main__":
    pipeline_duplo()
    

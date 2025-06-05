import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy import stats

def delong_test(y_true, pred1, pred2):
    """
    Compute DeLong's test statistic and p-value for comparing two AUC scores
    """
    n1 = len(y_true[y_true == 1])
    n2 = len(y_true[y_true == 0])
    
    # Get predictions for each class
    y_true = np.array(y_true)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    positive_scores1 = pred1[y_true == 1]
    negative_scores1 = pred1[y_true == 0]
    positive_scores2 = pred2[y_true == 1]
    negative_scores2 = pred2[y_true == 0]

    # Compute U-statistics for both models
    u_stats1 = 0
    u_stats2 = 0
    
    # For first model
    for pos_score in positive_scores1:
        u_stats1 += (sum(pos_score > negative_scores1) + 
                    0.5 * sum(pos_score == negative_scores1))
    u_stats1 = u_stats1 / (n1 * n2)
    
    # For second model
    for pos_score in positive_scores2:
        u_stats2 += (sum(pos_score > negative_scores2) + 
                    0.5 * sum(pos_score == negative_scores2))
    u_stats2 = u_stats2 / (n1 * n2)
    
    # Compute covariance matrix
    V10 = np.cov(positive_scores1, positive_scores2)[0, 1]
    V01 = np.cov(negative_scores1, negative_scores2)[0, 1]
    
    # Compute variance of the difference
    var = (V10 / n1 + V01 / n2)
    
    # Handle edge cases where variance is too small
    if var < 1e-10:  # Numerical threshold
        if abs(u_stats1 - u_stats2) < 1e-10:
            return 1.0  # No significant difference
        else:
            return 0.0  # Significant difference
    
    # Compute z-score
    z = (u_stats1 - u_stats2) / np.sqrt(var)
    
    # Compute p-value (two-tailed test)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value

def main():
    # Load features.csv
    df = pd.read_csv('data/features.csv', parse_dates=['timestamp'])

    # Time-based split
    train = df['timestamp'] < '2006-12-01'
    val   = (df['timestamp'] >= '2007-01-01') & (df['timestamp'] < '2007-12-01')
    test  = df['timestamp'] >= '2007-12-01'

    df_train = df[train].copy()
    df_val   = df[val].copy()
    df_test  = df[test].copy()

    # Feature groups
    vote_graph = ['support_ratio','out_degree','scaled_out_degree','clustering_coeff','balanced_triad_count','pagerank']
    cand_history = ['cand_support_count','cand_oppose_count','cand_total_votes','cand_support_ratio','vote_rank_on_candidate','time_since_last_vote_on_candidate','decayed_support','decayed_oppose','decay_support_ratio']
    temporal = ['window_out_degree','window_support_ratio','velocity','acceleration']
    full = vote_graph + cand_history + temporal

    # Impute missing values
    for df_sub in [df_train, df_val, df_test]:
        df_sub[full] = df_sub[full].fillna(0)

    # Prepare matrices
    X_train_full = df_train[full]; y_train = df_train['label']
    X_val_full   = df_val[full];   y_val   = df_val['label']
    X_test_full  = df_test[full];  y_test  = df_test['label']

    # Scale for LR
    scaler = StandardScaler().fit(X_train_full)
    X_train_lr = scaler.transform(X_train_full)
    X_val_lr   = scaler.transform(X_val_full)
    X_test_lr  = scaler.transform(X_test_full)

    # Models with single-threaded RF
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=1, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }

    # Evaluate on validation
    val_results = []
    val_predictions = {}
    print("\nModel Validation Results:")
    print("-----------------------")
    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_lr, y_train)
            pred = model.predict_proba(X_val_lr)[:,1]
        else:
            model.fit(X_train_full, y_train)
            pred = model.predict_proba(X_val_full)[:,1]
        auc = roc_auc_score(y_val, pred)
        val_results.append({'Model': name, 'Validation AUC': auc})
        val_predictions[name] = pred
        print(f"{name} Validation AUC: {auc:.4f}")

    # Perform DeLong's test between models
    print("\nDeLong's Test Results (p-values):")
    print("--------------------------------")
    model_names = list(models.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            p_value = delong_test(y_val, val_predictions[model1], val_predictions[model2])
            sign = "=" if p_value >= 0.05 else (">" if val_predictions[model1].mean() > val_predictions[model2].mean() else "<")
            print(f"{model1} {sign} {model2}: {p_value:.4f}")

    # Ablation (LR)
    print("\nAblation Study Results:")
    print("--------------------")
    for group_name, cols in [('Vote-Graph Only', vote_graph),
                            ('Candidate-History Only', cand_history),
                            ('Temporal Only', temporal),
                            ('All Features', full)]:
        sc = StandardScaler().fit(df_train[cols])
        X_tr = sc.transform(df_train[cols])
        X_vl = sc.transform(df_val[cols])
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_tr, y_train)
        pred = lr.predict_proba(X_vl)[:,1]
        auc = roc_auc_score(y_val, pred)
        print(f"{group_name} AUC: {auc:.4f}")

    # Final test evaluation
    best_model_name = max(val_results, key=lambda x: x['Validation AUC'])['Model']
    best_model = models[best_model_name]
    if best_model_name == 'Logistic Regression':
        pred_test = best_model.predict_proba(X_test_lr)[:,1]
    else:
        pred_test = best_model.predict_proba(X_test_full)[:,1]
    test_auc = roc_auc_score(y_test, pred_test)
    print(f"\nBest Model ({best_model_name}) Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()

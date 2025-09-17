from snippet import db_pd, del_outlier
from sklearn.model_selection import KFold
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt

kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 分割用
df = db_pd(
    """select 
            `serial`,SUBSTRING(zip1, 1, 1) AS area,datediff(CURDATE(), open_date) as since,base_monthly_price,item_variety,options,total 
        from dami2.seller_master 
        inner join (select m_seller_id,sum(kingaku) as total from dami2.order_dat_m where order_time is not null group by m_seller_id) as orders on `id`=m_seller_id 
        inner join (select m_id,count(*) as item_variety from dami2.m_data_dat where m_koukai_flg=1 group by m_id) as items on `id`=m_id 
        left join (select seller_id,sum(paid_amount)as options from accounting.option_master where del_flg=0 group by seller_id) as option_master on `id`=seller_id
        where status=2 and del_flg=0""",
    (),
)

params = {  # lightgbm
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "random_state": 42,
}


if df["status"] == "ok" or df["status"] == "fallback":
    print(df["data"].head)
    sales_since = del_outlier(df["data"], "total")
    sales_since.total = np.log1p(sales_since.total)  # 対数化
    X = sales_since.drop(columns=["total", "area", "serial"])
    area_dummies = pd.get_dummies(sales_since["area"], prefix="area")
    X = pd.concat([X, area_dummies], axis=1)

    X = X.replace("", float("nan")).fillna(0)
    y = sales_since.total

    # model = RandomForestRegressor(n_estimators=100, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
        )

        rmse_log = root_mean_squared_error(
            y_val, model.predict(X_val)
        )  # 対数化した値でのrmse
        print(f"[Fold {fold}] RMSE(log-scale): {rmse_log:.4f}")
        if fold == 2:
            y_pred = np.expm1(model.predict(X_val))
            y_true = np.expm1(y_val)
            print(X_val.head)
            plt.figure(figsize=(12, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot(
                [y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                color="red",
                linestyle="--",
            )
            plt.xlabel("Actual total sales")
            plt.ylabel("Predicted total sales")
            # plt.plot(y_true.values, label="Actual Total Sales", marker="o")
            # plt.plot(y_pred, label="Predicted Total Sales", marker="x")
            # plt.xlabel("Sample Index (Validation Set)")
            # plt.ylabel("Total Sales")
            plt.title(f"Fold {fold} Actual vs Predicted")
            plt.grid(True)
            plt.savefig("gbm.png")
    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,  # 適宜調整
    )

    new_data = pd.DataFrame(
        {
            "since": [720],
            "base_monthly_price": [35000],
            "item_variety": [30],
            "options": [0],
            "area_": [False],
            "area_0": [False],
            "area_1": [True],
            "area_2": [False],
            "area_3": [False],
            "area_4": [False],
            "area_5": [False],
            "area_6": [False],
            "area_7": [False],
            "area_8": [False],
            "area_9": [False],
        }
    )
    y_pred_log = model.predict(new_data)

    # 対数変換してたら戻す
    y_pred = np.expm1(y_pred_log)

    print(f"予測売上: {y_pred[0]}")
    importances = model.feature_importance()
    feature_names = model.feature_name()

    # 特徴量と重要度をDataFrame化
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values(by="importance", ascending=False)

    # 上位20特徴量だけ表示（調整可）
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["feature"][:20][::-1], importance_df["importance"][:20][::-1]
    )
    plt.xlabel("Feature Importance")
    plt.title("Top 20 Feature Importances (LightGBM)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("feature_importance.png")  # 保存
else:
    print(df["error"])

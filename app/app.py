"""Store Revenue Forecast & Planning Studio - ứng dụng web Flask.
Chạy:  python -m app.app   ->  http://127.0.0.1:5000
Ứng dụng chỉ NẠP mô hình đã lưu (models/forecaster.joblib); không huấn luyện lại.
"""
import base64
import io
import json
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config
from src.data.loader import DataNotFoundError, load_history
from src.models.predict import Forecaster, InputError
from src.models.whatif import four_scenarios, run_what_if

app = Flask(__name__)
app.jinja_env.filters["eur"] = lambda v: "—" if v is None else f"{v:,.0f} €".replace(",", ".")
app.jinja_env.filters["eur_signed"] = lambda v: "—" if v is None else f"{v:+,.0f} €".replace(",", ".")
app.jinja_env.filters["pct"] = lambda v: "không xác định (doanh thu ban đầu = 0)" if v is None else f"{v:+.2f}%"


@lru_cache(maxsize=1)
def get_forecaster():
    return Forecaster()


@lru_cache(maxsize=1)
def get_history():
    try:
        return load_history()
    except DataNotFoundError:
        return None


def png_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def table_html(df):
    """HTML bảng (định dạng số 4 chữ số thập phân); None nếu chưa có file."""
    if df is None:
        return None
    return df.to_html(index=False, border=0, float_format=lambda x: f"{x:.4f}" if abs(x) < 10 else f"{x:,.2f}")


def read_csv_safe(name):
    p = config.RESULTS_DIR / name
    return pd.read_csv(p, encoding="utf-8-sig") if p.exists() else None


def defaults_from_history(store, date):
    """Kịch bản ban đầu: lấy Promo/StateHoliday/SchoolHoliday có trong dữ liệu cho (Store, ngày); nếu không có -> mặc định."""
    h = get_history()
    info = {"source": "default", "open": None, "actual_sales": None,
            "promo": 0, "state_holiday": "0", "school_holiday": 0}
    if h is None:
        return info
    r = h[(h["Store"] == store) & (h["Date"] == pd.Timestamp(date))]
    if len(r):
        r = r.iloc[0]
        info.update(source="dữ liệu lịch sử/test.csv", open=int(r["Open"]),
                    promo=int(r["Promo"]), state_holiday=str(r["StateHoliday"]),
                    school_holiday=int(r["SchoolHoliday"]))
        if pd.notna(r.get("Sales")):
            info["actual_sales"] = float(r["Sales"])
    return info


def form_values(src):
    return {"store": src.get("store", "1"), "date": src.get("date", "2015-08-15"),
            "promo": src.get("promo", "0"), "state_holiday": src.get("state_holiday", "0"),
            "school_holiday": src.get("school_holiday", "0"),
            "new_promo": src.get("new_promo", "1"), "new_holiday": src.get("new_holiday", "0")}


def date_warning(dt):
    if dt > pd.Timestamp(config.DATA_LAST_DATE):
        return (f"Ngày chọn nằm sau {config.DATA_LAST_DATE} (ngoài giai đoạn dữ liệu Rossmann); "
                "mô hình ngoại suy theo xu hướng đã học, độ tin cậy thấp.")
    return None


def history_chart(store, date, forecast_value=None, title_extra=""):
    h = get_history()
    if h is None:
        return None
    s = h[(h["Store"] == store) & (h["Sales"].notna()) & (h["Open"] == 1) & (h["Sales"] > 0)]
    dt = pd.Timestamp(date)
    w = s[(s["Date"] >= dt - pd.Timedelta(days=60)) & (s["Date"] <= dt + pd.Timedelta(days=30))]
    if w.empty:
        w = s.tail(90)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(w["Date"], w["Sales"], color="#2563eb", marker=".", lw=1, label="Doanh thu thực tế")
    if forecast_value is not None:
        ax.scatter([dt], [forecast_value], color="#dc2626", s=70, zorder=5, label="Doanh thu dự báo")
    ax.set_title(f"Cửa hàng {store}: doanh thu theo thời gian {title_extra}")
    ax.set_ylabel("Doanh thu (€)")
    ax.legend(); ax.grid(alpha=.3); fig.autofmt_xdate()
    return png_b64(fig)


def whatif_chart(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1.3]})
    ax = axes[0]
    bars = ax.bar(["Kịch bản ban đầu", "Kịch bản What-if"], [res["base_sales"], res["new_sales"]],
                  color=["#64748b", "#16a34a" if res["diff"] >= 0 else "#dc2626"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():,.0f} €", ha="center", va="bottom")
    ax.set_ylabel("Doanh thu dự báo (€)"); ax.set_title("Doanh thu dự báo")
    ax2 = axes[1]
    ax2.bar(["Chênh lệch (€)"], [res["diff"]], color="#f59e0b")
    ax2.axhline(0, color="k", lw=.8)
    pct = "n/a" if res["pct"] is None else f"{res['pct']:+.2f}%"
    ax2.set_title(f"Chênh lệch: {res['diff']:+,.0f} €  ({pct})", fontsize=10)
    fig.tight_layout()
    return png_b64(fig)


def four_chart(rows):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    labels = [r["label"].replace(": ", ":\n") for r in rows]
    bars = ax.bar(labels, [r["sales"] for r in rows], color=["#64748b", "#16a34a", "#f59e0b", "#2563eb"])
    for b, r in zip(bars, rows):
        pct = "" if r["pct_vs_kb1"] is None or r is rows[0] else f"\n{r['pct_vs_kb1']:+.1f}% so với KB1"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():,.0f} €{pct}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Doanh thu dự báo (€)"); ax.set_title("So sánh 4 kịch bản (cùng cửa hàng, cùng ngày)")
    ax.set_ylim(0, max(r["sales"] for r in rows) * 1.25)
    fig.tight_layout()
    return png_b64(fig)


@app.route("/")
def overview():
    kpi = {}
    p = config.RESULTS_DIR / "dashboard_kpi.json"
    if p.exists():
        kpi = json.loads(p.read_text(encoding="utf-8"))
    comp = read_csv_safe("model_comparison.csv")
    chart = None
    h = get_history()
    if h is not None:
        m = h[(h["Sales"].notna()) & (h["Open"] == 1)].groupby(h["Date"].dt.to_period("M"))["Sales"].sum() / 1e6
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(m.index.to_timestamp(), m.values, marker="o", color="#2563eb")
        ax.set_title("Tổng doanh thu theo tháng (triệu €, các ngày mở cửa)"); ax.grid(alpha=.3)
        chart = png_b64(fig)
    try:
        info = get_forecaster().info
    except FileNotFoundError as e:
        info = {"error": str(e)}
    return render_template("overview.html", kpi=kpi, comp=table_html(comp), chart=chart, info=info,
                           n_hist=None if h is None else len(h))


@app.route("/forecast", methods=["GET", "POST"])
def forecast():
    vals = form_values(request.form if request.method == "POST" else request.args)
    ctx = {"vals": vals, "holidays": config.HOLIDAY_LABELS}
    if request.method == "POST":
        try:
            fc = get_forecaster()
            sc = fc.validate(vals["store"], vals["date"], vals["promo"], vals["state_holiday"], vals["school_holiday"])
            val, feats = fc.predict_one(sc)
            hist = defaults_from_history(sc.store, sc.date)
            ctx.update(result=val, scenario=sc, feats=feats[fc.bundle.features].to_dict(),
                       hist=hist, warn=date_warning(sc.date),
                       chart=history_chart(sc.store, sc.date, val), info=fc.info)
        except (InputError, FileNotFoundError) as e:
            ctx["error"] = str(e)
    return render_template("forecast.html", **ctx)


@app.route("/whatif", methods=["GET", "POST"])
def whatif():
    vals = form_values(request.form if request.method == "POST" else request.args)
    action = request.form.get("action", "")
    ctx = {"vals": vals, "holidays": config.HOLIDAY_LABELS, "step": 1}
    if request.method == "POST":
        try:
            fc = get_forecaster()
            sc0 = fc.validate(vals["store"], vals["date"], 0, "0", 0)
            if action == "load":  # Bước 3: hiển thị kịch bản ban đầu lấy từ dữ liệu
                d = defaults_from_history(sc0.store, sc0.date)
                vals.update(promo=str(d["promo"]), state_holiday=d["state_holiday"], school_holiday=str(d["school_holiday"]),
                            new_promo=str(1 - d["promo"]), new_holiday=d["state_holiday"])
                ctx.update(step=2, hist=d, warn=date_warning(sc0.date))
            else:  # Bước 6-9: chạy What-if
                base = fc.validate(vals["store"], vals["date"], vals["promo"], vals["state_holiday"], vals["school_holiday"])
                new = fc.validate(vals["store"], vals["date"], vals["new_promo"], vals["new_holiday"], vals["school_holiday"])
                res = run_what_if(fc, base, promo=new.promo, state_holiday=new.state_holiday)
                rows = four_scenarios(fc, base, holiday_code=new.state_holiday if new.state_holiday != "0" else "a")
                ctx.update(step=3, res=res, chart=whatif_chart(res), four=rows, four_chart=four_chart(rows),
                           hist=defaults_from_history(base.store, base.date), warn=date_warning(base.date), info=fc.info)
        except (InputError, FileNotFoundError) as e:
            ctx["error"] = str(e)
    return render_template("whatif.html", **ctx)


@app.route("/analysis")
def analysis():
    figs = sorted(p.name for p in config.FIGURES_DIR.glob("hinh4_*.png"))
    return render_template("analysis.html", figs=figs, comp=table_html(read_csv_safe("model_comparison.csv")),
                           run=table_html(read_csv_safe("model_comparison_run.csv")),
                           by_promo=table_html(read_csv_safe("error_by_Promo.csv")),
                           by_dow=table_html(read_csv_safe("error_by_DayOfWeek.csv")),
                           by_type=table_html(read_csv_safe("error_by_StoreType.csv")))


@app.route("/figures/<name>")
def figure(name):
    from flask import send_from_directory, abort
    if not name.startswith("hinh4_") or not name.endswith(".png"):
        abort(404)
    return send_from_directory(config.FIGURES_DIR, name)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    j = request.get_json(silent=True) or {}
    try:
        fc = get_forecaster()
        sc = fc.validate(j.get("store"), j.get("date"), j.get("promo", 0), j.get("state_holiday", "0"), j.get("school_holiday", 0))
        val, _ = fc.predict_one(sc)
        return jsonify({"store": sc.store, "date": str(sc.date.date()), "predicted_sales": round(val, 2)})
    except InputError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    j = request.get_json(silent=True) or {}
    try:
        fc = get_forecaster()
        base = fc.validate(j.get("store"), j.get("date"), j.get("promo", 0), j.get("state_holiday", "0"), j.get("school_holiday", 0))
        new = fc.validate(j.get("store"), j.get("date"), j.get("new_promo", base.promo), j.get("new_state_holiday", base.state_holiday), base.school_holiday)
        r = run_what_if(fc, base, promo=new.promo, state_holiday=new.state_holiday)
        return jsonify({"base_sales": round(r["base_sales"], 2), "new_sales": round(r["new_sales"], 2),
                        "diff": round(r["diff"], 2), "pct": None if r["pct"] is None else round(r["pct"], 4), "notes": r["notes"]})
    except InputError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

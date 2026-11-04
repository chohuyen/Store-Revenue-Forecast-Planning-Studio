"""What-if Scenario: đổi Promo / StateHoliday, GIỮ NGUYÊN mọi đầu vào khác, dự báo lại bằng mô hình đã lưu."""
from dataclasses import replace
from src import config
from src.models.predict import Forecaster, Scenario


def compare(base_value, new_value):
    """Chênh lệch tuyệt đối và tỷ lệ % (None nếu doanh thu ban đầu = 0 để tránh chia cho 0)."""
    diff = float(new_value) - float(base_value)
    pct = None if float(base_value) == 0 else diff / float(base_value) * 100.0
    return diff, pct


def run_what_if(fc: Forecaster, base: Scenario, promo=None, state_holiday=None):
    """Chỉ thay biến người dùng chọn (promo, state_holiday)."""
    new = replace(base, promo=base.promo if promo is None else int(promo),
                  state_holiday=base.state_holiday if state_holiday is None else str(state_holiday))
    (p_base, p_new), _ = fc.predict_scenarios([base, new])
    diff, pct = compare(p_base, p_new)
    notes = []
    if new.state_holiday != base.state_holiday and not fc.bundle.supports_holiday_type \
            and new.state_holiday != "0" and base.state_holiday != "0":
        notes.append("Mô hình hiện tại chỉ phân biệt 'có/không có ngày lễ', không phân biệt a/b/c.")
    if fc.bundle.supports_holiday_type is False and new.state_holiday not in ("0",) and base.state_holiday == "0":
        notes.append("Mô hình hiện tại chỉ biết 'có ngày lễ', không phân biệt loại a/b/c.")
    return {"base": base, "new": new, "base_sales": float(p_base), "new_sales": float(p_new),
            "diff": diff, "pct": pct, "notes": notes}


def four_scenarios(fc: Forecaster, base: Scenario, holiday_code="a"):
    """4 kịch bản cùng Store/ngày/SchoolHoliday: (Promo 0/1) x (không lễ / có lễ holiday_code)."""
    combos = [(0, "0", "KB1: Không KM, không lễ"), (1, "0", "KB2: Có KM, không lễ"),
              (0, holiday_code, f"KB3: Không KM, lễ '{holiday_code}'"),
              (1, holiday_code, f"KB4: Có KM, lễ '{holiday_code}'")]
    scs = [replace(base, promo=p, state_holiday=h) for p, h, _ in combos]
    preds, _ = fc.predict_scenarios(scs)
    ref = preds[0]
    out = []
    for (p, h, label), val in zip(combos, preds):
        d, pct = compare(ref, val)
        out.append({"label": label, "promo": p, "state_holiday": h, "sales": float(val), "diff_vs_kb1": d, "pct_vs_kb1": pct})
    return out
